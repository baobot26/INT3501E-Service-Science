"""Three locker-placement models: set covering, maximal covering, p-median.

Default (`--metric walking`): reads data/dist.npy, data/demand.csv, data/candidates.csv
and writes results/results.csv, one row per solution.
With `--metric l1`: reads data/dist_l1.npy, data/dist_l1_manifest.json, data/demand.csv,
data/candidates_feasible.csv, data/station_parameters.csv and writes results/results_l1.csv.

Set covering and maximal covering are solved exactly with MILP (scipy, HiGHS solver).
P-median has 1544 x 481 assignment variables, so it uses a swap heuristic (Teitz-Bart)
run several times from different starting points.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity

DATA = Path("data")
OUT = Path("results")

RADII = [300, 500, 800, 1000]   # service radius (m)
P_VALUES = list(range(3, 16))   # number of lockers
BIG = 10_000                    # stand-in distance for unreachable cells
TIME_LIMIT = 30                 # max seconds per MILP
RESTARTS = 10                   # p-median restarts from random starting points


def set_covering(D, R):
    """Fewest lockers covering every demand point that is coverable within radius R."""
    cover = D <= R
    coverable = cover.any(axis=1)
    A = csr_matrix(cover[coverable].astype(float))
    m = D.shape[1]
    res = milp(
        c=np.ones(m),
        constraints=LinearConstraint(A, lb=1, ub=np.inf),
        integrality=np.ones(m),
        bounds=Bounds(0, 1),
        options={"time_limit": TIME_LIMIT},
    )
    if res.x is None:
        raise RuntimeError(f"set covering R={R}: no solution ({res.message})")
    return np.flatnonzero(res.x > 0.5), coverable, res.status


def max_covering(D, pop, R, p):
    """Pick at most p lockers to cover as much population as possible within radius R."""
    n, m = D.shape
    cover = csr_matrix((D <= R).astype(float))
    c = np.concatenate([np.zeros(m), -pop])
    a_cover = hstack([-cover, identity(n)], format="csr")  # z_i <= sum of y_j around point i
    a_count = csr_matrix(np.concatenate([np.ones(m), np.zeros(n)])[None, :])
    res = milp(
        c=c,
        constraints=[LinearConstraint(a_cover, -np.inf, 0), LinearConstraint(a_count, 0, p)],
        integrality=np.concatenate([np.ones(m), np.zeros(n)]),
        bounds=Bounds(0, 1),
        options={"time_limit": TIME_LIMIT, "mip_rel_gap": 0.001},
    )
    if res.x is None:
        raise RuntimeError(f"max covering R={R} p={p}: no solution ({res.message})")
    return np.flatnonzero(res.x[:m] > 0.5), res.status


def _swap_search(Dp, pop, S):
    """Swap each locker with every candidate until no improvement remains."""
    S = list(S)
    cost = (pop * Dp[:, S].min(axis=1)).sum()
    improved = True
    while improved:
        improved = False
        for pos in range(len(S)):
            rest = S[:pos] + S[pos + 1:]
            base = Dp[:, rest].min(axis=1) if rest else np.full(len(pop), np.inf)
            costs = (pop[:, None] * np.minimum(base[:, None], Dp)).sum(axis=0)
            costs[rest] = np.inf
            j = int(costs.argmin())
            if costs[j] < cost - 1e-9:
                S[pos] = j
                cost = costs[j]
                improved = True
    return S, cost


def p_median(D, pop, p, rng):
    """Pick p lockers minimizing total population-weighted walking distance."""
    Dp = np.where(np.isfinite(D), D, BIG)
    n, m = Dp.shape

    # start 1: greedy, each step adds the locker that cuts cost the most
    S, base = [], np.full(n, np.inf)
    for _ in range(p):
        costs = (pop[:, None] * np.minimum(base[:, None], Dp)).sum(axis=0)
        costs[S] = np.inf
        j = int(costs.argmin())
        S.append(j)
        base = np.minimum(base, Dp[:, j])
    starts = [S] + [list(rng.choice(m, size=p, replace=False)) for _ in range(RESTARTS)]

    best_S, best_cost = None, np.inf
    for s in starts:
        S, cost = _swap_search(Dp, pop, s)
        if cost < best_cost:
            best_S, best_cost = S, cost
    return np.array(sorted(best_S))


def evaluate(D, pop, S, radii=RADII):
    """Shared metrics to compare the three models on the same scale."""
    nearest = D[:, S].min(axis=1)
    out = {f"cov_{r}": round(pop[nearest <= r].sum() / pop.sum() * 100, 2) for r in radii}
    ok = np.isfinite(nearest)
    out["avg_dist_m"] = round((pop[ok] * nearest[ok]).sum() / pop[ok].sum(), 1)
    out["max_dist_m"] = round(nearest[ok].max(), 1)
    return out


def enabled_columns(
    ids,
    cand_df: pd.DataFrame,
    params_df: pd.DataFrame,
    enabled_override: dict[str, bool] | None = None,
) -> np.ndarray:
    """Return boolean mask over ``ids`` where the candidate's station type is enabled."""
    type_enabled = {
        row["type"]: int(row["enabled"]) == 1
        for _, row in params_df.iterrows()
    }
    if enabled_override is not None:
        for stype, flag in enabled_override.items():
            type_enabled[stype] = bool(flag)
    id_to_type = {int(cid): stype for cid, stype in zip(cand_df["id"], cand_df["type"])}
    return np.array(
        [type_enabled.get(id_to_type.get(int(cid)), False) for cid in ids],
        dtype=bool,
    )


def run_baseline(
    model: str,
    D,
    pop,
    R: float,
    p: int | None = None,
    seed: int = 0,
) -> dict:
    """Run one baseline model on an already column-filtered distance matrix ``D``."""
    valid_models = {"set_covering", "max_covering", "p_median"}
    if model not in valid_models:
        raise ValueError(
            f"Mô hình không hợp lệ: {model!r}. "
            "Chỉ hỗ trợ set_covering, max_covering hoặc p_median."
        )

    D_arr = np.asarray(D, dtype=float)
    pop_arr = np.asarray(pop, dtype=float)
    if D_arr.ndim != 2 or D_arr.shape[1] == 0:
        raise ValueError(
            "Không có điểm ứng viên khả dụng nào (0 cột). "
            "Hãy bật ít nhất một loại trạm trước khi chạy."
        )
    if D_arr.shape[0] == 0 or pop_arr.ndim != 1 or pop_arr.shape[0] != D_arr.shape[0]:
        raise ValueError(
            f"Dữ liệu dân số ({pop_arr.shape}) không khớp với số điểm cầu của ma trận ({D_arr.shape})."
        )
    if not np.isfinite(pop_arr).all() or (pop_arr < 0).any() or float(pop_arr.sum()) <= 0:
        raise ValueError("Dữ liệu dân số phải không âm, hữu hạn và có tổng lớn hơn 0.")
    if isinstance(R, bool) or not isinstance(R, (int, float, np.number)) or not np.isfinite(R) or R < 0:
        raise ValueError(f"Bán kính phục vụ R = {R!r} không hợp lệ: cần là số không âm hữu hạn.")

    m = D_arr.shape[1]
    p_int: int | None = None
    if model in {"max_covering", "p_median"}:
        if p is None or isinstance(p, bool) or not isinstance(p, (int, np.integer)):
            raise ValueError(
                f"Mô hình {model} yêu cầu số trạm p là số nguyên trong khoảng từ 1 đến {m}."
            )
        p_int = int(p)
        if p_int < 1 or p_int > m:
            raise ValueError(
                f"Số trạm p = {p_int} không hợp lệ: cần nằm trong khoảng từ 1 đến {m} "
                "(số điểm ứng viên đang bật)."
            )

    t0 = time.perf_counter()
    total_pop = float(pop_arr.sum())
    coverable_pct: float | None = None

    if model == "set_covering":
        selected, coverable, raw_status = set_covering(D_arr, R)
        status: int | None = int(raw_status)
        status_label = "optimal" if status == 0 else "time_limit"
        coverable_pct = (
            round(float(pop_arr[coverable].sum() / total_pop * 100), 2)
            if total_pop > 0
            else 0.0
        )
    elif model == "max_covering":
        selected, raw_status = max_covering(D_arr, pop_arr, R, p_int)
        status = int(raw_status)
        status_label = "optimal" if status == 0 else "time_limit"
    else:
        rng = np.random.default_rng(seed)
        selected = p_median(D_arr, pop_arr, p_int, rng)
        status = None
        status_label = "heuristic"

    seconds = float(time.perf_counter() - t0)
    selected_arr = np.sort(np.asarray(selected, dtype=int))

    if len(selected_arr) == 0:
        metrics = {
            "coverage_pct": 0.0,
            "avg_dist_m": None,
            "max_dist_m": None,
            "coverable_pct": coverable_pct,
        }
    else:
        nearest = D_arr[:, selected_arr].min(axis=1)
        ok = np.isfinite(nearest)
        if ok.any() and pop_arr[ok].sum() > 0:
            ev = evaluate(D_arr, pop_arr, selected_arr, radii=[R])
            coverage_pct = float(ev[f"cov_{R}"])
            avg_dist_m: float | None = float(ev["avg_dist_m"])
            max_dist_m: float | None = float(ev["max_dist_m"])
        else:
            coverage_pct = (
                round(float(pop_arr[nearest <= R].sum() / total_pop * 100), 2)
                if total_pop > 0
                else 0.0
            )
            avg_dist_m = None
            max_dist_m = None
        metrics = {
            "coverage_pct": coverage_pct,
            "avg_dist_m": avg_dist_m,
            "max_dist_m": max_dist_m,
            "coverable_pct": coverable_pct,
        }

    return {
        "model": model,
        "selected": selected_arr,
        "status": status,
        "status_label": status_label,
        "seconds": seconds,
        "metrics": metrics,
    }


def load_inputs(metric: str = "walking", data_dir: Path = DATA):
    """Load (D, pop, ids) for either walking-network or L1 feasible candidates."""
    data_dir = Path(data_dir)
    pop = pd.read_csv(data_dir / "demand.csv")["population"].to_numpy(float)
    if metric == "walking":
        D = np.load(data_dir / "dist.npy").astype(float)
        ids = pd.read_csv(data_dir / "candidates.csv")["id"].to_numpy()
        return D, pop, ids
    if metric == "l1":
        D = np.load(data_dir / "dist_l1.npy")
        manifest = json.loads((data_dir / "dist_l1_manifest.json").read_text(encoding="utf-8"))
        ids = np.array(manifest["candidate_ids"])
        cand_df = pd.read_csv(data_dir / "candidates_feasible.csv")
        params_df = pd.read_csv(data_dir / "station_parameters.csv")
        mask = enabled_columns(ids, cand_df, params_df)
        return D[:, mask].astype(float), pop, ids[mask]
    raise ValueError(f"unsupported metric: {metric}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run 3 baseline locker-placement models")
    parser.add_argument("--metric", choices=["walking", "l1"], default="walking")
    args = parser.parse_args(argv)

    OUT.mkdir(exist_ok=True)
    D, pop, ids = load_inputs(args.metric, DATA)
    rng = np.random.default_rng(0)
    rows = []

    def add(model, R, S, t0, note=""):
        row = {"model": model, "R": R, "p": len(S), "stations": " ".join(str(ids[j]) for j in S)}
        row.update(evaluate(D, pop, S))
        row["seconds"] = round(time.time() - t0, 1)
        row["note"] = note
        rows.append(row)
        print(f"{model:13s} R={R!s:>5} p={len(S):2d}  cov@R={row.get(f'cov_{R}', '-')}  "
              f"avg={row['avg_dist_m']} m  ({row['seconds']}s) {note}")

    for R in RADII:
        t0 = time.time()
        S, coverable, status = set_covering(D, R)
        miss = pop[~coverable].sum() / pop.sum() * 100
        note = f"{miss:.1f}% of population has no candidate in range"
        if status != 0:
            note += "; not guaranteed optimal (time limit)"
        add("set_covering", R, S, t0, note)

    for R in RADII:
        for p in P_VALUES:
            t0 = time.time()
            S, status = max_covering(D, pop, R, p)
            add("max_covering", R, S, t0, "" if status == 0 else "not guaranteed optimal (time limit)")

    for p in P_VALUES:
        t0 = time.time()
        add("p_median", "", p_median(D, pop, p, rng), t0)

    out_name = "results_l1.csv" if args.metric == "l1" else "results.csv"
    out_path = OUT / out_name
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()