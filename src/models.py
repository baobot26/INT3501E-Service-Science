"""Ba mô hình chọn vị trí đặt tủ: set covering, maximal covering, p-median.

Đọc data/dist.npy, data/demand.csv, data/candidates.csv.
Ghi results/results.csv, mỗi dòng là một lời giải.

Set covering và maximal covering giải đúng bằng MILP (scipy, solver HiGHS).
P-median có 1544 x 481 biến gán nên dùng heuristic hoán đổi (Teitz-Bart)
chạy nhiều lần từ các điểm xuất phát khác nhau.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity

DATA = Path("data")
OUT = Path("results")

RADII = [300, 500, 800, 1000]   # bán kính phục vụ (m)
P_VALUES = list(range(3, 16))   # số tủ
BIG = 10_000                    # khoảng cách thay cho ô không đi tới được
TIME_LIMIT = 30                 # giây tối đa cho mỗi bài MILP
RESTARTS = 10                   # số lần chạy p-median từ điểm xuất phát ngẫu nhiên


def set_covering(D, R):
    """Ít tủ nhất để phủ mọi điểm nhu cầu có thể phủ được trong bán kính R."""
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
        raise RuntimeError(f"set covering R={R}: không có lời giải ({res.message})")
    return np.flatnonzero(res.x > 0.5), coverable, res.status


def max_covering(D, pop, R, p):
    """Chọn tối đa p tủ để phủ được nhiều dân nhất trong bán kính R."""
    n, m = D.shape
    cover = csr_matrix((D <= R).astype(float))
    c = np.concatenate([np.zeros(m), -pop])
    a_cover = hstack([-cover, identity(n)], format="csr")  # z_i <= tổng y_j quanh điểm i
    a_count = csr_matrix(np.concatenate([np.ones(m), np.zeros(n)])[None, :])
    res = milp(
        c=c,
        constraints=[LinearConstraint(a_cover, -np.inf, 0), LinearConstraint(a_count, 0, p)],
        integrality=np.concatenate([np.ones(m), np.zeros(n)]),
        bounds=Bounds(0, 1),
        options={"time_limit": TIME_LIMIT, "mip_rel_gap": 0.001},
    )
    if res.x is None:
        raise RuntimeError(f"max covering R={R} p={p}: không có lời giải ({res.message})")
    return np.flatnonzero(res.x[:m] > 0.5), res.status


def _swap_search(Dp, pop, S):
    """Hoán đổi từng tủ với mọi điểm ứng viên cho tới khi hết cải thiện được."""
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
    """Chọn p tủ để tổng quãng đường đi bộ (nhân dân số) ngắn nhất."""
    Dp = np.where(np.isfinite(D), D, BIG)
    n, m = Dp.shape

    # điểm xuất phát 1: tham lam, mỗi bước thêm tủ giảm chi phí nhiều nhất
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


def evaluate(D, pop, S):
    """Các chỉ số chung để so sánh ba mô hình trên cùng một thước đo."""
    nearest = D[:, S].min(axis=1)
    out = {f"cov_{r}": round(pop[nearest <= r].sum() / pop.sum() * 100, 2) for r in RADII}
    ok = np.isfinite(nearest)
    out["avg_dist_m"] = round((pop[ok] * nearest[ok]).sum() / pop[ok].sum(), 1)
    out["max_dist_m"] = round(nearest[ok].max(), 1)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    D = np.load(DATA / "dist.npy").astype(float)
    pop = pd.read_csv(DATA / "demand.csv")["population"].to_numpy(float)
    ids = pd.read_csv(DATA / "candidates.csv")["id"].to_numpy()
    rng = np.random.default_rng(0)
    rows = []

    def add(model, R, S, t0, note=""):
        row = {"model": model, "R": R, "p": len(S), "stations": " ".join(str(ids[j]) for j in S)}
        row.update(evaluate(D, pop, S))
        row["seconds"] = round(time.time() - t0, 1)
        row["note"] = note
        rows.append(row)
        print(f"{model:13s} R={R!s:>5} p={len(S):2d}  phủ@R={row.get(f'cov_{R}', '-')}  "
              f"tb={row['avg_dist_m']} m  ({row['seconds']}s) {note}")

    for R in RADII:
        t0 = time.time()
        S, coverable, status = set_covering(D, R)
        miss = pop[~coverable].sum() / pop.sum() * 100
        note = f"{miss:.1f}% dân không có điểm ứng viên nào trong tầm"
        if status != 0:
            note += "; chưa chắc tối ưu (hết thời gian)"
        add("set_covering", R, S, t0, note)

    for R in RADII:
        for p in P_VALUES:
            t0 = time.time()
            S, status = max_covering(D, pop, R, p)
            add("max_covering", R, S, t0, "" if status == 0 else "chưa chắc tối ưu (hết thời gian)")

    for p in P_VALUES:
        t0 = time.time()
        add("p_median", "", p_median(D, pop, p, rng), t0)

    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    print("đã lưu results/results.csv")


if __name__ == "__main__":
    main()
