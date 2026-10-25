"""Two-phase MILP for the main model.

Phase 1: minimize number of opened stations.
Phase 2: with station count fixed to the phase-1 result, minimize total fixed cost.

Constraints:
  - each demand point assigned to at most one station;
  - assignment only to opened stations;
  - assignment to the nearest opened station (tie-break by candidate id);
  - station capacity;
  - coverage >= alpha * total demand.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from src.preprocessing import ProblemData


def build_assignment_neighbors(dist: np.ndarray, candidate_ids: np.ndarray,
                               radius: float) -> list[np.ndarray]:
    """For each demand point, candidate indices within radius sorted by (distance, id)."""
    neighbors = []
    for i in range(dist.shape[0]):
        idx = np.where(dist[i] <= radius)[0]
        if idx.size:
            order = np.lexsort((candidate_ids[idx], dist[i, idx]))
            idx = idx[order]
        neighbors.append(idx)
    return neighbors


def build_milp(problem: ProblemData, dist: np.ndarray,
               neighbors: list[np.ndarray]) -> dict:
    """Build the sparse constraint matrix. Variable layout: x_j (0..m-1), then y_ij."""
    m = len(problem.candidate_ids)
    n = len(problem.demand_ids)

    pairs: list[tuple[int, int]] = []
    pair_index: dict[tuple[int, int], int] = {}
    for i, js in enumerate(neighbors):
        for j in js:
            j = int(j)
            pair_index[(i, j)] = m + len(pairs)
            pairs.append((i, j))
    k = len(pairs)
    nvar = m + k

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    names: list[str] = []

    def add_row(name: str, coeffs: list[tuple[int, float]], lo: float, hi: float) -> None:
        r = len(lb)
        names.append(name)
        for c, v in coeffs:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        lb.append(lo)
        ub.append(hi)

    # each demand assigned at most once
    for i in range(n):
        if neighbors[i].size:
            add_row("assignment_limit",
                    [(pair_index[(i, int(j))], 1.0) for j in neighbors[i]],
                    -np.inf, 1.0)

    # assignment only to opened stations
    for (i, j), v in pair_index.items():
        add_row("opened_station_link", [(v, 1.0), (j, -1.0)], -np.inf, 0.0)

    # nearest-prefix: if j is open, demand i goes to j or an earlier (nearer) candidate
    for i, js in enumerate(neighbors):
        prefix: list[int] = []
        for j in js:
            j = int(j)
            prefix.append(pair_index[(i, j)])
            add_row("nearest_prefix",
                    [(j, 1.0)] + [(p, -1.0) for p in prefix],
                    -np.inf, 0.0)

    # capacity per station
    pairs_by_j: dict[int, list[tuple[int, int]]] = {}
    for i, j in pairs:
        pairs_by_j.setdefault(j, []).append((i, pair_index[(i, j)]))
    for j in range(m):
        coeffs = [(j, -problem.capacities[j])]
        coeffs += [(v, problem.daily_demand[i]) for i, v in pairs_by_j.get(j, [])]
        add_row("capacity", coeffs, -np.inf, 0.0)

    # coverage
    total_demand = float(problem.daily_demand.sum())
    add_row("coverage",
            [(pair_index[(i, j)], problem.daily_demand[i]) for i, j in pairs],
            problem.alpha * total_demand, np.inf)

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(len(lb), nvar)).tocsr()
    return {
        "A": A, "lb": np.array(lb), "ub": np.array(ub),
        "constraint_names": names,
        "pairs": pairs, "pair_index": pair_index,
        "nvar": nvar, "m": m, "k": k,
    }


def _classify(res, time_limited: bool) -> str:
    if res.status == 0:
        return "optimal"
    if res.status == 1 and res.x is not None:
        return "feasible_time_limit"
    if res.status == 2:
        return "infeasible"
    return "solver_error"


def solve(problem: ProblemData, dist: np.ndarray,
          time_limit: float = 300.0, mip_rel_gap: float = 0.0) -> dict:
    neighbors = build_assignment_neighbors(dist, problem.candidate_ids, problem.radius)
    model = build_milp(problem, dist, neighbors)
    m, nvar = model["m"], model["nvar"]
    integrality = np.ones(nvar)
    bounds = Bounds(np.zeros(nvar), np.ones(nvar))
    options = {"time_limit": time_limit, "mip_rel_gap": mip_rel_gap}
    base = LinearConstraint(model["A"], model["lb"], model["ub"])

    result: dict = {"neighbors": neighbors, "model": model}

    # --- phase 1: minimize station count ---
    c1 = np.zeros(nvar)
    c1[:m] = 1.0
    t0 = time.perf_counter()
    res1 = milp(c1, constraints=base, integrality=integrality,
                bounds=bounds, options=options)
    p1_seconds = time.perf_counter() - t0
    p1_status = _classify(res1, time_limited=(res1.status == 1))
    result.update(phase1_status=p1_status, phase1_seconds=round(p1_seconds, 3),
                  phase1_message=str(res1.message))

    if p1_status not in ("optimal", "feasible_time_limit"):
        result.update(status=p1_status, selected=np.array([], dtype=int),
                      assignment=np.full(len(problem.demand_ids), -1, dtype=int))
        return result

    n_stations = int(round(float(res1.x[:m].sum())))

    # --- phase 2: fix station count, minimize cost ---
    count_row = sparse.coo_matrix((np.ones(m), (np.zeros(m), np.arange(m))),
                                  shape=(1, nvar)).tocsr()
    A2 = sparse.vstack([model["A"], count_row]).tocsr()
    cons2 = LinearConstraint(
        A2,
        np.append(model["lb"], n_stations),
        np.append(model["ub"], n_stations),
    )
    c2 = np.zeros(nvar)
    c2[:m] = problem.fixed_costs
    t0 = time.perf_counter()
    res2 = milp(c2, constraints=cons2, integrality=integrality,
                bounds=bounds, options=options)
    p2_seconds = time.perf_counter() - t0
    p2_status = _classify(res2, time_limited=(res2.status == 1))
    result.update(phase2_status=p2_status, phase2_seconds=round(p2_seconds, 3),
                  phase2_message=str(res2.message))

    if p2_status not in ("optimal", "feasible_time_limit"):
        # fall back to the phase-1 incumbent
        x = res1.x
        status = p1_status
    else:
        x = res2.x
        status = "optimal" if p1_status == "optimal" and p2_status == "optimal" \
            else "feasible_time_limit"

    selected = np.where(x[:m] > 0.5)[0]
    assignment = np.full(len(problem.demand_ids), -1, dtype=int)
    for (i, j), v in model["pair_index"].items():
        if x[v] > 0.5:
            assignment[i] = j

    result.update(status=status, selected=selected, assignment=assignment,
                  station_count=len(selected),
                  total_cost=float(problem.fixed_costs[selected].sum()))
    return result