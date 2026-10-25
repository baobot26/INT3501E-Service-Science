"""Independent validation of a main-model solution.

Recomputes nearest-station assignment from the distance matrix and selected
stations; does NOT trust the solver's assignment variables as ground truth.
"""
from __future__ import annotations

import numpy as np

from src.preprocessing import ProblemData

TOL = 1e-6


def validate_solution(problem: ProblemData, dist: np.ndarray,
                      selected_idx: np.ndarray, assignment_idx: np.ndarray,
                      station_count: int, total_cost: float) -> dict:
    """Return a validation dict with per-check booleans, metrics, and errors."""
    n = len(problem.demand_ids)
    errors: list[dict] = []

    # 1. selected indices valid
    selected_valid = bool(
        selected_idx.size == 0 or
        (selected_idx.min() >= 0 and selected_idx.max() < len(problem.candidate_ids)))
    if not selected_valid:
        errors.append({"check": "candidate_ids_valid", "detail": "index out of range"})

    # 2. independent nearest assignment among selected stations
    expected = np.full(n, -1, dtype=int)
    expected_dist = np.full(n, np.inf)
    if selected_idx.size:
        sub = dist[:, selected_idx]  # (n, len(selected))
        for i in range(n):
            d = sub[i]
            within = np.where(d <= problem.radius)[0]
            if within.size:
                ids = problem.candidate_ids[selected_idx[within]]
                order = np.lexsort((ids, d[within]))
                best = within[order[0]]
                expected[i] = selected_idx[best]
                expected_dist[i] = d[best]

    served = expected >= 0
    served_demand = float(problem.daily_demand[served].sum())
    total_demand = float(problem.daily_demand.sum())
    coverage = served_demand / total_demand if total_demand > 0 else 0.0

    # 3. solver assignment matches independent nearest
    assignment_match = True
    for i in range(n):
        if served[i] and assignment_idx[i] != expected[i]:
            assignment_match = False
            errors.append({"check": "assignment_nearest", "demand_id": int(problem.demand_ids[i]),
                           "solver": int(assignment_idx[i]), "expected": int(expected[i])})
            if len(errors) > 20:
                break

    # 4. coverage
    coverage_ok = coverage + TOL >= problem.alpha
    if not coverage_ok:
        errors.append({"check": "coverage", "coverage_pct": round(coverage * 100, 4),
                       "required_pct": round(problem.alpha * 100, 4)})

    # 5. capacity per station
    capacity_ok = True
    loads = np.zeros(len(problem.candidate_ids))
    for i in range(n):
        if assignment_idx[i] >= 0:
            loads[assignment_idx[i]] += problem.daily_demand[i]
    for j in range(len(problem.candidate_ids)):
        if loads[j] > problem.capacities[j] + TOL:
            capacity_ok = False
            errors.append({"check": "capacity", "station_id": int(problem.candidate_ids[j]),
                           "load": float(loads[j]), "capacity": float(problem.capacities[j])})

    # 6. objective consistency
    objective_ok = (len(selected_idx) == station_count and
                    abs(float(problem.fixed_costs[selected_idx].sum()) - total_cost) < 1.0
                    if selected_idx.size else station_count == 0)
    if not objective_ok:
        errors.append({"check": "objective_consistent"})

    checks = {
        "candidate_ids_valid": selected_valid,
        "assignment_nearest": assignment_match,
        "coverage": coverage_ok,
        "capacity": capacity_ok,
        "objective_consistent": objective_ok,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "metrics": {
            "selected_stations": int(len(selected_idx)),
            "served_demand": served_demand,
            "total_demand": total_demand,
            "coverage_pct": round(coverage * 100, 4),
            "total_cost": total_cost,
        },
        "loads": loads.tolist(),
        "errors": errors,
    }