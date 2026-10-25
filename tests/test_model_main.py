"""Brute-force verification of the two-phase MILP on tiny instances."""
import itertools

import numpy as np
import pytest

from src.model_main import build_assignment_neighbors, solve
from src.preprocessing import ProblemData


def make_problem(rng, n, m, radius, alpha):
    dxy = rng.uniform(0, 2000, size=(n, 2))
    cxy = rng.uniform(0, 2000, size=(m, 2))
    return ProblemData(
        demand_ids=np.arange(n),
        candidate_ids=np.arange(m),
        demand_xy=dxy,
        candidate_xy=cxy,
        daily_demand=rng.uniform(20, 200, size=n),
        capacities=rng.uniform(100, 1500, size=m),
        fixed_costs=rng.integers(1, 100, size=m).astype(float),
        candidate_types=[f"t{j}" for j in range(m)],
        demand_lat=dxy[:, 1],
        demand_lon=dxy[:, 0],
        cand_lat=cxy[:, 1],
        cand_lon=cxy[:, 0],
        radius=radius,
        alpha=alpha,
        cap_scale=1.0,
    )


def l1(dxy, cxy):
    return np.abs(dxy[:, None, :] - cxy[None, :, :]).sum(axis=2)


def brute_force(problem, dist):
    """(station_count, cost) of the best feasible subset, or None if infeasible."""
    n, m = len(problem.demand_ids), len(problem.candidate_ids)
    required = problem.alpha * problem.daily_demand.sum()
    best = None
    for r in range(m + 1):
        for combo in itertools.combinations(range(m), r):
            loads = np.zeros(m)
            served = 0.0
            for i in range(n):
                js = [j for j in combo if dist[i, j] <= problem.radius]
                if not js:
                    continue
                j = min(js, key=lambda j: (dist[i, j], problem.candidate_ids[j]))
                loads[j] += problem.daily_demand[i]
                served += problem.daily_demand[i]
            if served < required - 1e-9 or (loads > problem.capacities + 1e-9).any():
                continue
            key = (len(combo), float(problem.fixed_costs[list(combo)].sum()))
            if best is None or key < best:
                best = key
    return best


def test_random_instances_match_brute_force():
    for seed in range(40):
        rng = np.random.default_rng(seed)
        problem = make_problem(rng, int(rng.integers(2, 6)), int(rng.integers(2, 7)),
                               float(rng.uniform(600, 1800)),
                               float(rng.choice([0.5, 0.8, 1.0])))
        dist = l1(problem.demand_xy, problem.candidate_xy)
        expected = brute_force(problem, dist)
        result = solve(problem, dist, time_limit=60.0)
        if expected is None:
            assert result["status"] == "infeasible", (seed, result["status"])
        else:
            assert result["status"] == "optimal", (seed, result["status"])
            assert result["station_count"] == expected[0], seed
            assert result["total_cost"] == pytest.approx(expected[1], abs=1e-6), seed


def test_neighbors_sorted_by_distance_then_id():
    dist = np.array([[100.0, 100.0, 50.0]])
    ids = np.array([7, 3, 9])
    nb = build_assignment_neighbors(dist, ids, radius=200.0)
    # dist 50 first; tie at 100 broken by id: 3 before 7
    assert [int(j) for j in nb[0]] == [2, 1, 0]


def test_two_phase_prefers_fewer_stations_then_cheaper():
    # demand A only reaches P0/P1; demand B only reaches Q0/Q1 -> min 2 stations
    dxy = np.array([[0.0, 0.0], [5000.0, 0.0]])
    cxy = np.array([[10.0, 0.0], [20.0, 0.0], [5010.0, 0.0], [5020.0, 0.0]])
    problem = ProblemData(
        demand_ids=np.arange(2), candidate_ids=np.arange(4),
        demand_xy=dxy, candidate_xy=cxy,
        daily_demand=np.array([100.0, 100.0]),
        capacities=np.full(4, 1000.0),
        fixed_costs=np.array([300.0, 200.0, 1000.0, 100.0]),
        candidate_types=["t"] * 4,
        demand_lat=dxy[:, 1], demand_lon=dxy[:, 0],
        cand_lat=cxy[:, 1], cand_lon=cxy[:, 0],
        radius=1000.0, alpha=1.0, cap_scale=1.0,
    )
    result = solve(problem, l1(dxy, cxy), time_limit=60.0)
    assert result["status"] == "optimal"
    assert result["station_count"] == 2
    assert result["total_cost"] == pytest.approx(300.0)  # cheap one per side
    assert set(result["selected"].tolist()) == {1, 3}


def test_infeasible_when_nothing_in_radius():
    dxy = np.array([[0.0, 0.0]])
    cxy = np.array([[5000.0, 0.0]])
    problem = ProblemData(
        demand_ids=np.arange(1), candidate_ids=np.arange(1),
        demand_xy=dxy, candidate_xy=cxy,
        daily_demand=np.array([100.0]), capacities=np.array([1000.0]),
        fixed_costs=np.array([10.0]), candidate_types=["t"],
        demand_lat=dxy[:, 1], demand_lon=dxy[:, 0],
        cand_lat=cxy[:, 1], cand_lon=cxy[:, 0],
        radius=500.0, alpha=1.0, cap_scale=1.0,
    )
    result = solve(problem, l1(dxy, cxy), time_limit=60.0)
    assert result["status"] == "infeasible"