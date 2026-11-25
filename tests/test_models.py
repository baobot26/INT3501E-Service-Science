"""Brute-force oracles on small instances for the MILP models."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from src import distances_l1
from src.models import (
    BIG,
    enabled_columns,
    evaluate,
    load_inputs,
    max_covering,
    p_median,
    run_baseline,
    set_covering,
)


@pytest.fixture
def small():
    """4 demand points x 3 candidates; demand 3 is unreachable from every candidate."""
    D = np.array([
        [100.0, 600.0, 900.0],
        [200.0, 500.0, 800.0],
        [950.0, 400.0, 150.0],
        [np.inf, np.inf, 700.0],
    ])
    pop = np.array([10.0, 20.0, 30.0, 40.0])
    return D, pop


def brute_cover(D, R):
    """Size of the smallest subset covering every demand with a candidate within R."""
    cover = D <= R
    coverable = cover.any(axis=1)
    m = D.shape[1]
    for k in range(m + 1):
        for S in combinations(range(m), k):
            if cover[coverable][:, S].any(axis=1).all():
                return len(S), coverable
    raise AssertionError("unreachable")


def brute_max_cover(D, pop, R, p):
    """Most population coverable with at most p stations."""
    cover = D <= R
    m = D.shape[1]
    best = 0
    for k in range(1, min(p, m) + 1):
        for S in combinations(range(m), k):
            best = max(best, pop[cover[:, S].any(axis=1)].sum())
    return best


def test_set_covering_matches_brute_force(small):
    D, _ = small
    S, coverable, status = set_covering(D, 500)
    best_size, bc = brute_cover(D, 500)
    assert coverable.tolist() == bc.tolist()
    assert len(S) == best_size
    assert (D[coverable][:, S] <= 500).any(axis=1).all()  # every point has a station in range
    assert status == 0


def test_set_covering_nothing_coverable(small):
    D, _ = small
    S, coverable, _ = set_covering(D, 1)
    assert not coverable.any()
    assert len(S) == 0


def test_max_covering_matches_brute_force(small):
    D, pop = small
    for p in (1, 2):
        S, status = max_covering(D, pop, 500, p)
        assert len(S) <= p
        assert status == 0
        covered = pop[(D[:, S] <= 500).any(axis=1)].sum()
        assert covered == pytest.approx(brute_max_cover(D, pop, 500, p))


def test_p_median_returns_p_distinct_stations(small):
    D, pop = small
    S = p_median(D, pop, 2, np.random.default_rng(0))
    assert len(S) == 2 and len(set(S.tolist())) == 2


def test_evaluate_metrics():
    D = np.array([[100.0, 300.0], [200.0, np.inf]])
    pop = np.array([1.0, 3.0])
    out = evaluate(D, pop, np.array([0]))
    assert out["cov_500"] == 100.0
    assert out["avg_dist_m"] == 175.0  # (1*100 + 3*200) / 4
    assert out["max_dist_m"] == 200.0


def test_load_inputs_l1_filters_to_enabled_feasible(make_inputs):
    from tests.conftest import DEFAULT_CAND

    # mark the disabled building=apartments candidate as polygon-feasible too
    cand = DEFAULT_CAND.copy()
    cand.loc[2, "feasible"] = True
    data_dir = make_inputs(cand=cand)
    distances_l1.compute(data_dir)

    D, pop, ids = load_inputs("l1", data_dir)
    # 3 feasible candidates in dist_l1.npy, but only 2 have enabled == 1
    assert D.shape == (2, 2)
    assert len(pop) == 2
    assert ids.tolist() == [0, 1]


def _make_random_instance(seed: int, n: int = 8, m: int = 6, inf_prob: float = 0.2):
    rng = np.random.default_rng(seed)
    D = rng.integers(80, 950, size=(n, m)).astype(float)
    D[rng.random((n, m)) < inf_prob] = np.inf
    pop = rng.integers(10, 200, size=n).astype(float)
    return D, pop


def _p_median_cost(D: np.ndarray, pop: np.ndarray, S) -> float:
    Dp = np.where(np.isfinite(D), D, BIG)
    return float((pop * Dp[:, S].min(axis=1)).sum())


def _brute_p_median(D: np.ndarray, pop: np.ndarray, p: int):
    m = D.shape[1]
    best_S, best_cost = None, np.inf
    for S in combinations(range(m), p):
        cost = _p_median_cost(D, pop, list(S))
        if cost < best_cost:
            best_S, best_cost = np.array(S), cost
    return best_S, best_cost


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_set_covering_random_instances_match_brute_force(seed):
    D, pop = _make_random_instance(seed, n=8, m=6, inf_prob=0.25)
    R = 450.0
    S, coverable, status = set_covering(D, R)
    best_size, bc = brute_cover(D, R)

    assert status == 0
    assert coverable.tolist() == bc.tolist()
    assert len(S) == best_size
    assert ((0 <= S) & (S < D.shape[1])).all()
    if coverable.any():
        assert (D[coverable][:, S] <= R).any(axis=1).all()
    if (~coverable).any():
        assert not (D[~coverable] <= R).any()

    # Set covering ignores population weights.
    skewed_pop = pop * np.linspace(1.0, 50.0, len(pop))
    res_a = run_baseline("set_covering", D, pop, R)
    res_b = run_baseline("set_covering", D, skewed_pop, R)
    assert np.array_equal(res_a["selected"], res_b["selected"])


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
def test_max_covering_properties_and_brute_force(seed):
    D, pop = _make_random_instance(seed, n=8, m=6, inf_prob=0.15)
    R = 450.0
    coverable_pop = float(pop[(D <= R).any(axis=1)].sum())
    prev_covered = -1.0

    for p in range(1, D.shape[1] + 1):
        S, status = max_covering(D, pop, R, p)
        assert status == 0
        assert len(S) <= p
        covered = float(pop[(D[:, S] <= R).any(axis=1)].sum())
        oracle = brute_max_cover(D, pop, R, p)
        assert covered >= oracle * (1.0 - 0.001) - 1e-9
        assert covered <= oracle + 1e-9
        assert covered + 1e-9 >= prev_covered
        prev_covered = covered

    assert prev_covered == pytest.approx(coverable_pop)


def test_max_covering_respects_population_weights():
    # Candidate 0 covers point 0 only; candidate 1 covers points 1 and 2.
    D = np.array([
        [100.0, 900.0],
        [900.0, 100.0],
        [900.0, 100.0],
    ])
    pop_heavy_0 = np.array([1000.0, 10.0, 10.0])
    pop_heavy_12 = np.array([10.0, 500.0, 500.0])

    s0, _ = max_covering(D, pop_heavy_0, 300, 1)
    s1, _ = max_covering(D, pop_heavy_12, 300, 1)
    assert s0.tolist() == [0]
    assert s1.tolist() == [1]


@pytest.mark.parametrize("seed,p", [(0, 2), (1, 2), (2, 3), (3, 3)])
def test_p_median_matches_brute_force_and_is_1swap_optimal(seed, p):
    D, pop = _make_random_instance(seed, n=7, m=6, inf_prob=0.1)
    S = p_median(D, pop, p, np.random.default_rng(0))

    assert len(S) == p
    assert list(S) == sorted(S)
    assert len(set(S.tolist())) == p

    # 1-swap local optimality guarantee (Teitz-Bart).
    base_cost = _p_median_cost(D, pop, S)
    S_set = set(S.tolist())
    for pos in range(p):
        for cand in range(D.shape[1]):
            if cand in S_set:
                continue
            trial = S.copy()
            trial[pos] = cand
            assert _p_median_cost(D, pop, trial) >= base_cost - 1e-9

    # Matches global optimum on small instances.
    _, best_cost = _brute_p_median(D, pop, p)
    assert base_cost == pytest.approx(best_cost)


def test_p_median_1swap_larger_instance_determinism_and_inf():
    D, pop = _make_random_instance(99, n=12, m=8, inf_prob=0.25)
    # Force an all-inf row to confirm BIG handles unreachable demand without error.
    D[0, :] = np.inf
    p = 3

    s1 = p_median(D, pop, p, np.random.default_rng(7))
    s2 = p_median(D, pop, p, np.random.default_rng(7))
    assert np.array_equal(s1, s2)
    assert len(s1) == p
    assert list(s1) == sorted(s1)

    base_cost = _p_median_cost(D, pop, s1)
    assert np.isfinite(base_cost)
    assert base_cost >= pop[0] * BIG
    s1_set = set(s1.tolist())
    for pos in range(p):
        for cand in range(D.shape[1]):
            if cand not in s1_set:
                trial = s1.copy()
                trial[pos] = cand
                assert _p_median_cost(D, pop, trial) >= base_cost - 1e-9


def test_evaluate_custom_radii():
    D = np.array([[150.0, 600.0], [400.0, np.inf], [750.0, 500.0]])
    pop = np.array([10.0, 30.0, 60.0])
    out = evaluate(D, pop, np.array([0]), radii=[250, 500])
    assert set(out.keys()) == {"cov_250", "cov_500", "avg_dist_m", "max_dist_m"}
    assert out["cov_250"] == 10.0
    assert out["cov_500"] == 40.0
    assert out["avg_dist_m"] == 585.0  # (10*150 + 30*400 + 60*750) / 100
    assert out["max_dist_m"] == 750.0


def test_run_baseline_contract_and_empty_set_covering(small):
    D, pop = small

    sc = run_baseline("set_covering", D, pop, R=500)
    assert set(sc.keys()) == {"model", "selected", "status", "status_label", "seconds", "metrics"}
    assert sc["model"] == "set_covering"
    assert isinstance(sc["selected"], np.ndarray)
    assert np.issubdtype(sc["selected"].dtype, np.integer)
    assert list(sc["selected"]) == sorted(sc["selected"].tolist())
    assert sc["status"] == 0
    assert sc["status_label"] == "optimal"
    assert isinstance(sc["seconds"], float) and sc["seconds"] >= 0.0
    assert set(sc["metrics"].keys()) == {"coverage_pct", "avg_dist_m", "max_dist_m", "coverable_pct"}
    assert sc["metrics"]["coverable_pct"] == 60.0
    assert sc["metrics"]["coverage_pct"] == 60.0

    # Empty set_covering result when nothing is coverable.
    sc_empty = run_baseline("set_covering", D, pop, R=1)
    assert len(sc_empty["selected"]) == 0
    assert sc_empty["status"] == 0
    assert sc_empty["status_label"] == "optimal"
    assert sc_empty["metrics"] == {
        "coverage_pct": 0.0,
        "avg_dist_m": None,
        "max_dist_m": None,
        "coverable_pct": 0.0,
    }

    mc = run_baseline("max_covering", D, pop, R=500, p=2)
    assert mc["model"] == "max_covering"
    assert mc["status"] == 0
    assert mc["status_label"] == "optimal"
    assert mc["metrics"]["coverable_pct"] is None
    assert isinstance(mc["metrics"]["coverage_pct"], float)

    pm = run_baseline("p_median", D, pop, R=500, p=2, seed=5)
    assert pm["model"] == "p_median"
    assert pm["status"] is None
    assert pm["status_label"] == "heuristic"
    assert pm["metrics"]["coverable_pct"] is None
    assert np.array_equal(pm["selected"], run_baseline("p_median", D, pop, R=500, p=2, seed=5)["selected"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "unknown", "p": 1},
        {"model": "set_covering", "D": np.empty((4, 0))},
        {"model": "set_covering", "D": np.array([1.0, 2.0])},
        {"model": "max_covering", "p": None},
        {"model": "max_covering", "p": True},
        {"model": "max_covering", "p": 0},
        {"model": "max_covering", "p": 4},
        {"model": "p_median", "p": None},
        {"model": "p_median", "p": False},
        {"model": "p_median", "p": -1},
        {"model": "p_median", "p": 4},
        {"model": "set_covering", "R": -10},
        {"model": "set_covering", "R": float("nan")},
        {"model": "set_covering", "R": True},
        {"model": "set_covering", "pop": np.zeros(4)},
        {"model": "set_covering", "pop": np.array([10.0, -1.0, 20.0, 30.0])},
    ],
)
def test_run_baseline_validation_errors(small, kwargs):
    D, pop = small
    params = {"model": "set_covering", "D": D, "pop": pop, "R": 500}
    params.update(kwargs)
    with pytest.raises(ValueError) as exc_info:
        run_baseline(**params)
    assert str(exc_info.value).strip() != ""


def test_run_baseline_all_unreachable_max_covering_and_p_median():
    D = np.full((3, 2), np.inf)
    pop = np.array([10.0, 20.0, 30.0])

    mc = run_baseline("max_covering", D, pop, R=300, p=1)
    assert mc["metrics"] == {
        "coverage_pct": 0.0,
        "avg_dist_m": None,
        "max_dist_m": None,
        "coverable_pct": None,
    }

    pm = run_baseline("p_median", D, pop, R=300, p=1, seed=0)
    assert len(pm["selected"]) == 1
    assert pm["metrics"] == {
        "coverage_pct": 0.0,
        "avg_dist_m": None,
        "max_dist_m": None,
        "coverable_pct": None,
    }


def test_enabled_columns_edge_cases():
    from tests.conftest import DEFAULT_CAND, DEFAULT_PARAMS

    assert enabled_columns([], DEFAULT_CAND, DEFAULT_PARAMS).tolist() == []
    # Unknown candidate ID 999 and unknown station type both resolve to False.
    cand_unknown = pd.concat(
        [
            DEFAULT_CAND,
            pd.DataFrame(
                [
                    {
                        "id": 99,
                        "osm_id": "z",
                        "lat": 21.0,
                        "lon": 105.8,
                        "type": "unknown=type",
                        "name": "",
                        "in_residential": False,
                        "in_forbidden": False,
                        "feasible": True,
                        "exclusion_reason": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    mask = enabled_columns([0, 99, 999], cand_unknown, DEFAULT_PARAMS)
    assert mask.tolist() == [True, False, False]


def test_enabled_columns_and_app_cli_consistency(make_inputs):
    from tests.conftest import DEFAULT_CAND, DEFAULT_PARAMS

    cand = DEFAULT_CAND.copy()
    cand.loc[2, "feasible"] = True
    data_dir = make_inputs(cand=cand)
    distances_l1.compute(data_dir)

    D_raw, manifest = distances_l1.load(data_dir)
    ids = np.array(manifest["candidate_ids"])
    cand_df = pd.read_csv(data_dir / "candidates_feasible.csv")
    params_df = pd.read_csv(data_dir / "station_parameters.csv")
    pop = pd.read_csv(data_dir / "demand.csv")["population"].to_numpy(float)

    # Override tests.
    mask_default = enabled_columns(ids, cand_df, params_df)
    assert mask_default.tolist() == [True, True, False]
    mask_override = enabled_columns(
        ids, cand_df, DEFAULT_PARAMS, enabled_override={"building=apartments": True}
    )
    assert mask_override.tolist() == [True, True, True]
    mask_all_off = enabled_columns(
        ids,
        cand_df,
        DEFAULT_PARAMS,
        enabled_override={"amenity=parking": False, "shop=convenience": False},
    )
    assert mask_all_off.tolist() == [False, False, False]

    # App path (enabled_columns + run_baseline) matches CLI path (load_inputs + direct calls).
    D_app = D_raw[:, mask_default]
    D_cli, pop_cli, ids_cli = load_inputs("l1", data_dir)
    assert np.array_equal(D_app.astype(float), D_cli)
    assert np.array_equal(ids[mask_default], ids_cli)

    R = 500.0
    sc_cli, _, _ = set_covering(D_cli, R)
    sc_app = run_baseline("set_covering", D_app, pop, R)
    assert np.array_equal(sc_app["selected"], sc_cli)

    mc_cli, _ = max_covering(D_cli, pop_cli, R, p=1)
    mc_app = run_baseline("max_covering", D_app, pop, R, p=1)
    assert np.array_equal(mc_app["selected"], mc_cli)

    pm_cli = p_median(D_cli, pop_cli, p=1, rng=np.random.default_rng(0))
    pm_app = run_baseline("p_median", D_app, pop, R, p=1, seed=0)
    assert np.array_equal(pm_app["selected"], pm_cli)


