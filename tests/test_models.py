"""Brute-force oracles on small instances for the MILP models."""
from itertools import combinations

import numpy as np
import pytest

from src.models import evaluate, max_covering, p_median, set_covering


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
