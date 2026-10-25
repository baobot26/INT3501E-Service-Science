"""End-to-end offline pipeline on the tiny fixture in tmp_path."""
import json

import pytest

from src import distances_l1, run
from tests.conftest import DEFAULT_CAND, DEFAULT_DEMAND


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_run(make_inputs):
    data_dir = make_inputs()
    results = data_dir / "results"
    distances_l1.compute(data_dir)
    code = run.main(["--data-dir", str(data_dir), "--results-dir", str(results),
                     "--radius", "500", "--alpha", "0.95"])
    assert code == 0
    out_dirs = list(results.iterdir())
    assert len(out_dirs) == 1
    out = out_dirs[0]
    for name in ("config.json", "summary.json", "validation.json",
                 "selected_stations.csv", "assigned_demand.csv", "precheck.json"):
        assert (out / name).exists(), name
    summary = read_json(out / "summary.json")
    assert summary["status"] == "optimal"
    assert summary["validation_passed"] is True
    assert summary["station_count"] == 2  # each demand needs its own station at R=500
    assert summary["coverage_pct"] == 100.0


def test_two_runs_do_not_overwrite(make_inputs):
    data_dir = make_inputs()
    results = data_dir / "results"
    distances_l1.compute(data_dir)
    assert run.main(["--data-dir", str(data_dir), "--results-dir", str(results)]) == 0
    assert run.main(["--data-dir", str(data_dir), "--results-dir", str(results)]) == 0
    assert len(list(results.iterdir())) == 2


def test_precheck_stops_before_solver(make_inputs, monkeypatch):
    # demand 1 has no candidate within 500 m -> alpha=1 is certainly infeasible
    demand = DEFAULT_DEMAND.copy()
    demand.loc[1, "lat"] = 21.10  # ~7 km away
    data_dir = make_inputs(demand=demand)
    distances_l1.compute(data_dir)

    def boom(*a, **k):
        raise AssertionError("solver must not be called when precheck fails")

    monkeypatch.setattr("src.model_main.milp", boom)
    code = run.main(["--data-dir", str(data_dir),
                     "--results-dir", str(data_dir / "results"),
                     "--radius", "500", "--alpha", "1.0"])
    assert code == 1
    out = next((data_dir / "results").iterdir())
    summary = read_json(out / "summary.json")
    assert summary["status"] == "precheck_infeasible"
    assert summary["precheck"]["coverage_ok"] is False


def test_bad_input_exits_with_error(make_inputs, capsys):
    cand = DEFAULT_CAND.copy()
    cand["feasible"] = False
    data_dir = make_inputs(cand=cand)
    code = run.main(["--data-dir", str(data_dir),
                     "--results-dir", str(data_dir / "results")])
    assert code == 2
    assert "no feasible" in capsys.readouterr().err


def test_alpha_out_of_range_rejected(make_inputs):
    data_dir = make_inputs()
    with pytest.raises(SystemExit):
        run.main(["--data-dir", str(data_dir), "--alpha", "1.5"])