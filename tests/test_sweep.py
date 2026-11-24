"""Offline tests for the R x alpha sensitivity sweep CLI."""
import pandas as pd

from src import distances_l1, sweep


def test_sweep_grid_writes_csv(make_inputs):
    data_dir = make_inputs()
    results_dir = data_dir / "results"
    distances_l1.compute(data_dir)

    code = sweep.main([
        "--data-dir", str(data_dir),
        "--results-dir", str(results_dir),
        "--radii", "100", "500",
        "--alphas", "0.5", "1.0",
        "--time-limit", "30",
    ])
    assert code == 0

    csvs = list(results_dir.glob("sweep_*.csv"))
    assert len(csvs) == 1
    df = pd.read_csv(csvs[0])
    assert len(df) == 4
    assert list(df.columns) == sweep.COLUMNS
    optimal_rows = df[df["status"] == "optimal"]
    assert len(optimal_rows) >= 1
    assert optimal_rows["validation_passed"].all()
