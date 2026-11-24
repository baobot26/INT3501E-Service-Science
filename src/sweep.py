"""Sensitivity analysis sweep across a grid of service radii R and coverage targets alpha."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import distances_l1, preprocessing, run
from src.config import SolveConfig

COLUMNS = [
    "radius_m",
    "alpha",
    "cap_scale",
    "precheck_status",
    "coverage_ub_pct",
    "status",
    "solution_source",
    "station_count",
    "total_cost",
    "coverage_pct",
    "validation_passed",
    "seconds",
]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Sensitivity sweep over R x alpha grid")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--radii", nargs="+", type=float, default=[300, 400, 500, 600, 700, 800])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--cap-scale", type=float, default=1.0)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    demand_df = preprocessing.load_demand(args.data_dir / "demand.csv")
    cand_df = preprocessing.load_candidates(args.data_dir / "candidates_feasible.csv")
    params_df = preprocessing.load_station_parameters(args.data_dir / "station_parameters.csv")
    dist, manifest = distances_l1.load(args.data_dir)

    rows = []
    manifest_validated = False
    for radius in args.radii:
        for alpha in args.alphas:
            t0 = time.perf_counter()
            config = SolveConfig(
                radius=radius,
                alpha=alpha,
                objective_mode="min_stations_then_cost",
                station_count=None,
                demand_scale=1.0,
                capacity_scale=args.cap_scale,
                time_limit=args.time_limit,
                mip_rel_gap=args.mip_rel_gap,
            )
            problem = run.build_scenario_problem(config, demand_df, cand_df, params_df)
            if not manifest_validated:
                preprocessing.validate_distance_manifest(
                    dist, manifest, problem.demand_ids, problem.candidate_ids
                )
                manifest_validated = True

            pre = run.precheck(dist, problem)
            if pre["status"] != "ok":
                seconds = round(time.perf_counter() - t0, 3)
                row = {
                    "radius_m": radius,
                    "alpha": alpha,
                    "cap_scale": args.cap_scale,
                    "precheck_status": pre["status"],
                    "coverage_ub_pct": pre["coverage_upper_bound_pct"],
                    "status": pre["status"],
                    "solution_source": "",
                    "station_count": "",
                    "total_cost": "",
                    "coverage_pct": "",
                    "validation_passed": "",
                    "seconds": seconds,
                }
            else:
                result, validation = run.solve_scenario(config, problem, dist, manifest)
                seconds = round(result.get("total_seconds", time.perf_counter() - t0), 3)
                row = {
                    "radius_m": radius,
                    "alpha": alpha,
                    "cap_scale": args.cap_scale,
                    "precheck_status": pre["status"],
                    "coverage_ub_pct": pre["coverage_upper_bound_pct"],
                    "status": result["status"],
                    "solution_source": result.get("solution_source", ""),
                    "station_count": result["station_count"],
                    "total_cost": result["total_cost"],
                    "coverage_pct": validation["metrics"]["coverage_pct"],
                    "validation_passed": validation["valid"],
                    "seconds": seconds,
                }
            rows.append(row)
            print(
                f"R={radius:g} alpha={alpha:g} -> {row['status']} "
                f"(p={row['station_count']}, cov={row['coverage_pct']}, "
                f"valid={row['validation_passed']}, {row['seconds']}s)",
                flush=True,
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.results_dir / f"sweep_{ts}.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out_path, index=False)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
