"""One-command pipeline for the main two-phase model.

Exit codes: 0 solved+validated, 1 infeasible (precheck or solver),
2 input/validation error, 3 solver/validation failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src import distances_l1, evaluate, export, model_main, preprocessing
from src.config import SolveConfig
from src.preprocessing import InputError


def parse_args(argv):
    p = argparse.ArgumentParser(description="Main two-phase MILP pipeline")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--radius", type=float, default=500.0)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--cap-scale", type=float, default=1.0)
    p.add_argument("--demand-scale", type=float, default=1.0)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--mip-rel-gap", type=float, default=0.0)
    p.add_argument("--mode", default="", help="label only, stored in config")
    p.add_argument("--objective", default="min_stations_then_cost",
                   choices=["min_stations_then_cost", "fixed_station_count"])
    p.add_argument("--station-count", type=int, default=None,
                   help="required when --objective=fixed_station_count")
    p.add_argument("--type-params", type=Path, default=None,
                   help="JSON file with per-type overrides: "
                        "{type: {enabled, capacity, fixed_cost}}")
    return p.parse_args(argv)


def load_type_params(path) -> dict:
    """Read per-type overrides from a JSON file."""
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read --type-params {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("--type-params must be a JSON object keyed by station type")
    return data


def precheck(dist: np.ndarray, problem: preprocessing.ProblemData) -> dict:
    total = float(problem.daily_demand.sum())
    # disabled station types can never open, so they are not assignment options
    enabled = (problem.enabled_mask if problem.enabled_mask is not None
               else np.ones(len(problem.candidate_ids), dtype=bool))
    reachable = ((dist <= problem.radius) & enabled).any(axis=1)
    served_ub = float(problem.daily_demand[reachable].sum())
    coverage_ub = served_ub / total if total else 0.0
    capacity_ub = float(problem.capacities[enabled].sum())
    required = problem.alpha * total
    coverage_ok = served_ub + 1e-9 >= required
    capacity_ok = capacity_ub + 1e-9 >= required
    status = "ok"
    if not coverage_ok:
        status = "precheck_infeasible_coverage"
    elif not capacity_ok:
        status = "precheck_infeasible_capacity"
    return {
        "coverage_upper_bound_pct": round(coverage_ub * 100, 4),
        "required_coverage_pct": round(problem.alpha * 100, 4),
        "capacity_upper_bound": capacity_ub,
        "required_capacity": required,
        "coverage_ok": coverage_ok,
        "capacity_ok": capacity_ok,
        "status": status,
    }


def build_scenario_problem(config, demand_df, cand_df, params_df):
    """Build ProblemData for a scenario. Shared by the CLI and the Streamlit app."""
    return preprocessing.build_problem_data(
        demand_df, cand_df, params_df, config.radius, config.alpha,
        cap_scale=config.capacity_scale,
        demand_scale=config.demand_scale,
        type_parameters=config.type_parameters or None,
    )


def solve_scenario(config, problem, dist, manifest):
    """Solve and validate one scenario. Shared by the CLI and the Streamlit app
    so both produce the same result for the same configuration."""
    preprocessing.validate_distance_manifest(
        dist, manifest, problem.demand_ids, problem.candidate_ids)
    result = model_main.solve(
        problem, dist, time_limit=config.time_limit,
        mip_rel_gap=config.mip_rel_gap,
        objective_mode=config.objective_mode,
        fixed_station_count=config.station_count)
    validation = evaluate.validate_solution(
        problem, dist, result["selected"], result["assignment"],
        result["station_count"], result["total_cost"])
    return result, validation


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("alpha must be in [0, 1]")
    if args.radius <= 0 or args.cap_scale <= 0 or args.time_limit <= 0:
        raise SystemExit("radius, cap-scale and time-limit must be positive")
    if args.demand_scale <= 0:
        raise SystemExit("demand-scale must be positive")
    if args.objective == "fixed_station_count" and args.station_count is None:
        raise SystemExit("--station-count required with --objective=fixed_station_count")
    type_params = load_type_params(args.type_params)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{ts}_R{args.radius:g}_a{args.alpha:g}_cap{args.cap_scale:g}"
    out_dir = args.results_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_id": run_id, "mode": args.mode, "radius_m": args.radius,
        "alpha": args.alpha, "cap_scale": args.cap_scale,
        "demand_scale": args.demand_scale,
        "objective_mode": args.objective,
        "station_count": args.station_count,
        "time_limit": args.time_limit, "mip_rel_gap": args.mip_rel_gap,
        "type_parameters": type_params,
        "distance_metric": "manhattan_l1", "crs": "EPSG:32648",
    }

    def finish(status: str, exit_code: int, extra: dict | None = None,
               validation: dict | None = None, pre: dict | None = None) -> int:
        summary = {"run_id": run_id, "status": status, **run_config, **(extra or {})}
        if validation is not None:
            summary["validation_passed"] = validation["valid"]
        if pre is not None:
            summary["precheck"] = pre
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
        if pre is not None:
            (out_dir / "precheck.json").write_text(
                json.dumps(pre, indent=2, ensure_ascii=False), encoding="utf-8")
        if validation is not None:
            (out_dir / "validation.json").write_text(
                json.dumps(validation, indent=2, ensure_ascii=False, default=float),
                encoding="utf-8")
        print(json.dumps({"status": status, **(extra or {})}, ensure_ascii=False))
        return exit_code

    try:
        demand_df = preprocessing.load_demand(args.data_dir / "demand.csv")
        cand_df = preprocessing.load_candidates(args.data_dir / "candidates_feasible.csv")
        params_df = preprocessing.load_station_parameters(
            args.data_dir / "station_parameters.csv")
        solve_config = SolveConfig(
            radius=args.radius, alpha=args.alpha,
            objective_mode=args.objective,
            station_count=args.station_count,
            demand_scale=args.demand_scale,
            capacity_scale=args.cap_scale,
            time_limit=args.time_limit,
            mip_rel_gap=args.mip_rel_gap,
            type_parameters=type_params,
        )
        problem = build_scenario_problem(
            solve_config, demand_df, cand_df, params_df)
        dist, manifest = distances_l1.load(args.data_dir)
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return finish("input_error", 2)
    except FileNotFoundError as exc:
        print(f"input error: missing file — {exc}", file=sys.stderr)
        return finish("input_error", 2)
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return finish("input_error", 2)

    pre = precheck(dist, problem)
    if pre["status"] != "ok":
        return finish("precheck_infeasible", 1, pre=pre)

    result, validation = solve_scenario(solve_config, problem, dist, manifest)
    status = result["status"]
    if status in ("infeasible", "precheck_infeasible"):
        return finish(status, 1, extra={"solution_source": result.get("solution_source")}, pre=pre)
    if status not in ("optimal", "feasible_time_limit"):
        return finish(status, 3, extra={"solution_source": result.get("solution_source")}, pre=pre)

    selected, assignment = result["selected"], result["assignment"]
    if not validation["valid"]:
        return finish("validation_failed", 3,
                      extra={"station_count": result["station_count"],
                             "solution_source": result.get("solution_source")},
                      validation=validation, pre=pre)

    export.write_outputs(out_dir, problem, dist, selected, assignment,
                         validation, precheck=pre)
    extra = {
        "phase1_seconds": result.get("phase1_seconds"),
        "phase2_seconds": result.get("phase2_seconds"),
        "total_seconds": result.get("total_seconds"),
        "solution_source": result.get("solution_source"),
        "station_count": result["station_count"],
        "total_cost": result["total_cost"],
        "coverage_pct": validation["metrics"]["coverage_pct"],
    }
    return finish(status, 0, extra=extra, validation=validation, pre=pre)


if __name__ == "__main__":
    sys.exit(main())