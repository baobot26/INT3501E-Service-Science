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
from src.preprocessing import InputError


def parse_args(argv):
    p = argparse.ArgumentParser(description="Main two-phase MILP pipeline")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--radius", type=float, default=500.0)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--cap-scale", type=float, default=1.0)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--mip-rel-gap", type=float, default=0.0)
    p.add_argument("--mode", default="", help="label only, stored in config")
    return p.parse_args(argv)


def precheck(dist: np.ndarray, problem: preprocessing.ProblemData) -> dict:
    total = float(problem.daily_demand.sum())
    reachable = (dist <= problem.radius).any(axis=1)
    served_ub = float(problem.daily_demand[reachable].sum())
    coverage_ub = served_ub / total if total else 0.0
    capacity_ub = float(problem.capacities.sum())
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


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("alpha must be in [0, 1]")
    if args.radius <= 0 or args.cap_scale <= 0 or args.time_limit <= 0:
        raise SystemExit("radius, cap-scale and time-limit must be positive")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{ts}_R{args.radius:g}_a{args.alpha:g}_cap{args.cap_scale:g}"
    out_dir = args.results_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id, "mode": args.mode, "radius_m": args.radius,
        "alpha": args.alpha, "cap_scale": args.cap_scale,
        "time_limit": args.time_limit, "mip_rel_gap": args.mip_rel_gap,
        "distance_metric": "manhattan_l1", "crs": "EPSG:32648",
    }

    def finish(status: str, exit_code: int, extra: dict | None = None,
               validation: dict | None = None, pre: dict | None = None) -> int:
        summary = {"run_id": run_id, "status": status, **config, **(extra or {})}
        if validation is not None:
            summary["validation_passed"] = validation["valid"]
        if pre is not None:
            summary["precheck"] = pre
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
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
        problem = preprocessing.build_problem_data(
            demand_df, cand_df, params_df, args.radius, args.alpha, args.cap_scale)
        dist, manifest = distances_l1.load(args.data_dir)
        preprocessing.validate_distance_manifest(
            dist, manifest, problem.demand_ids, problem.candidate_ids)
    except InputError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return finish("input_error", 2)

    pre = precheck(dist, problem)
    if pre["status"] != "ok":
        return finish("precheck_infeasible", 1, pre=pre)

    result = model_main.solve(problem, dist, time_limit=args.time_limit,
                              mip_rel_gap=args.mip_rel_gap)
    status = result["status"]
    if status in ("infeasible", "precheck_infeasible"):
        return finish(status, 1, extra={"phase1_status": result["phase1_status"]}, pre=pre)
    if status not in ("optimal", "feasible_time_limit"):
        return finish(status, 3, extra={"phase1_status": result["phase1_status"],
                                        "phase2_status": result.get("phase2_status")}, pre=pre)

    selected, assignment = result["selected"], result["assignment"]
    validation = evaluate.validate_solution(
        problem, dist, selected, assignment,
        result["station_count"], result["total_cost"])
    if not validation["valid"]:
        return finish("validation_failed", 3,
                      extra={"phase1_status": result["phase1_status"],
                             "phase2_status": result.get("phase2_status"),
                             "station_count": result["station_count"]},
                      validation=validation, pre=pre)

    export.write_outputs(out_dir, problem, dist, selected, assignment,
                         validation,
                         summary={}, config={}, precheck=pre)
    # rewrite summary with full metrics (export.write_outputs left a stub)
    extra = {
        "phase1_status": result["phase1_status"],
        "phase2_status": result.get("phase2_status"),
        "phase1_seconds": result.get("phase1_seconds"),
        "phase2_seconds": result.get("phase2_seconds"),
        "station_count": result["station_count"],
        "total_cost": result["total_cost"],
        "coverage_pct": validation["metrics"]["coverage_pct"],
    }
    return finish(status, 0, extra=extra, validation=validation, pre=pre)


if __name__ == "__main__":
    sys.exit(main())