"""Write run outputs to a per-run directory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import ProblemData


def write_outputs(out_dir: Path, problem: ProblemData, dist: np.ndarray,
                  selected_idx: np.ndarray, assignment_idx: np.ndarray,
                  validation: dict, summary: dict, config: dict,
                  precheck: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    if precheck is not None:
        (out_dir / "precheck.json").write_text(json.dumps(precheck, indent=2, ensure_ascii=False), encoding="utf-8")

    # selected stations
    loads = validation.get("loads", np.zeros(len(problem.candidate_ids)))
    rows = []
    for j in selected_idx:
        cap = problem.capacities[j]
        load = loads[j]
        rows.append({
            "candidate_id": int(problem.candidate_ids[j]),
            "type": problem.candidate_types[j],
            "lat": problem.cand_lat[j],
            "lon": problem.cand_lon[j],
            "capacity": cap,
            "load": load,
            "utilization": load / cap if cap > 0 else 0.0,
            "fixed_cost": problem.fixed_costs[j],
        })
    pd.DataFrame(rows).to_csv(out_dir / "selected_stations.csv", index=False)

    # assigned demand
    rows = []
    for i in range(len(problem.demand_ids)):
        j = assignment_idx[i]
        d = dist[i, j] if j >= 0 else np.nan
        rows.append({
            "demand_id": int(problem.demand_ids[i]),
            "lat": problem.demand_lat[i],
            "lon": problem.demand_lon[i],
            "daily_demand": problem.daily_demand[i],
            "station_id": int(problem.candidate_ids[j]) if j >= 0 else "",
            "distance_m": d,
            "served": j >= 0,
        })
    pd.DataFrame(rows).to_csv(out_dir / "assigned_demand.csv", index=False)