"""Compute Manhattan (L1) distance matrix in EPSG:32648 and save with manifest."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from src.preprocessing import (InputError, load_candidates, load_demand,
                               load_station_parameters)

NPY_NAME = "dist_l1.npy"
MANIFEST_NAME = "dist_l1_manifest.json"


def _project_to_32648(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return (n, 2) array of (x, y) in EPSG:32648."""
    pts = [Point(lo, la) for la, lo in zip(lat, lon)]
    gdf = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")
    projected = gdf.to_crs("EPSG:32648")
    coords = np.array([[g.x, g.y] for g in projected.geometry])
    return coords


def compute(data_dir: Path, mode: str | None = None) -> Path:
    """Compute and save the L1 distance matrix. Returns path to npy."""
    data_dir = Path(data_dir)
    demand_df = load_demand(data_dir / "demand.csv")
    cand_df = load_candidates(data_dir / "candidates_feasible.csv")
    feasible = cand_df[cand_df["feasible"]].copy()
    # match build_problem_data: also drop disabled types so manifest ids align
    params_df = load_station_parameters(data_dir / "station_parameters.csv")
    feasible = feasible.merge(params_df[["type", "enabled"]], on="type", how="left")
    if feasible["enabled"].isna().any():
        raise InputError("candidate type missing from station_parameters.csv")
    feasible = feasible[feasible["enabled"] == 1]

    if feasible.empty:
        raise InputError("no feasible candidates for distance computation")

    demand_xy = _project_to_32648(demand_df["lat"].values, demand_df["lon"].values)
    cand_xy = _project_to_32648(feasible["lat"].values, feasible["lon"].values)

    # L1 distance: |x_i - x_j| + |y_i - y_j|
    dx = np.abs(demand_xy[:, 0:1] - cand_xy[:, 0:1].T)  # (n, m)
    dy = np.abs(demand_xy[:, 1:2] - cand_xy[:, 1:2].T)  # (n, m)
    dist = (dx + dy).astype(np.float32)

    npy_path = data_dir / NPY_NAME
    np.save(npy_path, dist)

    manifest = {
        "metric": "manhattan_l1",
        "crs": "EPSG:32648",
        "demand_ids": demand_df["id"].tolist(),
        "candidate_ids": feasible["id"].tolist(),
        "demand_count": len(demand_df),
        "candidate_count": len(feasible),
    }
    (data_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"distance matrix: {dist.shape}, saved to {npy_path}")
    return npy_path


def load(data_dir: Path) -> tuple[np.ndarray, dict]:
    """Load existing distance matrix and manifest."""
    data_dir = Path(data_dir)
    npy_path = data_dir / NPY_NAME
    manifest_path = data_dir / MANIFEST_NAME
    if not npy_path.exists():
        raise InputError(f"{npy_path} not found; run distances_l1.compute first")
    if not manifest_path.exists():
        raise InputError(f"{manifest_path} not found")
    matrix = np.load(npy_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return matrix, manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute L1 distance matrix")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    compute(args.data_dir)