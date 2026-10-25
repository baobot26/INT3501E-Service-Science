"""Load and validate pipeline inputs; produce a unified ProblemData."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ProblemData:
    demand_ids: np.ndarray          # int, shape (n,)
    candidate_ids: np.ndarray       # int, shape (m,)
    demand_xy: np.ndarray           # float64, shape (n, 2) in EPSG:32648
    candidate_xy: np.ndarray        # float64, shape (m, 2) in EPSG:32648
    daily_demand: np.ndarray        # float64, shape (n,)
    capacities: np.ndarray          # float64, shape (m,)
    fixed_costs: np.ndarray         # float64, shape (m,)
    candidate_types: list[str]
    demand_lat: np.ndarray
    demand_lon: np.ndarray
    cand_lat: np.ndarray
    cand_lon: np.ndarray
    radius: float
    alpha: float
    cap_scale: float


class InputError(Exception):
    pass


def load_demand(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("id", "lat", "lon", "population"):
        if col not in df.columns:
            raise InputError(f"demand.csv missing column '{col}'")
    if df["id"].duplicated().any():
        raise InputError("demand.csv has duplicate ids")
    if df[["lat", "lon"]].isnull().any().any():
        raise InputError("demand.csv has null lat/lon")
    if (df["population"] < 0).any():
        raise InputError("demand.csv has negative population")
    df["daily_demand"] = 2.0 * df["population"]
    return df


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("id", "lat", "lon", "type", "feasible"):
        if col not in df.columns:
            raise InputError(f"candidates_feasible.csv missing column '{col}'")
    if df["id"].duplicated().any():
        raise InputError("candidates_feasible.csv has duplicate ids")
    if df[["lat", "lon"]].isnull().any().any():
        raise InputError("candidates_feasible.csv has null lat/lon")
    df["feasible"] = df["feasible"].astype(bool)
    return df


def load_station_parameters(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("type", "capacity_demand_day", "fixed_cost", "enabled"):
        if col not in df.columns:
            raise InputError(f"station_parameters.csv missing column '{col}'")
    if df["type"].duplicated().any():
        raise InputError("station_parameters.csv has duplicate types")
    if (df["capacity_demand_day"] <= 0).any():
        raise InputError("station_parameters.csv has non-positive capacity")
    if (df["fixed_cost"] < 0).any():
        raise InputError("station_parameters.csv has negative fixed_cost")
    return df


def build_problem_data(
    demand_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    params_df: pd.DataFrame,
    radius: float,
    alpha: float,
    cap_scale: float = 1.0,
) -> ProblemData:
    feasible = cand_df[cand_df["feasible"]].copy()
    if feasible.empty:
        raise InputError("no feasible candidates after filtering")

    # join parameters
    merged = feasible.merge(params_df, on="type", how="left")
    missing = merged["capacity_demand_day"].isnull()
    if missing.any():
        bad_types = merged.loc[missing, "type"].unique().tolist()
        raise InputError(f"candidates with type not in station_parameters: {bad_types}")

    # filter disabled types
    merged = merged[merged["enabled"] == 1].copy()
    if merged.empty:
        raise InputError("no feasible candidates after filtering disabled types")

    n = len(demand_df)
    m = len(merged)

    return ProblemData(
        demand_ids=demand_df["id"].values.astype(int),
        candidate_ids=merged["id"].values.astype(int),
        demand_xy=np.zeros((n, 2)),  # filled later by distances_l1
        candidate_xy=np.zeros((m, 2)),
        daily_demand=demand_df["daily_demand"].values.astype(float),
        capacities=(merged["capacity_demand_day"].values * cap_scale).astype(float),
        fixed_costs=merged["fixed_cost"].values.astype(float),
        candidate_types=merged["type"].tolist(),
        demand_lat=demand_df["lat"].values.astype(float),
        demand_lon=demand_df["lon"].values.astype(float),
        cand_lat=merged["lat"].values.astype(float),
        cand_lon=merged["lon"].values.astype(float),
        radius=radius,
        alpha=alpha,
        cap_scale=cap_scale,
    )


def validate_distance_manifest(
    matrix: np.ndarray,
    manifest: dict,
    demand_ids: np.ndarray,
    candidate_ids: np.ndarray,
) -> None:
    if matrix.shape[0] != len(demand_ids):
        raise InputError(
            f"distance matrix rows {matrix.shape[0]} != demand count {len(demand_ids)}")
    if matrix.shape[1] != len(candidate_ids):
        raise InputError(
            f"distance matrix cols {matrix.shape[1]} != candidate count {len(candidate_ids)}")
    if np.isnan(matrix).any():
        raise InputError("distance matrix contains NaN")
    if (matrix < 0).any():
        raise InputError("distance matrix contains negative values")
    m_ids = manifest.get("demand_ids")
    c_ids = manifest.get("candidate_ids")
    if m_ids is not None and list(m_ids) != list(demand_ids):
        raise InputError("manifest demand_ids mismatch")
    if c_ids is not None and list(c_ids) != list(candidate_ids):
        raise InputError("manifest candidate_ids mismatch")