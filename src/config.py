"""Shared configuration for CLI and Streamlit."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SolveConfig:
    radius: float
    alpha: float
    objective_mode: str  # "min_stations_then_cost" or "fixed_station_count"
    station_count: int | None
    demand_scale: float
    capacity_scale: float
    # type_parameters: dict[str, dict] with keys enabled, capacity, fixed_cost
    type_parameters: dict = field(default_factory=dict)
    time_limit: float
    mip_rel_gap: float

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if self.objective_mode not in ("min_stations_then_cost", "fixed_station_count"):
            raise ValueError("objective_mode must be min_stations_then_cost or fixed_station_count")
        if self.objective_mode == "fixed_station_count" and self.station_count is None:
            raise ValueError("station_count required for fixed_station_count mode")
        if self.station_count is not None and self.station_count < 0:
            raise ValueError("station_count must be non-negative")
        if self.demand_scale <= 0:
            raise ValueError("demand_scale must be positive")
        if self.capacity_scale <= 0:
            raise ValueError("capacity_scale must be positive")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if not (0.0 <= self.mip_rel_gap <= 1.0):
            raise ValueError("mip_rel_gap must be in [0, 1]")