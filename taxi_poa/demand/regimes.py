"""
Demand regimes and configuration data structures.

Defines the supported demand distributions and configuration parameters
for generating zone demand profiles D(z) in taxi allocation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence, Optional, Union, Mapping, Callable


class DemandRegime(str, Enum):
    """
    Supported demand regimes as defined in the project specification.

    1. UNIFORM: All zones have approximately equal demand, with small random perturbations.
    2. CONCENTRATED: A small subset of zones receives a large share of total demand.
    3. DEGREE_CORRELATED: Demand increases with node connectivity/degree.
    4. POWER_LAW: Demand decays according to a power-law / Pareto rank distribution,
       enabling fine-grained sweeps of demand concentration independent of topology.
    """
    UNIFORM = "uniform"
    CONCENTRATED = "concentrated"
    DEGREE_CORRELATED = "degree_correlated"
    POWER_LAW = "power_law"

    @classmethod
    def from_str(cls, value: Union[str, DemandRegime]) -> DemandRegime:
        """Parse string or enum into DemandRegime safely."""
        if isinstance(value, DemandRegime):
            return value
        normalized = value.strip().lower().replace("-", "_")
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unknown demand regime '{value}'. Valid options are: {[m.value for m in cls]}"
        )


@dataclass
class DemandConfig:
    """
    Configuration parameters for demand generation.

    Attributes:
        regime: Target demand regime (UNIFORM, CONCENTRATED, DEGREE_CORRELATED, POWER_LAW).
        total_demand: Total aggregate demand sum sum_{z} D(z). If None, computed from mean_demand * |Z|.
        mean_demand: Target mean demand per zone (default: 10.0). Used if total_demand is None.
        noise_scale: Relative magnitude of random perturbations (default: 0.05, i.e., +/-5%).
        seed: Random seed for reproducible generation.
        min_demand: Minimum allowed demand score for any zone (default: 0.0, non-negative).

        # Concentrated regime specific parameters:
        hotspot_fraction: Fraction alpha in (0, 1] of zones designated as high-demand hotspots (default: 0.2).
        hotspot_share: Fraction beta in [0, 1] of total demand allocated across hotspots (default: 0.8).
        hotspot_zones: Optional explicit sequence of zone IDs designated as hotspots.

        # Power-law / Pareto concentration parameters:
        power_law_alpha: Exponent alpha >= 0 for power-law concentration w_i = (i+1)^(-alpha).
                         alpha=0 yields uniform demand, large alpha concentrates demand heavily.

        # Degree-correlated regime specific parameters:
        degree_power: Exponent gamma for scaling with degree: D(z) proportional to (degree)^gamma.
        degree_base: Base constant added before degree weighting: D(z) proportional to (degree_base + degree)^gamma.
        degrees: Optional mapping of zone_id -> degree, or callable, or sequence of degrees.
    """
    regime: Union[DemandRegime, str] = DemandRegime.UNIFORM
    total_demand: Optional[float] = None
    mean_demand: float = 10.0
    noise_scale: float = 0.05
    seed: Optional[int] = None
    min_demand: float = 0.0

    # Concentrated regime controls
    hotspot_fraction: float = 0.2
    hotspot_share: float = 0.8
    hotspot_zones: Optional[Sequence[Any]] = None

    # Power-law concentration controls
    power_law_alpha: float = 1.0

    # Degree-correlated controls
    degree_power: float = 1.0
    degree_base: float = 1.0
    degrees: Optional[Union[Mapping[Any, float], Sequence[float], Callable[[Any], float]]] = None

    def __post_init__(self):
        if isinstance(self.regime, str):
            self.regime = DemandRegime.from_str(self.regime)
        if self.min_demand < 0:
            raise ValueError(f"min_demand must be nonnegative (>= 0), got {self.min_demand}")
        if self.noise_scale < 0:
            raise ValueError(f"noise_scale must be nonnegative (>= 0), got {self.noise_scale}")
        if not (0.0 < self.hotspot_fraction <= 1.0):
            raise ValueError(
                f"hotspot_fraction must be in (0, 1], got {self.hotspot_fraction}"
            )
        if not (0.0 <= self.hotspot_share <= 1.0):
            raise ValueError(
                f"hotspot_share must be in [0, 1], got {self.hotspot_share}"
            )
        if self.power_law_alpha < 0:
            raise ValueError(
                f"power_law_alpha must be nonnegative (>= 0), got {self.power_law_alpha}"
            )
