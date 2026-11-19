"""
Demand Generation Module for Ride-Hailing Congestion Game.

This module provides tools for generating nonnegative demand scores D(z) >= 0
across spatial demand zones under various empirical demand regimes:
- Uniform demand (with small random perturbations)
- Concentrated demand (independent concentration controls via hotspot or power-law models)
- Degree-correlated demand (demand proportional to node connectivity)

It also provides metric functions such as the Herfindahl concentration index H.
"""

from demand.regimes import DemandRegime, DemandConfig
from demand.generator import (
    generate_demand,
    generate_uniform_demand,
    generate_concentrated_demand,
    generate_power_law_demand,
    generate_degree_correlated_demand,
    DemandGenerator,
)
from demand.metrics import (
    compute_herfindahl_index,
    compute_normalized_herfindahl_index,
    compute_demand_summary,
)

__all__ = [
    "DemandRegime",
    "DemandConfig",
    "generate_demand",
    "generate_uniform_demand",
    "generate_concentrated_demand",
    "generate_power_law_demand",
    "generate_degree_correlated_demand",
    "DemandGenerator",
    "compute_herfindahl_index",
    "compute_normalized_herfindahl_index",
    "compute_demand_summary",
]
