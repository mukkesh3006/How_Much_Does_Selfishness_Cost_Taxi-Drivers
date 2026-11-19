"""
Base data structures, status enums, and welfare evaluation utilities
for the centralized welfare optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class OptimizationStatus(str, Enum):
    """Status of the optimization result."""
    EXACT_OPTIMAL = "exact_optimal"
    SOLVER_OPTIMAL = "solver_optimal"
    HEURISTIC = "heuristic"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    ERROR = "error"


@dataclass
class OptimizationResult:
    """
    Structured outcome of the centralized welfare optimization.

    Attributes:
        assignment: Mapping of driver_id -> assigned zone_id (a_OPT).
        welfare: Optimal social welfare W(a_OPT).
        active_zones: List of zones selected by at least one driver (n_z > 0).
        total_demand_captured: Aggregate demand collected from active zones: sum_{z active} D(z).
        total_travel_cost: Aggregate travel cost incurred by all drivers: sum_i c(l_i, a_i).
        driver_utilities: Individual driver utilities U_i(a_OPT; l).
        status: Optimization status (e.g., EXACT_OPTIMAL, HEURISTIC).
        method_used: Name of the optimization method/algorithm used.
        num_evaluations: Number of candidate assignments or iterations evaluated.
        details: Additional solver-specific diagnostic metadata.
    """
    assignment: Dict[Any, Any]
    welfare: float
    active_zones: List[Any]
    total_demand_captured: float
    total_travel_cost: float
    driver_utilities: Dict[Any, float] = field(default_factory=dict)
    status: OptimizationStatus = OptimizationStatus.EXACT_OPTIMAL
    method_used: str = "exact"
    num_evaluations: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted string summary of the optimization result."""
        return (
            f"OptimizationResult({self.status.value}):\n"
            f"  Method:               {self.method_used}\n"
            f"  Optimal Welfare W(a): {self.welfare:.4f}\n"
            f"  Active Zones:         {len(self.active_zones)} zones {self.active_zones}\n"
            f"  Demand Captured:      {self.total_demand_captured:.4f}\n"
            f"  Total Travel Cost:    {self.total_travel_cost:.4f}\n"
            f"  Assignments:          {self.assignment}\n"
            f"  Evaluations:          {self.num_evaluations}"
        )


CostSource = Union[
    Mapping[Tuple[Any, Any], float],
    Mapping[Any, Mapping[Any, float]],
    Callable[[Any, Any], float],
]


def resolve_travel_cost(
    driver_id: Any,
    driver_loc: Any,
    zone: Any,
    cost_source: CostSource,
) -> float:
    """
    Resolve travel cost c(l_i, z) flexibly from various representations.

    Supports:
    1. Callable: cost_fn(driver_loc, zone) or cost_fn(driver_id, zone)
    2. Dict with Tuple keys: cost_dict[(driver_loc, zone)] or cost_dict[(driver_id, zone)]
    3. Nested Dict: cost_dict[driver_loc][zone] or cost_dict[driver_id][zone]

    Args:
        driver_id: Identifier of the driver.
        driver_loc: Current location node of the driver l_i.
        zone: Target demand zone z.
        cost_source: Cost mapping or callable function.

    Returns:
        float: Nonnegative travel cost c(l_i, z).
    """
    if callable(cost_source):
        try:
            return float(cost_source(driver_loc, zone))
        except (TypeError, KeyError):
            return float(cost_source(driver_id, zone))

    if isinstance(cost_source, Mapping):
        # 1. Try tuple key (driver_loc, zone)
        if (driver_loc, zone) in cost_source:
            return float(cost_source[(driver_loc, zone)])
        # 2. Try tuple key (driver_id, zone)
        if (driver_id, zone) in cost_source:
            return float(cost_source[(driver_id, zone)])
        # 3. Try nested dict cost_source[driver_loc][zone]
        if driver_loc in cost_source and isinstance(cost_source[driver_loc], Mapping):
            if zone in cost_source[driver_loc]:
                return float(cost_source[driver_loc][zone])
        # 4. Try nested dict cost_source[driver_id][zone]
        if driver_id in cost_source and isinstance(cost_source[driver_id], Mapping):
            if zone in cost_source[driver_id]:
                return float(cost_source[driver_id][zone])

    raise KeyError(
        f"Could not resolve travel cost from location '{driver_loc}' (driver '{driver_id}') "
        f"to zone '{zone}'. Please check cost_source mappings."
    )


def evaluate_welfare(
    assignment: Mapping[Any, Any],
    driver_locations: Mapping[Any, Any],
    demand: Mapping[Any, float],
    cost_source: CostSource,
    lambda_cost: float = 1.0,
) -> Tuple[float, float, float, Dict[Any, float]]:
    """
    Evaluate total social welfare W(a) and driver utilities for a given joint assignment.

    Welfare formulation (Eq. 4 from report):
        W(a) = sum_{z in Z: n_z(a) > 0} D(z) - lambda * sum_{i in N} c(l_i, a_i)

    Driver utility (Eq. 1 from report):
        U_i(a; l) = D(a_i) / n_{a_i}(a) - lambda * c(l_i, a_i)

    Args:
        assignment: Mapping of driver_id -> assigned_zone (a).
        driver_locations: Mapping of driver_id -> current location l_i.
        demand: Mapping of zone_id -> D(z).
        cost_source: Precomputed cost matrix or callable cost function.
        lambda_cost: Travel cost coefficient lambda > 0.

    Returns:
        Tuple containing:
            - total_welfare: W(a)
            - total_demand_captured: sum_{z active} D(z)
            - total_travel_cost: sum_i c(l_i, a_i)
            - driver_utilities: Dict[driver_id, U_i(a; l)]
    """
    if not assignment:
        return 0.0, 0.0, 0.0, {}

    # Count load per zone: n_z(a)
    zone_loads: Dict[Any, int] = {}
    for driver_id, zone in assignment.items():
        zone_loads[zone] = zone_loads.get(zone, 0) + 1

    # Active zones are those with load > 0
    active_zones = [z for z, count in zone_loads.items() if count > 0]
    total_demand_captured = sum(float(demand.get(z, 0.0)) for z in active_zones)

    # Compute travel costs and driver utilities
    total_travel_cost = 0.0
    driver_utilities: Dict[Any, float] = {}

    for driver_id, zone in assignment.items():
        loc = driver_locations.get(driver_id, driver_id)
        cost = resolve_travel_cost(driver_id, loc, zone, cost_source)
        total_travel_cost += cost

        d_z = float(demand.get(zone, 0.0))
        n_z = zone_loads[zone]
        utility = (d_z / n_z) - (lambda_cost * cost)
        driver_utilities[driver_id] = utility

    total_welfare = total_demand_captured - (lambda_cost * total_travel_cost)
    return total_welfare, total_demand_captured, total_travel_cost, driver_utilities
