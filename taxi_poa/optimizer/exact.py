"""
Exact Small-Instance Welfare Optimizer via exhaustive enumeration
and deterministic tie-breaking.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from optimizer.base import (
    CostSource,
    OptimizationResult,
    OptimizationStatus,
    evaluate_welfare,
    resolve_travel_cost,
)


def _canonical_key(assignment: Dict[Any, Any], drivers: Sequence[Any]) -> Tuple[str, ...]:
    """Generate a canonical string tuple for deterministic tie-breaking."""
    return tuple(str(assignment[d]) for d in drivers)


def optimize_welfare_exact(
    drivers: Sequence[Any],
    driver_locations: Union[Mapping[Any, Any], Sequence[Any]],
    zones: Sequence[Any],
    demand: Mapping[Any, float],
    cost_source: CostSource,
    lambda_cost: float = 1.0,
    max_combinations: int = 1_000_000,
) -> OptimizationResult:
    """
    Find the global welfare-maximizing assignment a_OPT by exhaustive enumeration.

        a_OPT = argmax_a [ sum_{z in active(a)} D(z) - lambda * sum_{i in N} c(l_i, a_i) ]

    This algorithm is guaranteed to find the true global optimum for small instances
    and serves as the ground-truth validation oracle for larger solvers.

    Args:
        drivers: Sequence of driver identifiers (e.g., [0, 1, 2] or ['d1', 'd2']).
        driver_locations: Mapping of driver_id -> location node, or sequence matching drivers.
        zones: Sequence of available demand zones Z.
        demand: Mapping of zone_id -> nonnegative demand score D(z).
        cost_source: Precomputed cost matrix or callable cost function c(l_i, z).
        lambda_cost: Travel cost trade-off parameter lambda > 0.
        max_combinations: Safety threshold on |Z|^|N| to prevent accidental memory exhaustion.

    Returns:
        OptimizationResult containing the exact optimal assignment and welfare.

    Raises:
        ValueError: If total search space exceeds max_combinations.
    """
    driver_list = list(drivers)
    zone_list = list(zones)
    num_drivers = len(driver_list)
    num_zones = len(zone_list)

    # Standardize driver_locations to mapping
    if isinstance(driver_locations, Mapping):
        loc_map = dict(driver_locations)
    elif isinstance(driver_locations, Sequence):
        if len(driver_locations) != num_drivers:
            raise ValueError(
                f"Length of driver_locations sequence ({len(driver_locations)}) "
                f"must match number of drivers ({num_drivers})."
            )
        loc_map = {d: loc for d, loc in zip(driver_list, driver_locations)}
    else:
        raise TypeError(f"Unsupported type for driver_locations: {type(driver_locations)}")

    # Edge cases:
    if num_drivers == 0 or num_zones == 0:
        return OptimizationResult(
            assignment={},
            welfare=0.0,
            active_zones=[],
            total_demand_captured=0.0,
            total_travel_cost=0.0,
            driver_utilities={},
            status=OptimizationStatus.EXACT_OPTIMAL,
            method_used="exact_enumeration",
            num_evaluations=0,
            details={"message": "Empty drivers or zones collection"},
        )

    # Single zone edge case: all drivers must choose the single zone
    if num_zones == 1:
        single_zone = zone_list[0]
        assignment = {d: single_zone for d in driver_list}
        welfare, dem_cap, travel_cost, utils = evaluate_welfare(
            assignment=assignment,
            driver_locations=loc_map,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
        )
        return OptimizationResult(
            assignment=assignment,
            welfare=welfare,
            active_zones=[single_zone],
            total_demand_captured=dem_cap,
            total_travel_cost=travel_cost,
            driver_utilities=utils,
            status=OptimizationStatus.EXACT_OPTIMAL,
            method_used="exact_enumeration",
            num_evaluations=1,
            details={"message": "Single zone trivial assignment"},
        )

    # Check search space size: |Z|^|N|
    total_combinations = num_zones ** num_drivers
    if total_combinations > max_combinations:
        raise ValueError(
            f"Search space |Z|^|N| = {num_zones}^{num_drivers} = {total_combinations:,} "
            f"exceeds max_combinations threshold ({max_combinations:,}). "
            f"Use ILPOptimizer or HeuristicOptimizer for larger instances."
        )

    best_welfare = float("-inf")
    best_assignment: Optional[Dict[Any, Any]] = None
    best_tie_key: Optional[Tuple[str, ...]] = None
    best_metrics: Optional[Tuple[float, float, Dict[Any, float]]] = None
    eval_count = 0

    # Exhaustive enumeration over all Cartesian product assignments Z^N
    for candidate_tuple in itertools.product(zone_list, repeat=num_drivers):
        eval_count += 1
        candidate_assignment = {d: z for d, z in zip(driver_list, candidate_tuple)}

        welfare, dem_cap, travel_cost, utils = evaluate_welfare(
            assignment=candidate_assignment,
            driver_locations=loc_map,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
        )

        tie_key = _canonical_key(candidate_assignment, driver_list)

        # Update if strictly better (with numerical tolerance 1e-9)
        if welfare > best_welfare + 1e-9:
            best_welfare = welfare
            best_assignment = candidate_assignment
            best_tie_key = tie_key
            best_metrics = (dem_cap, travel_cost, utils)
        elif abs(welfare - best_welfare) <= 1e-9:
            # Deterministic tie-breaking rule: lexicographically smallest string representation
            if best_tie_key is None or tie_key < best_tie_key:
                best_welfare = welfare
                best_assignment = candidate_assignment
                best_tie_key = tie_key
                best_metrics = (dem_cap, travel_cost, utils)

    dem_cap, travel_cost, utils = best_metrics  # type: ignore
    active_zones = list(set(best_assignment.values()))  # type: ignore

    return OptimizationResult(
        assignment=best_assignment,  # type: ignore
        welfare=best_welfare,
        active_zones=active_zones,
        total_demand_captured=dem_cap,
        total_travel_cost=travel_cost,
        driver_utilities=utils,
        status=OptimizationStatus.EXACT_OPTIMAL,
        method_used="exact_enumeration",
        num_evaluations=eval_count,
        details={
            "total_search_space": total_combinations,
            "evaluated": eval_count,
        },
    )


class ExactEnumerationOptimizer:
    """Class wrapper for the exact small-instance optimizer."""

    def __init__(self, max_combinations: int = 1_000_000):
        self.max_combinations = max_combinations

    def optimize(
        self,
        drivers: Sequence[Any],
        driver_locations: Union[Mapping[Any, Any], Sequence[Any]],
        zones: Sequence[Any],
        demand: Mapping[Any, float],
        cost_source: CostSource,
        lambda_cost: float = 1.0,
    ) -> OptimizationResult:
        return optimize_welfare_exact(
            drivers=drivers,
            driver_locations=driver_locations,
            zones=zones,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
            max_combinations=self.max_combinations,
        )
