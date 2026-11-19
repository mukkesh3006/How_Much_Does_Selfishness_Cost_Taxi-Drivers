"""
Greedy Construction and Multi-Neighborhood Local Search Heuristic Optimizer
for moderate to large problem instances.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from optimizer.base import (
    CostSource,
    OptimizationResult,
    OptimizationStatus,
    evaluate_welfare,
    resolve_travel_cost,
)


def _get_rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed)


def optimize_welfare_heuristic(
    drivers: Sequence[Any],
    driver_locations: Union[Mapping[Any, Any], Sequence[Any]],
    zones: Sequence[Any],
    demand: Mapping[Any, float],
    cost_source: CostSource,
    lambda_cost: float = 1.0,
    n_restarts: int = 10,
    max_iter_per_restart: int = 200,
    seed: Optional[int] = None,
) -> OptimizationResult:
    """
    Find high-welfare assignments for larger instances using Multi-Start Greedy
    Construction followed by Variable Neighborhood Local Search (1-Opt + Zone Toggles + Swaps).

    This provides a fast, scalable heuristic solver when exact enumeration is intractable.

    Args:
        drivers: Sequence of driver identifiers.
        driver_locations: Mapping of driver_id -> location node, or sequence matching drivers.
        zones: Sequence of demand zones Z.
        demand: Mapping of zone_id -> D(z).
        cost_source: Precomputed cost matrix or callable c(l_i, z).
        lambda_cost: Travel cost coefficient lambda > 0.
        n_restarts: Number of independent restarts from diverse starting points.
        max_iter_per_restart: Maximum local search improvement steps per restart.
        seed: Random seed for deterministic reproducibility.

    Returns:
        OptimizationResult with status=OptimizationStatus.HEURISTIC.
    """
    driver_list = list(drivers)
    zone_list = list(zones)
    num_drivers = len(driver_list)
    num_zones = len(zone_list)

    if isinstance(driver_locations, Mapping):
        loc_map = dict(driver_locations)
    else:
        loc_map = {d: loc for d, loc in zip(driver_list, driver_locations)}

    if num_drivers == 0 or num_zones == 0:
        return OptimizationResult(
            assignment={},
            welfare=0.0,
            active_zones=[],
            total_demand_captured=0.0,
            total_travel_cost=0.0,
            driver_utilities={},
            status=OptimizationStatus.HEURISTIC,
            method_used="heuristic_multi_start_local_search",
            num_evaluations=0,
            details={"message": "Empty drivers or zones collection"},
        )

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
            status=OptimizationStatus.HEURISTIC,
            method_used="heuristic_single_zone",
            num_evaluations=1,
            details={"message": "Single zone trivial assignment"},
        )

    rng = _get_rng(seed)

    # Precompute travel cost matrix: cost_matrix[driver][zone]
    cost_matrix: Dict[Any, Dict[Any, float]] = {}
    for d in driver_list:
        loc = loc_map.get(d, d)
        cost_matrix[d] = {}
        for z in zone_list:
            cost_matrix[d][z] = resolve_travel_cost(d, loc, z, cost_source)

    total_evaluations = 0
    best_overall_welfare = float("-inf")
    best_overall_assignment: Dict[Any, Any] = {}

    for restart_idx in range(n_restarts):
        # 1. INITIALIZATION STRATEGIES
        current_assignment: Dict[Any, Any] = {}
        if restart_idx == 0:
            # Strategy A: Each driver chooses best single-driver net value
            for d in driver_list:
                best_z = max(
                    zone_list,
                    key=lambda z: demand.get(z, 0.0) - (lambda_cost * cost_matrix[d][z]),
                )
                current_assignment[d] = best_z
        elif restart_idx <= min(len(zone_list), num_drivers):
            # Strategy B: Active subset of top-k zones by demand
            sorted_by_demand = sorted(zone_list, key=lambda z: demand.get(z, 0.0), reverse=True)
            k = max(1, min(len(zone_list), restart_idx))
            active_subset = sorted_by_demand[:k]
            for d in driver_list:
                best_z = min(active_subset, key=lambda z: cost_matrix[d][z])
                current_assignment[d] = best_z
        elif restart_idx == min(len(zone_list), num_drivers) + 1:
            # Strategy C: Pure minimum travel cost
            for d in driver_list:
                best_z = min(zone_list, key=lambda z: cost_matrix[d][z])
                current_assignment[d] = best_z
        else:
            # Randomized Start
            for d in driver_list:
                current_assignment[d] = rng.choice(zone_list)

        # Evaluate initial assignment
        current_welfare, _, _, _ = evaluate_welfare(
            assignment=current_assignment,
            driver_locations=loc_map,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
        )
        total_evaluations += 1

        # 2. LOCAL SEARCH IMPROVEMENT LOOP
        improved = True
        step_count = 0

        while improved and step_count < max_iter_per_restart:
            improved = False
            step_count += 1

            # Neighborhood 1: 1-Opt Driver Moves
            driver_order = list(driver_list)
            rng.shuffle(driver_order)

            for d in driver_order:
                orig_zone = current_assignment[d]
                best_move_zone = orig_zone
                best_move_gain = 0.0

                for candidate_z in zone_list:
                    if candidate_z == orig_zone:
                        continue

                    # Tentative assignment
                    current_assignment[d] = candidate_z
                    test_welfare, _, _, _ = evaluate_welfare(
                        assignment=current_assignment,
                        driver_locations=loc_map,
                        demand=demand,
                        cost_source=cost_source,
                        lambda_cost=lambda_cost,
                    )
                    total_evaluations += 1

                    gain = test_welfare - current_welfare
                    if gain > best_move_gain + 1e-9:
                        best_move_gain = gain
                        best_move_zone = candidate_z

                    # Revert tentative move
                    current_assignment[d] = orig_zone

                # Apply best 1-opt move if profitable
                if best_move_gain > 1e-9:
                    current_assignment[d] = best_move_zone
                    current_welfare += best_move_gain
                    improved = True

            # Neighborhood 2: 2-Opt Driver Pair Swaps
            if num_drivers > 1:
                for i in range(num_drivers):
                    d1 = driver_order[i]
                    z1 = current_assignment[d1]
                    for j in range(i + 1, num_drivers):
                        d2 = driver_order[j]
                        z2 = current_assignment[d2]
                        if z1 == z2:
                            continue

                        current_assignment[d1] = z2
                        current_assignment[d2] = z1
                        test_welfare, _, _, _ = evaluate_welfare(
                            assignment=current_assignment,
                            driver_locations=loc_map,
                            demand=demand,
                            cost_source=cost_source,
                            lambda_cost=lambda_cost,
                        )
                        total_evaluations += 1

                        if test_welfare > current_welfare + 1e-9:
                            current_welfare = test_welfare
                            improved = True
                            break
                        else:
                            current_assignment[d1] = z1
                            current_assignment[d2] = z2
                    if improved:
                        break

            # Neighborhood 3: Zone Activation / Deactivation Moves
            active_set = set(current_assignment.values())
            inactive_zones = [z for z in zone_list if z not in active_set]

            for inact_z in inactive_zones:
                if demand.get(inact_z, 0.0) <= 0:
                    continue
                # Pick driver closest to inact_z
                best_d = min(driver_list, key=lambda d: cost_matrix[d][inact_z])
                prev_z = current_assignment[best_d]

                current_assignment[best_d] = inact_z
                test_welfare, _, _, _ = evaluate_welfare(
                    assignment=current_assignment,
                    driver_locations=loc_map,
                    demand=demand,
                    cost_source=cost_source,
                    lambda_cost=lambda_cost,
                )
                total_evaluations += 1

                if test_welfare > current_welfare + 1e-9:
                    current_welfare = test_welfare
                    improved = True
                else:
                    current_assignment[best_d] = prev_z

        # Track global best across restarts
        if current_welfare > best_overall_welfare + 1e-9:
            best_overall_welfare = current_welfare
            best_overall_assignment = dict(current_assignment)

    # Compute final detailed metrics on best found assignment
    final_welfare, dem_cap, travel_cost, utils = evaluate_welfare(
        assignment=best_overall_assignment,
        driver_locations=loc_map,
        demand=demand,
        cost_source=cost_source,
        lambda_cost=lambda_cost,
    )
    active_zones = list(set(best_overall_assignment.values()))

    return OptimizationResult(
        assignment=best_overall_assignment,
        welfare=final_welfare,
        active_zones=active_zones,
        total_demand_captured=dem_cap,
        total_travel_cost=travel_cost,
        driver_utilities=utils,
        status=OptimizationStatus.HEURISTIC,
        method_used="heuristic_multi_start_local_search",
        num_evaluations=total_evaluations,
        details={
            "restarts": n_restarts,
            "seed": seed,
            "evaluations": total_evaluations,
        },
    )


class GreedyLocalSearchOptimizer:
    """Class wrapper for the heuristic optimizer."""

    def __init__(
        self,
        n_restarts: int = 10,
        max_iter_per_restart: int = 200,
        seed: Optional[int] = None,
    ):
        self.n_restarts = n_restarts
        self.max_iter_per_restart = max_iter_per_restart
        self.seed = seed

    def optimize(
        self,
        drivers: Sequence[Any],
        driver_locations: Union[Mapping[Any, Any], Sequence[Any]],
        zones: Sequence[Any],
        demand: Mapping[Any, float],
        cost_source: CostSource,
        lambda_cost: float = 1.0,
        seed: Optional[int] = None,
    ) -> OptimizationResult:
        active_seed = seed if seed is not None else self.seed
        return optimize_welfare_heuristic(
            drivers=drivers,
            driver_locations=driver_locations,
            zones=zones,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
            n_restarts=self.n_restarts,
            max_iter_per_restart=self.max_iter_per_restart,
            seed=active_seed,
        )
