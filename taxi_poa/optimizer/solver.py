"""
Unified Centralized Welfare Optimizer Interface.

Provides the primary public entry point `optimize_welfare` and the `CentralizedOptimizer`
class for finding the joint driver assignment a_OPT maximizing total social welfare W(a).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from optimizer.base import (
    CostSource,
    OptimizationResult,
    OptimizationStatus,
    evaluate_welfare,
)
from optimizer.exact import optimize_welfare_exact
from optimizer.heuristic import optimize_welfare_heuristic
from optimizer.ilp import is_ilp_solver_available, optimize_welfare_ilp


def optimize_welfare(
    drivers: Sequence[Any],
    driver_locations: Union[Mapping[Any, Any], Sequence[Any]],
    zones: Sequence[Any],
    demand: Mapping[Any, float],
    cost_source: CostSource,
    lambda_cost: float = 1.0,
    method: str = "auto",
    seed: Optional[int] = None,
    max_exact_combinations: int = 50_000,
    **kwargs: Any,
) -> OptimizationResult:
    """
    Find the centralized welfare-maximizing driver assignment a_OPT:

        a_OPT = argmax_a [ sum_{z in Z: n_z(a) > 0} D(z) - lambda * sum_{i in N} c(l_i, a_i) ]

    And compute the optimal social welfare:
        W_OPT = W(a_OPT)

    Args:
        drivers: Collection/sequence of driver identifiers (e.g., [0, 1, 2] or ['d1', 'd2']).
        driver_locations: Current location nodes for each driver l_i (Mapping or Sequence).
        zones: Collection/sequence of available demand zones Z.
        demand: Mapping of zone_id -> nonnegative demand score D(z) (e.g. from demand module).
        cost_source: Precomputed travel cost matrix or callable cost function c(l_i, z).
        lambda_cost: Travel cost trade-off parameter lambda > 0 (default: 1.0).
        method: Optimization method:
            - 'auto': Uses exact enumeration if |Z|^|N| <= max_exact_combinations;
                      otherwise uses ILP if available, or multi-start local search heuristic.
            - 'exact': Exhaustive enumeration (guaranteed exact global optimum).
            - 'ilp': Integer linear programming solver with automatic fallback.
            - 'heuristic': Multi-start greedy construction + variable neighborhood local search.
        seed: Random seed for deterministic reproducibility in heuristic restarts.
        max_exact_combinations: Threshold on |Z|^|N| for automatic exact solving (default: 50,000).
        **kwargs: Additional solver parameters (e.g. n_restarts, max_iter_per_restart).

    Returns:
        OptimizationResult: Dataclass containing:
            - assignment: Dict[driver_id, assigned_zone]
            - welfare: float W(a_OPT)
            - active_zones: List of activated zones
            - total_demand_captured: float
            - total_travel_cost: float
            - driver_utilities: Dict[driver_id, U_i]
            - status: OptimizationStatus (EXACT_OPTIMAL, SOLVER_OPTIMAL, HEURISTIC)
            - method_used: str

    Examples:
        >>> from demand import generate_demand
        >>> from optimizer import optimize_welfare
        >>> drivers = ['d1', 'd2']
        >>> locations = {'d1': 'A', 'd2': 'B'}
        >>> zones = ['Z1', 'Z2']
        >>> demand = generate_demand(zones, regime="uniform", seed=42)
        >>> cost_matrix = {('A', 'Z1'): 1.0, ('A', 'Z2'): 5.0, ('B', 'Z1'): 4.0, ('B', 'Z2'): 1.0}
        >>> result = optimize_welfare(drivers, locations, zones, demand, cost_matrix, lambda_cost=1.0)
        >>> print(result.welfare, result.assignment)
    """
    driver_list = list(drivers)
    zone_list = list(zones)
    num_drivers = len(driver_list)
    num_zones = len(zone_list)

    norm_method = method.strip().lower().replace("-", "_")

    if norm_method == "auto":
        # Calculate search space size: |Z|^|N|
        if num_zones == 0 or num_drivers == 0:
            total_combinations = 0
        else:
            total_combinations = num_zones ** num_drivers

        if total_combinations <= max_exact_combinations:
            return optimize_welfare_exact(
                drivers=driver_list,
                driver_locations=driver_locations,
                zones=zone_list,
                demand=demand,
                cost_source=cost_source,
                lambda_cost=lambda_cost,
                max_combinations=max_exact_combinations,
            )

        # Check if an ILP solver is available in the environment
        ilp_available, _ = is_ilp_solver_available()
        if ilp_available:
            return optimize_welfare_ilp(
                drivers=driver_list,
                driver_locations=driver_locations,
                zones=zone_list,
                demand=demand,
                cost_source=cost_source,
                lambda_cost=lambda_cost,
                fallback_to_heuristic=True,
                seed=seed,
            )

        # Scalable heuristic for large instances
        n_restarts = kwargs.get("n_restarts", 10)
        max_iter = kwargs.get("max_iter_per_restart", 200)
        return optimize_welfare_heuristic(
            drivers=driver_list,
            driver_locations=driver_locations,
            zones=zone_list,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
            n_restarts=n_restarts,
            max_iter_per_restart=max_iter,
            seed=seed,
        )

    elif norm_method == "exact":
        return optimize_welfare_exact(
            drivers=driver_list,
            driver_locations=driver_locations,
            zones=zone_list,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
            max_combinations=kwargs.get("max_combinations", 1_000_000),
        )

    elif norm_method == "ilp":
        return optimize_welfare_ilp(
            drivers=driver_list,
            driver_locations=driver_locations,
            zones=zone_list,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
            fallback_to_heuristic=kwargs.get("fallback_to_heuristic", True),
            seed=seed,
        )

    elif norm_method in ("heuristic", "local_search", "greedy"):
        n_restarts = kwargs.get("n_restarts", 10)
        max_iter = kwargs.get("max_iter_per_restart", 200)
        return optimize_welfare_heuristic(
            drivers=driver_list,
            driver_locations=driver_locations,
            zones=zone_list,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
            n_restarts=n_restarts,
            max_iter_per_restart=max_iter,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown optimization method '{method}'. Valid options are: ['auto', 'exact', 'ilp', 'heuristic']"
        )


class CentralizedOptimizer:
    """
    Object-oriented centralized welfare optimizer storing configuration parameters.
    """

    def __init__(
        self,
        lambda_cost: float = 1.0,
        default_method: str = "auto",
        seed: Optional[int] = None,
        max_exact_combinations: int = 50_000,
    ):
        self.lambda_cost = lambda_cost
        self.default_method = default_method
        self.seed = seed
        self.max_exact_combinations = max_exact_combinations

    def optimize(
        self,
        drivers: Sequence[Any],
        driver_locations: Union[Mapping[Any, Any], Sequence[Any]],
        zones: Sequence[Any],
        demand: Mapping[Any, float],
        cost_source: CostSource,
        lambda_cost: Optional[float] = None,
        method: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Solve for optimal driver assignment a_OPT and social welfare W_OPT."""
        active_lambda = lambda_cost if lambda_cost is not None else self.lambda_cost
        active_method = method if method is not None else self.default_method
        active_seed = seed if seed is not None else self.seed
        return optimize_welfare(
            drivers=drivers,
            driver_locations=driver_locations,
            zones=zones,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=active_lambda,
            method=active_method,
            seed=active_seed,
            max_exact_combinations=self.max_exact_combinations,
            **kwargs,
        )
