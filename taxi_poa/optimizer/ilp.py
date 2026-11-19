"""
Integer Linear Programming (ILP) Solver Adapter for the Centralized Welfare Problem.

Formulates and solves:
    max   sum_{z in Z} D(z) * y_z - lambda * sum_{i in N} sum_{z in Z} c(l_i, z) * x_{i, z}
    s.t.  sum_{z in Z} x_{i, z} = 1          forall i in N
          x_{i, z} <= y_z                    forall i in N, z in Z
          y_z <= sum_i x_{i, z}              forall z in Z
          x_{i, z} in {0, 1}, y_z in {0, 1}

Automatically detects available solvers (SciPy MILP, PuLP, OR-Tools).
If no third-party solver is present, provides a documented pure-Python exact/heuristic fallback.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from optimizer.base import (
    CostSource,
    OptimizationResult,
    OptimizationStatus,
    evaluate_welfare,
    resolve_travel_cost,
)
from optimizer.heuristic import optimize_welfare_heuristic


def is_ilp_solver_available() -> Tuple[bool, str]:
    """Check if an external ILP solver is available in the environment."""
    try:
        import scipy.optimize  # type: ignore
        if hasattr(scipy.optimize, "milp") and hasattr(scipy.optimize, "LinearConstraint"):
            return True, "scipy"
    except ImportError:
        pass

    try:
        import pulp  # type: ignore
        return True, "pulp"
    except ImportError:
        pass

    try:
        import ortools.linear_solver  # type: ignore
        return True, "ortools"
    except ImportError:
        pass

    return False, "none"


def _solve_with_scipy(
    driver_list: List[Any],
    zone_list: List[Any],
    loc_map: Dict[Any, Any],
    demand: Mapping[Any, float],
    cost_source: CostSource,
    lambda_cost: float,
) -> Optional[OptimizationResult]:
    """Solve the ILP using scipy.optimize.milp."""
    try:
        import numpy as np  # type: ignore
        from scipy.optimize import milp, LinearConstraint, Bounds  # type: ignore
    except ImportError:
        return None

    num_drivers = len(driver_list)
    num_zones = len(zone_list)

    # Variables:
    # x_{i, z} for i in 0..num_drivers-1, z in 0..num_zones-1 -> num_drivers * num_zones vars
    # y_z for z in 0..num_zones-1 -> num_zones vars
    num_x = num_drivers * num_zones
    num_vars = num_x + num_zones

    def var_x(i: int, z: int) -> int:
        return i * num_zones + z

    def var_y(z: int) -> int:
        return num_x + z

    # Objective: min sum_i sum_z (lambda * c(l_i, z)) * x_{i, z} - sum_z D(z) * y_z
    c_obj = np.zeros(num_vars, dtype=float)
    for i, d in enumerate(driver_list):
        loc = loc_map.get(d, d)
        for j, z in enumerate(zone_list):
            travel_c = resolve_travel_cost(d, loc, z, cost_source)
            c_obj[var_x(i, j)] = lambda_cost * travel_c

    for j, z in enumerate(zone_list):
        c_obj[var_y(j)] = -float(demand.get(z, 0.0))

    # Constraints:
    # 1. sum_z x_{i, z} = 1 for each driver i
    # 2. x_{i, z} - y_z <= 0 for each i, z
    num_eq = num_drivers
    num_ineq = num_drivers * num_zones + num_zones  # linking + activation
    total_constraints = num_eq + num_ineq

    A = np.zeros((total_constraints, num_vars), dtype=float)
    b_l = np.zeros(total_constraints, dtype=float)
    b_u = np.zeros(total_constraints, dtype=float)

    # Equality constraints
    for i in range(num_drivers):
        for j in range(num_zones):
            A[i, var_x(i, j)] = 1.0
        b_l[i] = 1.0
        b_u[i] = 1.0

    # Linking constraints: x_{i, z} - y_z <= 0
    row = num_drivers
    for i in range(num_drivers):
        for j in range(num_zones):
            A[row, var_x(i, j)] = 1.0
            A[row, var_y(j)] = -1.0
            b_l[row] = -np.inf
            b_u[row] = 0.0
            row += 1

    # Activation constraints: y_z <= sum_i x_{i, z}.
    # Without these, the solver can set y_z = 1 for a zone nobody serves and
    # collect D(z) for free, which overstates welfare and corrupts the assignment.
    for j in range(num_zones):
        A[row, var_y(j)] = 1.0
        for i in range(num_drivers):
            A[row, var_x(i, j)] = -1.0
        b_l[row] = -np.inf
        b_u[row] = 0.0
        row += 1

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones(num_vars, dtype=int)  # all binary 0-1
    bounds = Bounds(lb=0.0, ub=1.0)

    res = milp(c=c_obj, integrality=integrality, constraints=constraints, bounds=bounds)

    if not res.success:
        return None

    # Reconstruct assignment
    assignment: Dict[Any, Any] = {}
    for i, d in enumerate(driver_list):
        best_j = max(range(num_zones), key=lambda j: res.x[var_x(i, j)])
        assignment[d] = zone_list[best_j]

    welfare, dem_cap, travel_cost, utils = evaluate_welfare(
        assignment=assignment,
        driver_locations=loc_map,
        demand=demand,
        cost_source=cost_source,
        lambda_cost=lambda_cost,
    )
    active_zones = list(set(assignment.values()))

    return OptimizationResult(
        assignment=assignment,
        welfare=welfare,
        active_zones=active_zones,
        total_demand_captured=dem_cap,
        total_travel_cost=travel_cost,
        driver_utilities=utils,
        status=OptimizationStatus.SOLVER_OPTIMAL,
        method_used="ilp_scipy",
        num_evaluations=int(res.get("mip_node_count", 1)),
        details={
            "solver": "scipy.optimize.milp",
            "status_message": res.message,
            "success": res.success,
        },
    )


def _solve_with_pulp(
    driver_list: List[Any],
    zone_list: List[Any],
    loc_map: Dict[Any, Any],
    demand: Mapping[Any, float],
    cost_source: CostSource,
    lambda_cost: float,
) -> Optional[OptimizationResult]:
    """Solve the ILP using PuLP."""
    try:
        import pulp  # type: ignore
    except ImportError:
        return None

    prob = pulp.LpProblem("CentralizedTaxiWelfare", pulp.LpMaximize)

    # Variables
    x = {}
    for d in driver_list:
        for z in zone_list:
            x[(d, z)] = pulp.LpVariable(f"x_{d}_{z}", cat=pulp.LpBinary)

    y = {z: pulp.LpVariable(f"y_{z}", cat=pulp.LpBinary) for z in zone_list}

    # Objective: sum_z D(z)*y_z - lambda * sum_{i,z} c(l_i, z)*x_{i,z}
    obj = pulp.lpSum(demand.get(z, 0.0) * y[z] for z in zone_list)
    cost_term = []
    for d in driver_list:
        loc = loc_map.get(d, d)
        for z in zone_list:
            cost = resolve_travel_cost(d, loc, z, cost_source)
            cost_term.append(lambda_cost * cost * x[(d, z)])

    prob += obj - pulp.lpSum(cost_term)

    # Constraints: sum_z x_{d, z} == 1
    for d in driver_list:
        prob += pulp.lpSum(x[(d, z)] for z in zone_list) == 1

    # Linking constraints: x_{d, z} <= y_z
    for d in driver_list:
        for z in zone_list:
            prob += x[(d, z)] <= y[z]

    # Activation: a zone only counts as active if at least one driver is assigned to it
    for z in zone_list:
        prob += y[z] <= pulp.lpSum(x[(d, z)] for d in driver_list)

    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    assignment = {}
    for d in driver_list:
        for z in zone_list:
            if pulp.value(x[(d, z)]) > 0.5:
                assignment[d] = z
                break

    welfare, dem_cap, travel_cost, utils = evaluate_welfare(
        assignment=assignment,
        driver_locations=loc_map,
        demand=demand,
        cost_source=cost_source,
        lambda_cost=lambda_cost,
    )
    active_zones = list(set(assignment.values()))

    return OptimizationResult(
        assignment=assignment,
        welfare=welfare,
        active_zones=active_zones,
        total_demand_captured=dem_cap,
        total_travel_cost=travel_cost,
        driver_utilities=utils,
        status=OptimizationStatus.SOLVER_OPTIMAL,
        method_used="ilp_pulp",
        num_evaluations=1,
        details={"solver": "pulp_cbc", "status": pulp.LpStatus[prob.status]},
    )


def optimize_welfare_ilp(
    drivers: Sequence[Any],
    driver_locations: Union[Mapping[Any, Any], Sequence[Any]],
    zones: Sequence[Any],
    demand: Mapping[Any, float],
    cost_source: CostSource,
    lambda_cost: float = 1.0,
    fallback_to_heuristic: bool = True,
    seed: Optional[int] = None,
) -> OptimizationResult:
    """
    Solve the centralized welfare problem using Integer Linear Programming.

    If an external solver (SciPy / PuLP / OR-Tools) is present, solves the ILP exactly.
    Otherwise, if fallback_to_heuristic is True, uses the heuristic optimizer and clearly
    reports status=OptimizationStatus.HEURISTIC in the result.

    Args:
        drivers: Sequence of driver identifiers.
        driver_locations: Mapping or sequence of driver locations.
        zones: Sequence of demand zones.
        demand: Mapping of zone -> demand score D(z).
        cost_source: Travel cost matrix or function.
        lambda_cost: Travel cost parameter lambda.
        fallback_to_heuristic: Whether to fallback to heuristic if no ILP solver is installed.
        seed: Random seed for fallback heuristic.

    Returns:
        OptimizationResult with exact solver status or heuristic status.
    """
    driver_list = list(drivers)
    zone_list = list(zones)
    if isinstance(driver_locations, Mapping):
        loc_map = dict(driver_locations)
    else:
        loc_map = {d: loc for d, loc in zip(driver_list, driver_locations)}

    # Check available solvers
    available, solver_name = is_ilp_solver_available()

    if available:
        if solver_name == "scipy":
            res = _solve_with_scipy(driver_list, zone_list, loc_map, demand, cost_source, lambda_cost)
            if res is not None:
                return res
        elif solver_name == "pulp":
            res = _solve_with_pulp(driver_list, zone_list, loc_map, demand, cost_source, lambda_cost)
            if res is not None:
                return res

    # Fallback path
    if fallback_to_heuristic:
        heuristic_res = optimize_welfare_heuristic(
            drivers=drivers,
            driver_locations=loc_map,
            zones=zones,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
            seed=seed,
        )
        heuristic_res.details["ilp_fallback"] = True
        heuristic_res.details["ilp_reason"] = "No external ILP solver library installed"
        return heuristic_res

    raise RuntimeError(
        "No Integer Linear Programming solver library (scipy, pulp, ortools) is installed. "
        "Install scipy or pulp, or enable fallback_to_heuristic=True."
    )


class ILPOptimizer:
    """Class wrapper for the ILP solver adapter."""

    def __init__(self, fallback_to_heuristic: bool = True):
        self.fallback_to_heuristic = fallback_to_heuristic

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
        return optimize_welfare_ilp(
            drivers=drivers,
            driver_locations=driver_locations,
            zones=zones,
            demand=demand,
            cost_source=cost_source,
            lambda_cost=lambda_cost,
            fallback_to_heuristic=self.fallback_to_heuristic,
            seed=seed,
        )
