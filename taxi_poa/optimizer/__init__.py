"""
Centralized Welfare Optimizer Package for Spatial Taxi Repositioning.

Implements the centralized welfare-maximizing benchmark:
    a_OPT = argmax_a sum_i U_i(a; l)
    W_OPT = W(a_OPT) = sum_{z in active(a)} D(z) - lambda * sum_i c(l_i, a_i)
"""

from optimizer.base import (
    OptimizationResult,
    OptimizationStatus,
    CostSource,
    evaluate_welfare,
    resolve_travel_cost,
)
from optimizer.exact import optimize_welfare_exact, ExactEnumerationOptimizer
from optimizer.heuristic import optimize_welfare_heuristic, GreedyLocalSearchOptimizer
from optimizer.ilp import optimize_welfare_ilp, ILPOptimizer, is_ilp_solver_available
from optimizer.solver import optimize_welfare, CentralizedOptimizer
from optimizer.cost import (
    create_mock_cost_matrix,
    create_euclidean_cost_fn,
    adapter_from_graph_shortest_paths,
)

__all__ = [
    "OptimizationResult",
    "OptimizationStatus",
    "CostSource",
    "evaluate_welfare",
    "resolve_travel_cost",
    "optimize_welfare",
    "optimize_welfare_exact",
    "optimize_welfare_heuristic",
    "optimize_welfare_ilp",
    "ExactEnumerationOptimizer",
    "GreedyLocalSearchOptimizer",
    "ILPOptimizer",
    "CentralizedOptimizer",
    "is_ilp_solver_available",
    "create_mock_cost_matrix",
    "create_euclidean_cost_fn",
    "adapter_from_graph_shortest_paths",
]
