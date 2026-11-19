"""
Cost helper utilities and graph adapters for travel cost calculations.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union


def create_mock_cost_matrix(
    locations: Sequence[Any],
    zones: Sequence[Any],
    min_cost: float = 1.0,
    max_cost: float = 10.0,
    seed: Optional[int] = None,
) -> Dict[Tuple[Any, Any], float]:
    """
    Generate a synthetic deterministic travel cost lookup table for testing.

    Args:
        locations: Sequence of driver location nodes.
        zones: Sequence of demand zone nodes.
        min_cost: Minimum travel distance/cost.
        max_cost: Maximum travel distance/cost.
        seed: Random seed for reproducibility.

    Returns:
        Dict[(loc, zone), float]: Mapping of (location, zone) -> positive travel cost.
    """
    rng = random.Random(seed)
    cost_matrix: Dict[Tuple[Any, Any], float] = {}
    for loc in locations:
        for z in zones:
            if loc == z:
                cost_matrix[(loc, z)] = 0.0
            else:
                cost_matrix[(loc, z)] = round(rng.uniform(min_cost, max_cost), 2)
    return cost_matrix


def create_euclidean_cost_fn(
    coordinates: Mapping[Any, Tuple[float, float]]
) -> Callable[[Any, Any], float]:
    """
    Create a travel cost function based on 2D coordinates.

    Args:
        coordinates: Mapping of node_id -> (x, y) coordinates.

    Returns:
        Callable[[node1, node2], float]: Euclidean distance function.
    """
    def euclidean_dist(u: Any, v: Any) -> float:
        if u == v:
            return 0.0
        if u not in coordinates:
            raise KeyError(f"Node '{u}' not found in coordinates mapping.")
        if v not in coordinates:
            raise KeyError(f"Node '{v}' not found in coordinates mapping.")
        x1, y1 = coordinates[u]
        x2, y2 = coordinates[v]
        return math.hypot(x1 - x2, y1 - y2)

    return euclidean_dist


def adapter_from_graph_shortest_paths(
    shortest_paths: Union[Mapping[Any, Mapping[Any, float]], Mapping[Tuple[Any, Any], float]]
) -> Callable[[Any, Any], float]:
    """
    Adapter to consume shortest path outputs from Mukkesh's graph module.

    Args:
        shortest_paths: Precomputed all-pairs shortest paths as nested dict or tuple-key dict.

    Returns:
        Callable[[u, v], float]: Shortest path distance lookup function c(u, v).
    """
    def get_distance(u: Any, v: Any) -> float:
        if u == v:
            return 0.0
        if (u, v) in shortest_paths:
            return float(shortest_paths[(u, v)])  # type: ignore
        if u in shortest_paths and isinstance(shortest_paths[u], Mapping):
            if v in shortest_paths[u]:
                return float(shortest_paths[u][v])
        raise KeyError(f"Shortest path distance between '{u}' and '{v}' not found in graph data.")

    return get_distance
