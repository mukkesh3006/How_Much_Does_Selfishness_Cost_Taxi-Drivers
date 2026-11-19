"""Graph validation and integrity checking for city network topologies."""

from __future__ import annotations

import collections
import math

from src.network.core import CityGraph


class GraphValidationError(Exception):
    """Base exception for all graph validation errors."""


class DisconnectedGraphError(GraphValidationError):
    """Raised when the graph is not a single connected component."""


class InvalidWeightError(GraphValidationError):
    """Raised when edge weights are non-positive, NaN, or infinite."""


class InvalidTopologyError(GraphValidationError):
    """Raised when graph structure violates invariants (e.g. self-loops, asymmetry)."""


class NodeIndexError(GraphValidationError):
    """Raised when node indices are non-contiguous or do not match 0..n-1."""


def is_connected(graph: CityGraph) -> bool:
    """Checks whether the graph forms a single connected component via BFS."""
    if graph.num_nodes <= 1:
        return True

    visited: set[int] = set()
    queue: collections.deque[int] = collections.deque([0])
    visited.add(0)

    while queue:
        curr = queue.popleft()
        for neighbor in graph.adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == graph.num_nodes


def validate_graph(
    graph: CityGraph,
    check_connected: bool = True,
    check_weights: bool = True,
    min_nodes: int = 2,
) -> bool:
    """Validates that a CityGraph satisfies all structural and weighting invariants.

    Raises a subclass of GraphValidationError if any check fails.
    Returns True if the graph is valid.
    """
    if graph.num_nodes < min_nodes:
        raise GraphValidationError(
            f"Graph has {graph.num_nodes} nodes, minimum required is {min_nodes}"
        )

    # 1. Validate contiguous node indices [0, 1, ..., n-1]
    expected_nodes = set(range(graph.num_nodes))
    actual_nodes = set(graph.adj.keys())
    if actual_nodes != expected_nodes:
        missing = expected_nodes - actual_nodes
        extra = actual_nodes - expected_nodes
        raise NodeIndexError(
            f"Graph nodes must strictly match 0..{graph.num_nodes - 1}. "
            f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
        )

    # 2. Validate topology (no self-loops, undirected symmetry)
    for u in range(graph.num_nodes):
        neighbors = graph.adj[u]
        if u in neighbors:
            raise InvalidTopologyError(f"Self-loop detected on node {u}")

        for v, weight in neighbors.items():
            if not (0 <= v < graph.num_nodes):
                raise NodeIndexError(
                    f"Edge ({u}, {v}) references non-existent node {v}"
                )
            if u not in graph.adj[v]:
                raise InvalidTopologyError(
                    f"Asymmetric edge: ({u}, {v}) exists but reverse edge ({v}, {u}) does not"
                )

            reverse_weight = graph.adj[v][u]
            if check_weights:
                if math.isnan(weight) or math.isnan(reverse_weight):
                    raise InvalidWeightError(f"Edge ({u}, {v}) has NaN weight")
                if math.isinf(weight) or math.isinf(reverse_weight):
                    raise InvalidWeightError(f"Edge ({u}, {v}) has infinite weight")
                if weight <= 0 or reverse_weight <= 0:
                    raise InvalidWeightError(
                        f"Edge ({u}, {v}) must have strictly positive weight, got {weight}"
                    )

            if reverse_weight != weight:
                raise InvalidTopologyError(
                    f"Asymmetric edge weight between {u} and {v}: "
                    f"w({u}, {v}) = {weight} != w({v}, {u}) = {reverse_weight}"
                )

    # 4. Validate connectivity
    if check_connected and not is_connected(graph):
        raise DisconnectedGraphError(
            "Graph is disconnected. A city network must have a single connected component."
        )

    return True
