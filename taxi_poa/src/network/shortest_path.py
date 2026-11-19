"""All-pairs shortest-path computation for city network topologies."""

from __future__ import annotations

import heapq

from src.network.core import CityGraph, DistanceMatrix
from src.network.validation import validate_graph


def _dijkstra_single_source(
    graph: CityGraph, source: int
) -> tuple[list[float], list[int | None]]:
    """Runs Dijkstra's algorithm from a single source node using a min-heap."""
    n = graph.num_nodes
    dist: list[float] = [float("inf")] * n
    pred: list[int | None] = [None] * n

    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        for v, weight in graph.adj[u].items():
            new_dist = d + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                pred[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, pred


def compute_all_pairs_shortest_paths(graph: CityGraph) -> DistanceMatrix:
    """Precomputes all-pairs shortest travel costs and paths using Dijkstra's algorithm.

    Runs Dijkstra's algorithm from every node as source, utilizing binary min-heaps.
    Validates graph connectivity and weight positivity before execution.

    Args:
        graph: Connected CityGraph with strictly positive edge weights.

    Returns:
        DistanceMatrix instance supporting O(1) distance lookups and path reconstruction.

    Raises:
        DisconnectedGraphError: If the graph has multiple disconnected components.
        InvalidWeightError: If any edge weight is non-positive, NaN, or infinite.
        GraphValidationError: If the graph structure is invalid.
    """
    validate_graph(graph, check_connected=True, check_weights=True)

    n = graph.num_nodes
    dist_matrix: list[list[float]] = []
    pred_matrix: list[list[int | None]] = []

    for u in range(n):
        dist, pred = _dijkstra_single_source(graph, u)
        dist_matrix.append(dist)
        pred_matrix.append(pred)

    return DistanceMatrix(matrix=dist_matrix, predecessors=pred_matrix)


def floyd_warshall_reference(graph: CityGraph) -> list[list[float]]:
    """O(V^3) Floyd-Warshall reference implementation for testing/cross-verification."""
    n = graph.num_nodes
    dist = [[float("inf")] * n for _ in range(n)]

    for u in range(n):
        dist[u][u] = 0.0
        for v, weight in graph.adj[u].items():
            dist[u][v] = min(dist[u][v], weight)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist
