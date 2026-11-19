"""Core graph data representation for city network topologies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CityGraph:
    """Represents a city road network as an undirected, weighted graph."""

    def __init__(
        self,
        num_nodes: int,
        positions: dict[int, tuple[float, float]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}")

        self.num_nodes: int = num_nodes
        self.adj: dict[int, dict[int, float]] = {i: {} for i in range(num_nodes)}
        self.positions: dict[int, tuple[float, float]] = (
            dict(positions) if positions is not None else {}
        )
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}

    def _check_node_index(self, u: int) -> None:
        if not (0 <= u < self.num_nodes):
            raise IndexError(
                f"Node index {u} out of range for graph with {self.num_nodes} nodes"
            )

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """Adds an undirected edge between u and v with the given weight."""
        self._check_node_index(u)
        self._check_node_index(v)
        if u == v:
            raise ValueError(f"Self-loops are not allowed: ({u}, {v})")

        self.adj[u][v] = float(weight)
        self.adj[v][u] = float(weight)

    def has_edge(self, u: int, v: int) -> bool:
        """Returns True if there is an edge between u and v."""
        if 0 <= u < self.num_nodes and 0 <= v < self.num_nodes:
            return v in self.adj[u]
        return False

    def get_weight(self, u: int, v: int) -> float:
        """Returns edge weight between u and v."""
        self._check_node_index(u)
        self._check_node_index(v)
        if v not in self.adj[u]:
            raise KeyError(f"No edge exists between node {u} and node {v}")
        return self.adj[u][v]

    def neighbors(self, u: int) -> list[int]:
        """Returns sorted list of neighbors for node u."""
        self._check_node_index(u)
        return sorted(self.adj[u].keys())

    def degree(self, u: int) -> int:
        """Returns degree (number of neighbors) of node u."""
        self._check_node_index(u)
        return len(self.adj[u])

    def nodes(self) -> list[int]:
        """Returns list of all node indices [0, 1, ..., num_nodes - 1]."""
        return list(range(self.num_nodes))

    def edges(self) -> list[tuple[int, int, float]]:
        """Returns list of unique undirected edges as (u, v, weight) with u < v."""
        result: list[tuple[int, int, float]] = []
        for u in range(self.num_nodes):
            for v, weight in self.adj[u].items():
                if u < v:
                    result.append((u, v, weight))
        result.sort(key=lambda e: (e[0], e[1]))
        return result

    def set_position(self, u: int, x: float, y: float) -> None:
        """Sets the 2D spatial coordinates for node u."""
        self._check_node_index(u)
        self.positions[u] = (float(x), float(y))

    def get_position(self, u: int) -> tuple[float, float] | None:
        """Gets the 2D spatial coordinates for node u, or None if unset."""
        self._check_node_index(u)
        return self.positions.get(u)

    def copy(self) -> CityGraph:
        """Returns a copy of the graph."""
        copied = CityGraph(
            num_nodes=self.num_nodes,
            positions=self.positions,
            metadata=self.metadata,
        )
        for u, v, weight in self.edges():
            copied.add_edge(u, v, weight)
        return copied


class DistanceMatrix:
    """Precomputed all-pairs shortest path distance lookup table."""

    def __init__(
        self,
        matrix: list[list[float]],
        predecessors: list[list[int | None]] | None = None,
    ) -> None:
        if not matrix or not matrix[0]:
            raise ValueError("Distance matrix cannot be empty")
        n = len(matrix)
        for row in matrix:
            if len(row) != n:
                raise ValueError(
                    f"Distance matrix must be square, got row length {len(row)} != {n}"
                )
        self._matrix: list[list[float]] = [[float(val) for val in row] for row in matrix]
        self._predecessors: list[list[int | None]] | None = predecessors
        self._num_nodes: int = n

    @property
    def num_nodes(self) -> int:
        """Number of nodes in distance matrix."""
        return self._num_nodes

    def get_distance(self, u: int, v: int) -> float:
        """Returns precomputed shortest-path travel distance between u and v in O(1)."""
        if not (0 <= u < self._num_nodes and 0 <= v < self._num_nodes):
            raise IndexError(
                f"Indices ({u}, {v}) out of bounds for matrix of size {self._num_nodes}"
            )
        return self._matrix[u][v]

    def get_path(self, u: int, v: int) -> list[int]:
        """Reconstructs shortest path from node u to node v as a list [u, ..., v]."""
        if not (0 <= u < self._num_nodes and 0 <= v < self._num_nodes):
            raise IndexError(
                f"Indices ({u}, {v}) out of bounds for matrix of size {self._num_nodes}"
            )
        if u == v:
            return [u]
        if self._predecessors is None:
            raise ValueError(
                "Predecessor matrix was not provided for path reconstruction"
            )

        path: list[int] = []
        curr: int | None = v
        while curr is not None and curr != u:
            path.append(curr)
            curr = self._predecessors[u][curr]

        if curr is None:
            return []
        path.append(u)
        path.reverse()
        return path

    def to_list(self) -> list[list[float]]:
        """Returns raw 2D list of distances."""
        return [list(row) for row in self._matrix]

    def to_numpy(self) -> Any:
        """Exports distance matrix as NumPy ndarray if NumPy is installed."""
        try:
            import numpy as np

            return np.array(self._matrix, dtype=float)
        except ImportError as err:
            raise ImportError(
                "NumPy is not installed. Use .to_list() or install numpy."
            ) from err


@dataclass
class CityGraphInstance:
    """Bundles a city graph with its precomputed shortest-path distances."""

    graph: CityGraph
    distances: DistanceMatrix | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_travel_cost(self, u: int, v: int) -> float:
        """Returns travel cost between node u and v from precomputed distance matrix."""
        if self.distances is None:
            raise ValueError(
                "Distance matrix has not been computed for this city graph instance."
            )
        return self.distances.get_distance(u, v)

    def get_route(self, u: int, v: int) -> list[int]:
        """Returns sequence of node IDs along shortest path from node u to node v."""
        if self.distances is None:
            raise ValueError(
                "Distance matrix has not been computed for this city graph instance."
            )
        return self.distances.get_path(u, v)

    def get_neighbors(self, u: int) -> list[int]:
        """Returns adjacent nodes to node u."""
        return self.graph.neighbors(u)

    def get_coordinates(self, u: int) -> tuple[float, float] | None:
        """Returns (x, y) coordinates of node u, if available."""
        return self.graph.get_position(u)
