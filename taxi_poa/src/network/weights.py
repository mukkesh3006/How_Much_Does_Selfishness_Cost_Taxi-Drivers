"""Edge weight generation for city network topologies."""

from __future__ import annotations

import math
import random

from src.network.core import CityGraph


class UniformWeightModel:
    """Samples edge travel costs independently from Uniform(min_weight, max_weight)."""

    def __init__(self, min_weight: float = 1.0, max_weight: float = 10.0) -> None:
        if math.isnan(min_weight) or math.isinf(min_weight):
            raise ValueError(f"min_weight must be a finite number, got {min_weight}")
        if math.isnan(max_weight) or math.isinf(max_weight):
            raise ValueError(f"max_weight must be a finite number, got {max_weight}")
        if min_weight <= 0:
            raise ValueError(
                f"min_weight must be strictly positive (> 0), got {min_weight}"
            )
        if max_weight < min_weight:
            raise ValueError(
                f"max_weight ({max_weight}) must be >= min_weight ({min_weight})"
            )

        self.min_weight: float = float(min_weight)
        self.max_weight: float = float(max_weight)

    def apply(self, graph: CityGraph, seed: int | None = None) -> CityGraph:
        """Assigns uniform edge weights to existing edges using an isolated RNG."""
        rng = random.Random(seed)

        # Iterate over unique undirected edges (u < v)
        for u, v, _ in graph.edges():
            weight = rng.uniform(self.min_weight, self.max_weight)
            graph.adj[u][v] = weight
            graph.adj[v][u] = weight

        graph.metadata.update(
            {
                "weight_model": "UniformWeightModel",
                "weight_seed": seed,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
            }
        )
        return graph


def apply_edge_weights(
    graph: CityGraph,
    seed: int | None = None,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
) -> CityGraph:
    """Applies uniform random edge weights in [min_weight, max_weight] to graph edges.

    Args:
        graph: CityGraph whose existing edges will be weighted.
        seed: Random seed for reproducible weight assignment.
        min_weight: Minimum travel cost (must be > 0).
        max_weight: Maximum travel cost (must be >= min_weight).

    Returns:
        The CityGraph with updated edge weights.
    """
    model = UniformWeightModel(min_weight=min_weight, max_weight=max_weight)
    return model.apply(graph=graph, seed=seed)
