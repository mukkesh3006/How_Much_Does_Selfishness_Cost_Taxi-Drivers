"""High-level API for generating complete city network instances."""

from __future__ import annotations

from typing import Any

from src.network.core import CityGraphInstance
from src.network.generators.grid import GridGenerator
from src.network.generators.hub_spoke import HubAndSpokeGenerator
from src.network.generators.scale_free import ScaleFreeGenerator
from src.network.shortest_path import compute_all_pairs_shortest_paths
from src.network.validation import validate_graph
from src.network.weights import apply_edge_weights

_GENERATORS = {
    "grid": GridGenerator(),
    "hub_spoke": HubAndSpokeGenerator(),
    "scale_free": ScaleFreeGenerator(),
}


def generate_city_network(
    topology: str,
    num_nodes: int = 16,
    seed: int | None = None,
    min_weight: float = 1.0,
    max_weight: float = 10.0,
    **topology_kwargs: Any,
) -> CityGraphInstance:
    """Generates a complete, validated city network instance with precomputed APSP.

    Args:
        topology: Network topology type: "grid", "hub_spoke", or "scale_free".
        num_nodes: Total number of intersections/nodes in the city (>= 2).
        seed: Random seed for deterministic reproducibility.
        min_weight: Minimum road travel cost (> 0).
        max_weight: Maximum road travel cost (>= min_weight).
        **topology_kwargs: Topology-specific parameters:
            - grid: rows, cols, periodic
            - hub_spoke: num_hubs
            - scale_free: m, m0

    Returns:
        CityGraphInstance bundling the graph, distance matrix, and query helpers.

    Raises:
        ValueError: If topology is unknown or parameters are invalid.
        GraphValidationError: If the resulting graph fails integrity checks.
    """
    normalized_topology = topology.lower().strip()
    if normalized_topology not in _GENERATORS:
        available = ", ".join(f"'{k}'" for k in sorted(_GENERATORS.keys()))
        raise ValueError(
            f"Unknown topology '{topology}'. Available topologies are: {available}"
        )

    generator = _GENERATORS[normalized_topology]

    # 1. Generate unweighted topology with isolated seed
    graph = generator.create(num_nodes=num_nodes, seed=seed, **topology_kwargs)

    # 2. Assign positive random edge weights using derived seed
    weight_seed = (seed + 10007) if seed is not None else None
    apply_edge_weights(
        graph,
        seed=weight_seed,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    # 3. Validate structural and weighting invariants
    validate_graph(graph, check_connected=True, check_weights=True)

    # 4. Compute all-pairs shortest paths using Dijkstra
    distances = compute_all_pairs_shortest_paths(graph)

    # 5. Bundle and return CityGraphInstance
    metadata: dict[str, Any] = {
        "topology": normalized_topology,
        "num_nodes": num_nodes,
        "seed": seed,
        "min_weight": min_weight,
        "max_weight": max_weight,
        **topology_kwargs,
    }

    return CityGraphInstance(
        graph=graph,
        distances=distances,
        metadata=metadata,
    )
