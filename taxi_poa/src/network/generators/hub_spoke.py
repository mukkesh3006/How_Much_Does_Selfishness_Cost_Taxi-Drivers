"""Hub-and-spoke topology generator for city road networks."""

from __future__ import annotations

import math
import random
from typing import Any

from src.network.core import CityGraph
from src.network.generators.base import BaseGraphGenerator


class HubAndSpokeGenerator(BaseGraphGenerator):
    """Generates a connected hub-and-spoke city network."""

    @staticmethod
    def get_hubs(graph: CityGraph) -> list[int]:
        """Returns the list of hub node IDs from graph metadata."""
        return list(graph.metadata.get("hubs", []))

    @staticmethod
    def get_spokes(graph: CityGraph) -> list[int]:
        """Returns the list of spoke node IDs from graph metadata."""
        return list(graph.metadata.get("spokes", []))

    @staticmethod
    def is_hub(graph: CityGraph, node: int) -> bool:
        """Returns True if the given node ID is a central hub."""
        return node in graph.metadata.get("hubs", [])

    def generate(
        self,
        num_nodes: int,
        rng: random.Random,
        num_hubs: int = 1,
        **kwargs: Any,
    ) -> CityGraph:
        """Generates a connected hub-and-spoke graph.

        Args:
            num_nodes: Total number of nodes in the network (>= 2).
            rng: Isolated random generator instance.
            num_hubs: Number of central hub nodes (1 <= num_hubs < num_nodes).
            **kwargs: Additional parameters passed to metadata.

        Returns:
            CityGraph with hub-and-spoke connectivity and 2D coordinates.
        """
        if num_nodes < 2:
            raise ValueError(f"num_nodes must be at least 2, got {num_nodes}")
        if num_hubs < 1:
            raise ValueError(f"num_hubs must be at least 1, got {num_hubs}")
        if num_hubs >= num_nodes:
            raise ValueError(
                f"num_hubs ({num_hubs}) must be strictly less than num_nodes ({num_nodes})"
            )

        hubs = list(range(num_hubs))
        spokes = list(range(num_hubs, num_nodes))

        hub_assignments: dict[int, int] = {
            s: (i % num_hubs) for i, s in enumerate(spokes)
        }

        metadata: dict[str, Any] = {
            "num_hubs": num_hubs,
            "hubs": hubs,
            "spokes": spokes,
            "hub_assignments": hub_assignments,
            **kwargs,
        }

        graph = CityGraph(num_nodes=num_nodes, metadata=metadata)

        # 1. Connected hub backbone (for multiple hubs, fully interconnect hubs)
        for h1 in range(num_hubs):
            for h2 in range(h1 + 1, num_hubs):
                graph.add_edge(h1, h2, weight=1.0)

        # 2. Connect each spoke to its designated primary hub
        for s in spokes:
            h = hub_assignments[s]
            graph.add_edge(h, s, weight=1.0)

        # 3. Store 2D coordinates for visual representation
        if num_hubs == 1:
            # Single hub at the center; spokes arranged in a circle
            graph.set_position(0, 0.0, 0.0)
            num_spokes = len(spokes)
            radius = 10.0
            for i, s in enumerate(spokes):
                theta = (2.0 * math.pi * i) / num_spokes
                x = radius * math.cos(theta)
                y = radius * math.sin(theta)
                graph.set_position(s, round(x, 4), round(y, 4))
        else:
            # Multiple hubs in an inner circle; spokes clustered around their assigned hub
            hub_radius = 5.0
            hub_positions: dict[int, tuple[float, float]] = {}
            for h in range(num_hubs):
                theta_h = (2.0 * math.pi * h) / num_hubs
                hx = hub_radius * math.cos(theta_h)
                hy = hub_radius * math.sin(theta_h)
                graph.set_position(h, round(hx, 4), round(hy, 4))
                hub_positions[h] = (hx, hy)

            spokes_by_hub: dict[int, list[int]] = {h: [] for h in range(num_hubs)}
            for s in spokes:
                spokes_by_hub[hub_assignments[s]].append(s)

            spoke_distance = 6.0
            for h in range(num_hubs):
                hx, hy = hub_positions[h]
                cluster = spokes_by_hub[h]
                count = len(cluster)
                theta_h = (2.0 * math.pi * h) / num_hubs
                arc = (2.0 * math.pi / num_hubs) * 0.75
                for j, s in enumerate(cluster):
                    if count == 1:
                        angle = theta_h
                    else:
                        angle = theta_h - (arc / 2.0) + (arc * j) / (count - 1)
                    sx = hx + spoke_distance * math.cos(angle)
                    sy = hy + spoke_distance * math.sin(angle)
                    graph.set_position(s, round(sx, 4), round(sy, 4))

        return graph
