"""Scale-free topology generator using preferential attachment (Barabási-Albert model)."""

from __future__ import annotations

import math
import random
from typing import Any

from src.network.core import CityGraph
from src.network.generators.base import BaseGraphGenerator


class ScaleFreeGenerator(BaseGraphGenerator):
    """Generates a connected scale-free city network using preferential attachment."""

    def generate(
        self,
        num_nodes: int,
        rng: random.Random,
        m: int = 2,
        m0: int | None = None,
        **kwargs: Any,
    ) -> CityGraph:
        """Generates a connected scale-free graph.

        Args:
            num_nodes: Total number of nodes in the graph (>= 2).
            rng: Isolated random generator instance.
            m: Number of edges each new node adds (1 <= m < num_nodes).
            m0: Size of initial connected seed graph (m <= m0 <= num_nodes).
                Defaults to m + 1.
            **kwargs: Additional parameters passed to metadata.

        Returns:
            CityGraph with scale-free connectivity and 2D coordinates.
        """
        # 1. Parameter validation
        if num_nodes < 2:
            raise ValueError(f"num_nodes must be at least 2, got {num_nodes}")
        if m < 1:
            raise ValueError(f"m must be at least 1, got {m}")
        if m >= num_nodes:
            raise ValueError(f"m ({m}) must be strictly less than num_nodes ({num_nodes})")

        resolved_m0 = m + 1 if m0 is None else m0
        if resolved_m0 < m:
            raise ValueError(f"seed graph size m0 ({resolved_m0}) must be >= m ({m})")
        if resolved_m0 > num_nodes:
            raise ValueError(
                f"seed graph size m0 ({resolved_m0}) cannot exceed num_nodes ({num_nodes})"
            )

        metadata: dict[str, Any] = {
            "m": m,
            "m0": resolved_m0,
            **kwargs,
        }

        graph = CityGraph(num_nodes=num_nodes, metadata=metadata)

        # 2. Build connected seed graph K_{m0} (clique on nodes 0..m0-1)
        repeated_nodes: list[int] = []
        for u in range(resolved_m0):
            for v in range(u + 1, resolved_m0):
                graph.add_edge(u, v, weight=1.0)
                repeated_nodes.append(u)
                repeated_nodes.append(v)

        # 3. Preferential attachment for nodes m0..num_nodes-1
        for new_node in range(resolved_m0, num_nodes):
            targets: set[int] = set()
            while len(targets) < m:
                target = rng.choice(repeated_nodes)
                targets.add(target)

            for target in targets:
                graph.add_edge(new_node, target, weight=1.0)

            # Update preferential attachment sampling pool
            repeated_nodes.extend(targets)
            repeated_nodes.extend([new_node] * m)

        # 4. Compute 2D coordinates
        # Radial layout: core high-degree hubs in inner circle, others on outer circle
        sorted_by_degree = sorted(
            range(num_nodes), key=lambda u: graph.degree(u), reverse=True
        )

        core_count = max(3, int(num_nodes * 0.15))
        core_nodes = set(sorted_by_degree[:core_count])
        outer_nodes = [u for u in sorted_by_degree if u not in core_nodes]

        # Inner ring for hubs
        inner_radius = 4.0
        for i, u in enumerate(sorted_by_degree[:core_count]):
            theta = (2.0 * math.pi * i) / core_count
            x = inner_radius * math.cos(theta)
            y = inner_radius * math.sin(theta)
            graph.set_position(u, round(x, 4), round(y, 4))

        # Outer ring for peripheral nodes
        outer_radius = 10.0
        outer_count = len(outer_nodes)
        for i, u in enumerate(outer_nodes):
            theta = (2.0 * math.pi * i) / outer_count
            x = outer_radius * math.cos(theta)
            y = outer_radius * math.sin(theta)
            graph.set_position(u, round(x, 4), round(y, 4))

        return graph
