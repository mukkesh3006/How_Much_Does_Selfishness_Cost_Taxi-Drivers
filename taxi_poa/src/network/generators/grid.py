"""2D Grid topology generator for city road networks."""

from __future__ import annotations

import math
import random
from typing import Any

from src.network.core import CityGraph
from src.network.generators.base import BaseGraphGenerator


class GridGenerator(BaseGraphGenerator):
    """Generates a connected 2D grid city network."""

    @staticmethod
    def resolve_dimensions(
        num_nodes: int, rows: int | None = None, cols: int | None = None
    ) -> tuple[int, int]:
        """Resolves (rows, cols) dimensions matching num_nodes."""
        if num_nodes < 2:
            raise ValueError(f"num_nodes must be at least 2, got {num_nodes}")

        if rows is not None and cols is not None:
            if rows <= 0 or cols <= 0:
                raise ValueError(
                    f"rows and cols must be positive, got rows={rows}, cols={cols}"
                )
            if rows * cols != num_nodes:
                raise ValueError(
                    f"rows * cols ({rows} * {cols} = {rows * cols}) does not match num_nodes ({num_nodes})"
                )
            return rows, cols

        if rows is not None:
            if rows <= 0:
                raise ValueError(f"rows must be positive, got rows={rows}")
            if num_nodes % rows != 0:
                raise ValueError(
                    f"num_nodes ({num_nodes}) is not divisible by rows ({rows})"
                )
            return rows, num_nodes // rows

        if cols is not None:
            if cols <= 0:
                raise ValueError(f"cols must be positive, got cols={cols}")
            if num_nodes % cols != 0:
                raise ValueError(
                    f"num_nodes ({num_nodes}) is not divisible by cols ({cols})"
                )
            return num_nodes // cols, cols

        # Closest-to-square factorization (finds largest factor <= sqrt(num_nodes))
        r = int(math.isqrt(num_nodes))
        while r > 1 and (num_nodes % r != 0):
            r -= 1
        c = num_nodes // r
        return r, c

    def generate(
        self,
        num_nodes: int,
        rng: random.Random,
        rows: int | None = None,
        cols: int | None = None,
        periodic: bool = False,
        **kwargs: Any,
    ) -> CityGraph:
        """Generates a connected 2D grid graph.

        Args:
            num_nodes: Total number of nodes in the grid.
            rng: Isolated random generator instance.
            rows: Optional number of rows.
            cols: Optional number of columns.
            periodic: If True, connects boundaries to form a torus.
            **kwargs: Additional parameters passed to metadata.

        Returns:
            CityGraph instance with 2D grid connectivity and node coordinates.
        """
        resolved_rows, resolved_cols = self.resolve_dimensions(
            num_nodes=num_nodes, rows=rows, cols=cols
        )

        metadata: dict[str, Any] = {
            "rows": resolved_rows,
            "cols": resolved_cols,
            "periodic": periodic,
            **kwargs,
        }

        graph = CityGraph(num_nodes=num_nodes, metadata=metadata)

        for r in range(resolved_rows):
            for c in range(resolved_cols):
                u = r * resolved_cols + c
                # Store 2D coordinates: x = col, y = row
                graph.set_position(u, float(c), float(r))

                # Horizontal edge (right neighbor)
                if c + 1 < resolved_cols:
                    v = r * resolved_cols + (c + 1)
                    graph.add_edge(u, v, weight=1.0)
                elif periodic and resolved_cols > 2:
                    v = r * resolved_cols + 0
                    if not graph.has_edge(u, v):
                        graph.add_edge(u, v, weight=1.0)

                # Vertical edge (down neighbor)
                if r + 1 < resolved_rows:
                    v = (r + 1) * resolved_cols + c
                    graph.add_edge(u, v, weight=1.0)
                elif periodic and resolved_rows > 2:
                    v = 0 * resolved_cols + c
                    if not graph.has_edge(u, v):
                        graph.add_edge(u, v, weight=1.0)

        return graph
