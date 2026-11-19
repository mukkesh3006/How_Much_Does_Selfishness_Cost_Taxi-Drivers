"""Base generator interface for synthetic city network topologies."""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any

from src.network.core import CityGraph


class BaseGraphGenerator(ABC):
    """Abstract base class for all city graph topology generators."""

    @abstractmethod
    def generate(self, num_nodes: int, rng: random.Random, **kwargs: Any) -> CityGraph:
        """Generates an unweighted or default-weighted graph topology.

        Must be implemented by concrete generator subclasses.
        """
        raise NotImplementedError

    def create(self, num_nodes: int, seed: int | None = None, **kwargs: Any) -> CityGraph:
        """Template method for generating a graph with reproducible isolated RNG."""
        if num_nodes < 2:
            raise ValueError(f"City networks require at least 2 nodes, got {num_nodes}")

        rng = random.Random(seed)
        graph = self.generate(num_nodes=num_nodes, rng=rng, **kwargs)

        graph.metadata.update(
            {
                "generator": self.__class__.__name__,
                "seed": seed,
                "params": kwargs,
            }
        )
        return graph
