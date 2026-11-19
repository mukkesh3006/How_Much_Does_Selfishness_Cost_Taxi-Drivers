"""City network infrastructure module for taxi allocation."""

from src.network.api import generate_city_network
from src.network.core import CityGraph, CityGraphInstance, DistanceMatrix
from src.network.generators.base import BaseGraphGenerator
from src.network.generators.grid import GridGenerator
from src.network.generators.hub_spoke import HubAndSpokeGenerator
from src.network.generators.scale_free import ScaleFreeGenerator
from src.network.shortest_path import (
    compute_all_pairs_shortest_paths,
    floyd_warshall_reference,
)
from src.network.validation import (
    DisconnectedGraphError,
    GraphValidationError,
    InvalidTopologyError,
    InvalidWeightError,
    NodeIndexError,
    is_connected,
    validate_graph,
)
from src.network.weights import UniformWeightModel, apply_edge_weights

__all__ = [
    "CityGraph",
    "DistanceMatrix",
    "CityGraphInstance",
    "generate_city_network",
    "BaseGraphGenerator",
    "GridGenerator",
    "HubAndSpokeGenerator",
    "ScaleFreeGenerator",
    "UniformWeightModel",
    "apply_edge_weights",
    "compute_all_pairs_shortest_paths",
    "floyd_warshall_reference",
    "GraphValidationError",
    "DisconnectedGraphError",
    "InvalidWeightError",
    "InvalidTopologyError",
    "NodeIndexError",
    "is_connected",
    "validate_graph",
]
