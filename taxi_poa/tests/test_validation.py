"""Unit tests for graph validation and integrity checking."""

import math
import unittest

from src.network.core import CityGraph
from src.network.validation import (
    DisconnectedGraphError,
    GraphValidationError,
    InvalidTopologyError,
    InvalidWeightError,
    NodeIndexError,
    is_connected,
    validate_graph,
)


class TestGraphValidation(unittest.TestCase):
    def test_valid_graph_passes(self):
        g = CityGraph(num_nodes=3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 2.0)
        self.assertTrue(validate_graph(g))

    def test_insufficient_nodes_raises(self):
        g = CityGraph(num_nodes=1)
        with self.assertRaises(GraphValidationError):
            validate_graph(g)

    def test_is_connected_helper(self):
        g = CityGraph(num_nodes=4)
        g.add_edge(0, 1, 1.0)
        g.add_edge(2, 3, 1.0)
        self.assertFalse(is_connected(g))

        g.add_edge(1, 2, 1.0)
        self.assertTrue(is_connected(g))

    def test_disconnected_graph_raises(self):
        g = CityGraph(num_nodes=4)
        g.add_edge(0, 1, 1.0)
        g.add_edge(2, 3, 1.0)
        with self.assertRaises(DisconnectedGraphError):
            validate_graph(g)

    def test_zero_weight_raises(self):
        g = CityGraph(num_nodes=2)
        g.add_edge(0, 1, 0.0)
        with self.assertRaises(InvalidWeightError):
            validate_graph(g)

    def test_negative_weight_raises(self):
        g = CityGraph(num_nodes=2)
        g.add_edge(0, 1, -2.5)
        with self.assertRaises(InvalidWeightError):
            validate_graph(g)

    def test_nan_weight_raises(self):
        g = CityGraph(num_nodes=2)
        g.add_edge(0, 1, float("nan"))
        with self.assertRaises(InvalidWeightError):
            validate_graph(g)

    def test_inf_weight_raises(self):
        g = CityGraph(num_nodes=2)
        g.add_edge(0, 1, float("inf"))
        with self.assertRaises(InvalidWeightError):
            validate_graph(g)

    def test_self_loop_raises(self):
        g = CityGraph(num_nodes=3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        # Manually inject self-loop into adj
        g.adj[0][0] = 1.0
        with self.assertRaises(InvalidTopologyError):
            validate_graph(g)

    def test_asymmetric_edge_raises(self):
        g = CityGraph(num_nodes=3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        # Remove back-edge
        del g.adj[1][0]
        with self.assertRaises(InvalidTopologyError):
            validate_graph(g)

    def test_asymmetric_weight_raises(self):
        g = CityGraph(num_nodes=3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        # Modify back-edge weight
        g.adj[1][0] = 5.0
        with self.assertRaises(InvalidTopologyError):
            validate_graph(g)

    def test_corrupted_node_indices_raises(self):
        g = CityGraph(num_nodes=3)
        # Corrupt node keys (e.g. key 99 instead of 2)
        del g.adj[2]
        g.adj[99] = {}
        with self.assertRaises(NodeIndexError):
            validate_graph(g)


if __name__ == "__main__":
    unittest.main()
