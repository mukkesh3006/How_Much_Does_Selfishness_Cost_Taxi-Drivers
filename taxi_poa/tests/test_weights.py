"""Unit tests for edge-weight generation module."""

import math
import unittest

from src.network.generators.grid import GridGenerator
from src.network.generators.hub_spoke import HubAndSpokeGenerator
from src.network.validation import validate_graph
from src.network.weights import UniformWeightModel, apply_edge_weights


class TestEdgeWeights(unittest.TestCase):
    def setUp(self):
        # Generate a small 3x3 grid graph to test weighting
        self.grid_generator = GridGenerator()
        self.hub_generator = HubAndSpokeGenerator()

    def test_every_edge_receives_weight(self):
        """1. Every edge receives a newly sampled weight."""
        graph = self.grid_generator.create(num_nodes=9, rows=3, cols=3)
        # Default unweighted edges have weight 1.0
        apply_edge_weights(graph, seed=42, min_weight=2.0, max_weight=5.0)

        for u, v, weight in graph.edges():
            self.assertGreaterEqual(weight, 2.0)
            self.assertLessEqual(weight, 5.0)

    def test_all_weights_strictly_positive(self):
        """2. All weights are strictly positive (> 0)."""
        graph = self.hub_generator.create(num_nodes=10, num_hubs=2, seed=12)
        apply_edge_weights(graph, seed=99, min_weight=0.1, max_weight=1.0)

        for u, v, weight in graph.edges():
            self.assertGreater(weight, 0.0)

    def test_all_weights_finite(self):
        """3. All weights are finite (no NaN or infinity)."""
        graph = self.grid_generator.create(num_nodes=16, rows=4, cols=4)
        apply_edge_weights(graph, seed=123, min_weight=1.0, max_weight=10.0)

        for u, v, weight in graph.edges():
            self.assertFalse(math.isnan(weight))
            self.assertFalse(math.isinf(weight))

    def test_symmetry_of_weights(self):
        """4. Edge weights are symmetric for undirected graph: w(u, v) == w(v, u)."""
        graph = self.hub_generator.create(num_nodes=12, num_hubs=3, seed=55)
        apply_edge_weights(graph, seed=77, min_weight=1.5, max_weight=8.5)

        for u in graph.nodes():
            for v, weight in graph.adj[u].items():
                self.assertEqual(graph.adj[v][u], weight)

        # Full validation check passes
        self.assertTrue(validate_graph(graph))

    def test_weights_remain_within_bounds(self):
        """5. Sampled weights remain strictly within [min_weight, max_weight]."""
        graph = self.grid_generator.create(num_nodes=25, rows=5, cols=5)
        min_w, max_w = 3.5, 7.5
        apply_edge_weights(graph, seed=888, min_weight=min_w, max_weight=max_w)

        for _, _, weight in graph.edges():
            self.assertGreaterEqual(weight, min_w)
            self.assertLessEqual(weight, max_w)

    def test_same_seed_produces_identical_weights(self):
        """6. Applying weights with the same seed produces identical results."""
        g1 = self.grid_generator.create(num_nodes=16, rows=4, cols=4)
        g2 = self.grid_generator.create(num_nodes=16, rows=4, cols=4)

        apply_edge_weights(g1, seed=42, min_weight=1.0, max_weight=10.0)
        apply_edge_weights(g2, seed=42, min_weight=1.0, max_weight=10.0)

        self.assertEqual(g1.edges(), g2.edges())

    def test_different_seeds_produce_different_weights(self):
        """7. Applying weights with different seeds produces different results."""
        g1 = self.grid_generator.create(num_nodes=16, rows=4, cols=4)
        g2 = self.grid_generator.create(num_nodes=16, rows=4, cols=4)

        apply_edge_weights(g1, seed=42, min_weight=1.0, max_weight=10.0)
        apply_edge_weights(g2, seed=999, min_weight=1.0, max_weight=10.0)

        weights1 = [w for _, _, w in g1.edges()]
        weights2 = [w for _, _, w in g2.edges()]
        self.assertNotEqual(weights1, weights2)

    def test_invalid_parameters_raise_errors(self):
        """8. Invalid min/max bounds raise clear ValueError exceptions."""
        graph = self.grid_generator.create(num_nodes=4, rows=2, cols=2)

        # min_weight <= 0
        with self.assertRaises(ValueError):
            apply_edge_weights(graph, min_weight=0.0)
        with self.assertRaises(ValueError):
            apply_edge_weights(graph, min_weight=-2.0)

        # max_weight < min_weight
        with self.assertRaises(ValueError):
            apply_edge_weights(graph, min_weight=5.0, max_weight=3.0)

        # NaN or Infinite values
        with self.assertRaises(ValueError):
            apply_edge_weights(graph, min_weight=float("nan"))
        with self.assertRaises(ValueError):
            apply_edge_weights(graph, max_weight=float("inf"))

        # Direct UniformWeightModel instantiation validation
        with self.assertRaises(ValueError):
            UniformWeightModel(min_weight=0.0)
        with self.assertRaises(ValueError):
            UniformWeightModel(min_weight=5.0, max_weight=2.0)

    def test_topology_remains_unchanged_after_weighting(self):
        """9. Applying weights modifies only edge costs; topology and coordinates remain invariant."""
        graph = self.grid_generator.create(num_nodes=12, rows=3, cols=4)

        original_nodes = list(graph.nodes())
        original_neighbor_sets = {u: set(graph.neighbors(u)) for u in graph.nodes()}
        original_edge_count = len(graph.edges())
        original_positions = {u: graph.get_position(u) for u in graph.nodes()}

        apply_edge_weights(graph, seed=42, min_weight=2.0, max_weight=8.0)

        # Nodes invariant
        self.assertEqual(graph.nodes(), original_nodes)
        # Edges count invariant
        self.assertEqual(len(graph.edges()), original_edge_count)
        # Neighbor sets invariant
        for u in graph.nodes():
            self.assertEqual(set(graph.neighbors(u)), original_neighbor_sets[u])
        # Positions invariant
        for u in graph.nodes():
            self.assertEqual(graph.get_position(u), original_positions[u])


if __name__ == "__main__":
    unittest.main()
