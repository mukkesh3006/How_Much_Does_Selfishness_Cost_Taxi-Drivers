"""Unit tests for core graph data representations and base generator."""

import random
import unittest

from src.network.core import CityGraph, CityGraphInstance, DistanceMatrix
from src.network.generators.base import BaseGraphGenerator


class DummyGenerator(BaseGraphGenerator):
    """Dummy generator for testing BaseGraphGenerator template method."""

    def generate(self, num_nodes: int, rng: random.Random, **kwargs) -> CityGraph:
        g = CityGraph(num_nodes=num_nodes)
        # Add simple line graph edges: 0-1, 1-2, ...
        for i in range(num_nodes - 1):
            g.add_edge(i, i + 1, weight=rng.uniform(1.0, 5.0))
        return g


class TestCityGraph(unittest.TestCase):
    def test_init_valid(self):
        g = CityGraph(num_nodes=5)
        self.assertEqual(g.num_nodes, 5)
        self.assertEqual(g.nodes(), [0, 1, 2, 3, 4])
        for i in range(5):
            self.assertEqual(g.degree(i), 0)
            self.assertEqual(g.neighbors(i), [])

    def test_init_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            CityGraph(num_nodes=0)
        with self.assertRaises(ValueError):
            CityGraph(num_nodes=-3)

    def test_add_edge_and_symmetry(self):
        g = CityGraph(num_nodes=4)
        g.add_edge(0, 1, 2.5)
        self.assertTrue(g.has_edge(0, 1))
        self.assertTrue(g.has_edge(1, 0))
        self.assertEqual(g.get_weight(0, 1), 2.5)
        self.assertEqual(g.get_weight(1, 0), 2.5)
        self.assertEqual(g.degree(0), 1)
        self.assertEqual(g.degree(1), 1)
        self.assertEqual(g.neighbors(0), [1])
        self.assertEqual(g.neighbors(1), [0])

    def test_self_loop_rejected(self):
        g = CityGraph(num_nodes=3)
        with self.assertRaises(ValueError):
            g.add_edge(1, 1, 1.0)

    def test_out_of_bounds_node_rejected(self):
        g = CityGraph(num_nodes=3)
        with self.assertRaises(IndexError):
            g.add_edge(0, 3, 1.0)
        with self.assertRaises(IndexError):
            g.get_weight(0, 5)
        with self.assertRaises(IndexError):
            g.neighbors(10)
        with self.assertRaises(IndexError):
            g.degree(-1)

    def test_edges_unique_undirected(self):
        g = CityGraph(num_nodes=3)
        g.add_edge(0, 1, 1.5)
        g.add_edge(1, 2, 2.5)
        edges = g.edges()
        self.assertEqual(len(edges), 2)
        self.assertEqual(edges, [(0, 1, 1.5), (1, 2, 2.5)])

    def test_positions(self):
        g = CityGraph(num_nodes=2)
        self.assertIsNone(g.get_position(0))
        g.set_position(0, 10.0, 20.0)
        self.assertEqual(g.get_position(0), (10.0, 20.0))

    def test_copy(self):
        g = CityGraph(num_nodes=3)
        g.add_edge(0, 1, 4.0)
        g.set_position(0, 1.0, 1.0)
        g_copy = g.copy()

        self.assertEqual(g_copy.num_nodes, 3)
        self.assertTrue(g_copy.has_edge(0, 1))
        self.assertEqual(g_copy.get_weight(0, 1), 4.0)
        self.assertEqual(g_copy.get_position(0), (1.0, 1.0))

        # Modifying original does not affect copy
        g.add_edge(1, 2, 5.0)
        self.assertFalse(g_copy.has_edge(1, 2))


class TestDistanceMatrix(unittest.TestCase):
    def test_valid_matrix(self):
        mat = [
            [0.0, 2.5, 5.0],
            [2.5, 0.0, 2.5],
            [5.0, 2.5, 0.0],
        ]
        dm = DistanceMatrix(mat)
        self.assertEqual(dm.num_nodes, 3)
        self.assertEqual(dm.get_distance(0, 1), 2.5)
        self.assertEqual(dm.get_distance(0, 2), 5.0)
        self.assertEqual(dm.get_distance(1, 1), 0.0)
        self.assertEqual(dm.to_list(), mat)

    def test_invalid_matrix_dimensions(self):
        with self.assertRaises(ValueError):
            DistanceMatrix([])
        with self.assertRaises(ValueError):
            DistanceMatrix([[0.0, 1.0], [1.0]])

    def test_out_of_bounds_query(self):
        dm = DistanceMatrix([[0.0, 1.0], [1.0, 0.0]])
        with self.assertRaises(IndexError):
            dm.get_distance(0, 3)


class TestCityGraphInstance(unittest.TestCase):
    def test_instance_delegation(self):
        g = CityGraph(num_nodes=3)
        g.add_edge(0, 1, 2.0)
        g.add_edge(1, 2, 3.0)
        g.set_position(0, 0.0, 0.0)

        dm = DistanceMatrix([
            [0.0, 2.0, 5.0],
            [2.0, 0.0, 3.0],
            [5.0, 3.0, 0.0],
        ])

        instance = CityGraphInstance(graph=g, distances=dm, metadata={"tag": "test"})
        self.assertEqual(instance.get_travel_cost(0, 2), 5.0)
        self.assertEqual(instance.get_neighbors(1), [0, 2])
        self.assertEqual(instance.get_coordinates(0), (0.0, 0.0))
        self.assertEqual(instance.metadata["tag"], "test")

    def test_missing_distance_matrix_raises(self):
        g = CityGraph(num_nodes=2)
        instance = CityGraphInstance(graph=g)
        with self.assertRaises(ValueError):
            instance.get_travel_cost(0, 1)


class TestBaseGraphGenerator(unittest.TestCase):
    def test_create_with_seed(self):
        gen = DummyGenerator()
        g1 = gen.create(num_nodes=4, seed=42)
        g2 = gen.create(num_nodes=4, seed=42)
        g3 = gen.create(num_nodes=4, seed=99)

        # Same seed yields identical edge weights
        self.assertEqual(g1.edges(), g2.edges())
        # Different seed yields different edge weights
        self.assertNotEqual(g1.edges(), g3.edges())

        self.assertEqual(g1.metadata["generator"], "DummyGenerator")
        self.assertEqual(g1.metadata["seed"], 42)

    def test_create_invalid_num_nodes(self):
        gen = DummyGenerator()
        with self.assertRaises(ValueError):
            gen.create(num_nodes=1)


if __name__ == "__main__":
    unittest.main()
