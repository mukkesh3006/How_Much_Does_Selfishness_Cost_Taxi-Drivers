"""Unit tests for topology generators (GridGenerator, HubAndSpokeGenerator, ScaleFreeGenerator)."""

import collections
import statistics
import unittest

from src.network.generators.grid import GridGenerator
from src.network.generators.hub_spoke import HubAndSpokeGenerator
from src.network.generators.scale_free import ScaleFreeGenerator
from src.network.validation import validate_graph


class TestGridGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = GridGenerator()

    def test_exact_node_count(self):
        """1. Exact node count matches num_nodes."""
        for n in [4, 9, 12, 20, 25]:
            graph = self.generator.create(num_nodes=n)
            self.assertEqual(graph.num_nodes, n)
            self.assertEqual(len(graph.nodes()), n)
            self.assertEqual(graph.nodes(), list(range(n)))

    def test_exact_edge_count_normal_grid(self):
        """2. Exact edge count for normal grid: 2RC - R - C."""
        test_cases = [
            (2, 3),  # 6 nodes -> 2*2*3 - 2 - 3 = 7 edges
            (3, 4),  # 12 nodes -> 2*3*4 - 3 - 4 = 17 edges
            (5, 5),  # 25 nodes -> 2*5*5 - 5 - 5 = 40 edges
            (4, 6),  # 24 nodes -> 2*4*6 - 4 - 6 = 38 edges
        ]
        for r, c in test_cases:
            graph = self.generator.create(num_nodes=r * c, rows=r, cols=c)
            expected_edges = 2 * r * c - r - c
            self.assertEqual(len(graph.edges()), expected_edges)

    def test_degree_distribution_normal_grid(self):
        """3. Correct degree distribution for a normal grid (corners=2, boundaries=3, interior=4)."""
        r, c = 4, 5
        graph = self.generator.create(num_nodes=r * c, rows=r, cols=c)

        degree_counts = collections.Counter(graph.degree(u) for u in graph.nodes())

        # For R >= 2, C >= 2:
        # 4 corners have degree 2
        expected_corners = 4
        # Perimeter non-corners have degree 3
        expected_boundaries = 2 * (r - 2) + 2 * (c - 2)
        # Interior nodes have degree 4
        expected_interior = (r - 2) * (c - 2)

        self.assertEqual(degree_counts[2], expected_corners)
        self.assertEqual(degree_counts[3], expected_boundaries)
        self.assertEqual(degree_counts[4], expected_interior)
        self.assertEqual(sum(degree_counts.values()), r * c)

    def test_correct_coordinates(self):
        """4. Correct (x, y) coordinates stored: x = col, y = row."""
        r, c = 3, 4
        graph = self.generator.create(num_nodes=r * c, rows=r, cols=c)

        for row in range(r):
            for col in range(c):
                u = row * c + col
                pos = graph.get_position(u)
                self.assertIsNotNone(pos)
                self.assertEqual(pos, (float(col), float(row)))

    def test_connectivity(self):
        """5. Connectivity verified via validate_graph."""
        for n in [2, 6, 12, 16, 23]:
            graph = self.generator.create(num_nodes=n)
            self.assertTrue(validate_graph(graph, check_connected=True, check_weights=True))

    def test_explicit_rows_cols_validation(self):
        """6. Explicit rows and cols validation."""
        # Valid matching rows/cols
        g = self.generator.create(num_nodes=12, rows=3, cols=4)
        self.assertEqual(g.metadata["rows"], 3)
        self.assertEqual(g.metadata["cols"], 4)

        # Infer cols from rows
        g_inferred_cols = self.generator.create(num_nodes=12, rows=3)
        self.assertEqual(g_inferred_cols.metadata["cols"], 4)

        # Infer rows from cols
        g_inferred_rows = self.generator.create(num_nodes=12, cols=6)
        self.assertEqual(g_inferred_rows.metadata["rows"], 2)

        # Mismatch between rows*cols and num_nodes
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=12, rows=3, cols=5)

        # Non-divisible rows
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=12, rows=5)

        # Non-divisible cols
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=12, cols=5)

        # Non-positive rows/cols
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=12, rows=0)
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=12, cols=-2)

    def test_prime_num_nodes_behavior(self):
        """7. Prime num_nodes produces 1 x num_nodes line graph."""
        primes = [7, 11, 13]
        for p in primes:
            graph = self.generator.create(num_nodes=p)
            self.assertEqual(graph.metadata["rows"], 1)
            self.assertEqual(graph.metadata["cols"], p)

            # In a line graph of length p: exactly p - 1 edges
            self.assertEqual(len(graph.edges()), p - 1)

            # Degrees: 2 end nodes have degree 1, p - 2 internal nodes have degree 2
            degree_counts = collections.Counter(graph.degree(u) for u in graph.nodes())
            self.assertEqual(degree_counts[1], 2)
            self.assertEqual(degree_counts[2], p - 2)

            self.assertTrue(validate_graph(graph))

    def test_periodic_grid_degree_and_edge_behavior(self):
        """8. Periodic grid (torus) wrap-around behavior."""
        # 4x4 periodic grid: every node has degree 4, total edges = 2 * R * C = 32
        g_4x4 = self.generator.create(num_nodes=16, rows=4, cols=4, periodic=True)
        self.assertEqual(len(g_4x4.edges()), 32)
        for u in g_4x4.nodes():
            self.assertEqual(g_4x4.degree(u), 4)
        self.assertTrue(validate_graph(g_4x4))

        # 3x5 periodic grid: every node has degree 4, total edges = 2 * 3 * 5 = 30
        g_3x5 = self.generator.create(num_nodes=15, rows=3, cols=5, periodic=True)
        self.assertEqual(len(g_3x5.edges()), 30)
        for u in g_3x5.nodes():
            self.assertEqual(g_3x5.degree(u), 4)
        self.assertTrue(validate_graph(g_3x5))

        # 1x5 periodic line grid forms a cycle (5 edges, all degrees 2)
        g_cycle = self.generator.create(num_nodes=5, rows=1, cols=5, periodic=True)
        self.assertEqual(len(g_cycle.edges()), 5)
        for u in g_cycle.nodes():
            self.assertEqual(g_cycle.degree(u), 2)
        self.assertTrue(validate_graph(g_cycle))

    def test_reproducibility_with_seed(self):
        """9. Reproducibility with the same seed."""
        g1 = self.generator.create(num_nodes=16, seed=42)
        g2 = self.generator.create(num_nodes=16, seed=42)

        self.assertEqual(g1.edges(), g2.edges())
        self.assertEqual(g1.metadata, g2.metadata)
        for u in g1.nodes():
            self.assertEqual(g1.get_position(u), g2.get_position(u))


class TestHubAndSpokeGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = HubAndSpokeGenerator()

    def test_exact_node_count(self):
        """1. Exact node count matches num_nodes across configurations."""
        for n in [3, 8, 15, 30]:
            for h in [1, 2, 3]:
                if h < n:
                    graph = self.generator.create(num_nodes=n, num_hubs=h)
                    self.assertEqual(graph.num_nodes, n)
                    self.assertEqual(len(graph.nodes()), n)
                    self.assertEqual(graph.nodes(), list(range(n)))

    def test_connectivity(self):
        """2. Connectivity verified via validate_graph."""
        for n in [3, 6, 12, 20]:
            for h in [1, 2, 4]:
                if h < n:
                    graph = self.generator.create(num_nodes=n, num_hubs=h)
                    self.assertTrue(validate_graph(graph, check_connected=True, check_weights=True))

    def test_single_hub_star_structure(self):
        """3. Single hub produces a classic star topology."""
        n = 10
        graph = self.generator.create(num_nodes=n, num_hubs=1)

        # Hub 0 is connected to all n-1 spokes
        self.assertEqual(graph.degree(0), n - 1)
        self.assertEqual(graph.neighbors(0), list(range(1, n)))

        # Every spoke has degree 1 and connects directly to hub 0
        for s in range(1, n):
            self.assertEqual(graph.degree(s), 1)
            self.assertEqual(graph.neighbors(s), [0])

        # Exactly n - 1 edges
        self.assertEqual(len(graph.edges()), n - 1)

    def test_multiple_hub_connectivity(self):
        """4. Multiple hubs form a connected backbone."""
        n = 12
        h = 3
        graph = self.generator.create(num_nodes=n, num_hubs=h)

        # Check backbone clique among hubs: every pair of hubs is directly connected
        for h1 in range(h):
            for h2 in range(h1 + 1, h):
                self.assertTrue(graph.has_edge(h1, h2))

        # Overall graph is connected
        self.assertTrue(validate_graph(graph))

    def test_every_spoke_connected_to_at_least_one_hub(self):
        """5. Every peripheral spoke node is connected to at least one hub."""
        n = 16
        h = 4
        graph = self.generator.create(num_nodes=n, num_hubs=h)
        hubs = set(HubAndSpokeGenerator.get_hubs(graph))
        spokes = HubAndSpokeGenerator.get_spokes(graph)

        self.assertEqual(len(spokes), n - h)
        for s in spokes:
            spoke_neighbors = set(graph.neighbors(s))
            # Intersection with hubs must not be empty
            connected_hubs = spoke_neighbors & hubs
            self.assertGreaterEqual(len(connected_hubs), 1)

    def test_hub_identification(self):
        """6. Hubs and spokes are consistently identified."""
        n = 10
        h = 3
        graph = self.generator.create(num_nodes=n, num_hubs=h)

        self.assertEqual(HubAndSpokeGenerator.get_hubs(graph), [0, 1, 2])
        self.assertEqual(HubAndSpokeGenerator.get_spokes(graph), list(range(3, 10)))

        for node in range(3):
            self.assertTrue(HubAndSpokeGenerator.is_hub(graph, node))
        for node in range(3, 10):
            self.assertFalse(HubAndSpokeGenerator.is_hub(graph, node))

    def test_invalid_num_hubs_handling(self):
        """7. Invalid num_hubs raises ValueError."""
        # num_hubs < 1
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=10, num_hubs=0)
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=10, num_hubs=-1)

        # num_hubs >= num_nodes
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=10, num_hubs=10)
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=10, num_hubs=12)

        # num_nodes < 2
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=1, num_hubs=1)

    def test_reproducibility_with_same_seed(self):
        """8. Reproducibility with the same seed."""
        g1 = self.generator.create(num_nodes=15, num_hubs=3, seed=42)
        g2 = self.generator.create(num_nodes=15, num_hubs=3, seed=42)

        self.assertEqual(g1.edges(), g2.edges())
        self.assertEqual(g1.metadata, g2.metadata)
        for u in g1.nodes():
            self.assertEqual(g1.get_position(u), g2.get_position(u))


class TestScaleFreeGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ScaleFreeGenerator()

    def test_exact_node_count(self):
        """1. Exact node count matches num_nodes."""
        for n in [5, 10, 25, 50]:
            graph = self.generator.create(num_nodes=n, m=2)
            self.assertEqual(graph.num_nodes, n)
            self.assertEqual(len(graph.nodes()), n)
            self.assertEqual(graph.nodes(), list(range(n)))

    def test_connectivity(self):
        """2. Connectivity guaranteed via validate_graph across various m parameters."""
        for n in [10, 20, 30]:
            for m in [1, 2, 3]:
                graph = self.generator.create(num_nodes=n, m=m, seed=42)
                self.assertTrue(validate_graph(graph, check_connected=True, check_weights=True))

    def test_expected_edge_count(self):
        """3. Expected edge count: |E| = m0*(m0 - 1)/2 + m*(n - m0)."""
        test_cases = [
            (10, 1, 2),  # 2*1/2 + 1*(10 - 2) = 1 + 8 = 9
            (15, 2, 3),  # 3*2/2 + 2*(15 - 3) = 3 + 24 = 27
            (20, 3, 4),  # 4*3/2 + 3*(20 - 4) = 6 + 48 = 54
            (25, 2, 5),  # 5*4/2 + 2*(25 - 5) = 10 + 40 = 50 (explicit m0)
        ]
        for n, m, m0 in test_cases:
            graph = self.generator.create(num_nodes=n, m=m, m0=m0, seed=123)
            expected_edges = (m0 * (m0 - 1)) // 2 + m * (n - m0)
            self.assertEqual(len(graph.edges()), expected_edges)

    def test_minimum_degree_behavior(self):
        """4. Every node in the network has degree >= m."""
        for m in [1, 2, 3]:
            graph = self.generator.create(num_nodes=35, m=m, seed=77)
            min_deg = min(graph.degree(u) for u in graph.nodes())
            self.assertGreaterEqual(min_deg, m)

    def test_reproducibility_with_same_seed(self):
        """5. Reproducibility with the same seed."""
        g1 = self.generator.create(num_nodes=30, m=2, seed=42)
        g2 = self.generator.create(num_nodes=30, m=2, seed=42)

        self.assertEqual(g1.edges(), g2.edges())
        self.assertEqual(g1.metadata, g2.metadata)
        for u in g1.nodes():
            self.assertEqual(g1.get_position(u), g2.get_position(u))

    def test_invalid_m_handling(self):
        """6. Invalid m or parameter combinations raise ValueError."""
        # m < 1
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=10, m=0)
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=10, m=-1)

        # m >= num_nodes
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=10, m=10)
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=10, m=15)

        # num_nodes < 2
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=1, m=1)

        # m0 < m
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=10, m=3, m0=2)

        # m0 > num_nodes
        with self.assertRaises(ValueError):
            self.generator.create(num_nodes=10, m=2, m0=12)

    def test_deterministic_behavior_with_same_seed(self):
        """7. Identical seed yields identical degree sequence; different seeds diverge."""
        g1 = self.generator.create(num_nodes=40, m=2, seed=123)
        g2 = self.generator.create(num_nodes=40, m=2, seed=123)
        g3 = self.generator.create(num_nodes=40, m=2, seed=999)

        degrees1 = [g1.degree(u) for u in g1.nodes()]
        degrees2 = [g2.degree(u) for u in g2.nodes()]

        self.assertEqual(degrees1, degrees2)
        self.assertNotEqual(g1.edges(), g3.edges())

    def test_heterogeneous_degree_behavior(self):
        """8. Sanity check: preferential attachment generates heterogeneous degrees with high-degree hubs."""
        n = 60
        m = 2
        graph = self.generator.create(num_nodes=n, m=m, seed=42)

        degrees = [graph.degree(u) for u in graph.nodes()]
        min_deg = min(degrees)
        max_deg = max(degrees)
        med_deg = statistics.median(degrees)

        # The degree distribution is heterogeneous (range > 0)
        self.assertGreater(max_deg, min_deg)
        # Hub node exists with degree significantly higher than minimum attachment m
        self.assertGreaterEqual(max_deg, 3 * m)
        # Variance of degrees is strictly positive
        self.assertGreater(statistics.variance(degrees), 0)


if __name__ == "__main__":
    unittest.main()
