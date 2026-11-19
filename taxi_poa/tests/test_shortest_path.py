"""Unit tests for all-pairs shortest-path computation."""

import unittest

from src.network.core import CityGraph
from src.network.generators.grid import GridGenerator
from src.network.generators.hub_spoke import HubAndSpokeGenerator
from src.network.generators.scale_free import ScaleFreeGenerator
from src.network.shortest_path import (
    compute_all_pairs_shortest_paths,
    floyd_warshall_reference,
)
from src.network.validation import DisconnectedGraphError, InvalidWeightError
from src.network.weights import apply_edge_weights


class TestShortestPath(unittest.TestCase):
    def setUp(self):
        self.grid_gen = GridGenerator()
        self.hub_gen = HubAndSpokeGenerator()
        self.sf_gen = ScaleFreeGenerator()

    def test_zero_diagonal(self):
        """1. Zero diagonal: D[u][u] == 0 for all u."""
        graph = self.grid_gen.create(num_nodes=9, rows=3, cols=3)
        apply_edge_weights(graph, seed=42, min_weight=1.0, max_weight=5.0)
        dm = compute_all_pairs_shortest_paths(graph)

        for u in graph.nodes():
            self.assertEqual(dm.get_distance(u, u), 0.0)

    def test_symmetry_undirected_graphs(self):
        """2. Symmetry for undirected graphs: D[u][v] == D[v][u]."""
        graph = self.sf_gen.create(num_nodes=15, m=2, seed=77)
        apply_edge_weights(graph, seed=88, min_weight=2.0, max_weight=8.0)
        dm = compute_all_pairs_shortest_paths(graph)

        for u in graph.nodes():
            for v in graph.nodes():
                self.assertAlmostEqual(dm.get_distance(u, v), dm.get_distance(v, u), places=7)

    def test_positive_non_zero_distances(self):
        """3. Positive non-zero distances: D[u][v] > 0 for all u != v."""
        graph = self.hub_gen.create(num_nodes=12, num_hubs=2, seed=10)
        apply_edge_weights(graph, seed=20, min_weight=1.5, max_weight=6.0)
        dm = compute_all_pairs_shortest_paths(graph)

        for u in graph.nodes():
            for v in graph.nodes():
                if u != v:
                    self.assertGreater(dm.get_distance(u, v), 0.0)

    def test_correct_shortest_path_manual_graph(self):
        """4. Correct shortest path on a small manually constructed graph.

        Graph layout:
          0 --- (weight 10.0) --- 1
           \\                     /
          (2.0)                 (3.0)
             \\                 /
                    2
        Direct edge cost is 10.0, but path 0 -> 2 -> 1 has cost 2.0 + 3.0 = 5.0.
        """
        graph = CityGraph(num_nodes=3)
        graph.add_edge(0, 1, 10.0)
        graph.add_edge(0, 2, 2.0)
        graph.add_edge(2, 1, 3.0)

        dm = compute_all_pairs_shortest_paths(graph)

        self.assertEqual(dm.get_distance(0, 1), 5.0)
        self.assertEqual(dm.get_distance(1, 0), 5.0)
        self.assertEqual(dm.get_distance(0, 2), 2.0)
        self.assertEqual(dm.get_distance(2, 1), 3.0)

        path = dm.get_path(0, 1)
        self.assertEqual(path, [0, 2, 1])

    def test_path_reconstruction_cost_matches(self):
        """5. Reconstructed path must have total edge weight equal to D[u][v]."""
        graph = self.grid_gen.create(num_nodes=12, rows=3, cols=4)
        apply_edge_weights(graph, seed=333, min_weight=1.0, max_weight=5.0)
        dm = compute_all_pairs_shortest_paths(graph)

        for u in graph.nodes():
            for v in graph.nodes():
                path = dm.get_path(u, v)
                if u == v:
                    self.assertEqual(path, [u])
                    continue

                # Path must start with u and end with v
                self.assertEqual(path[0], u)
                self.assertEqual(path[-1], v)

                # Sum weights along consecutive steps in path
                cost = 0.0
                for i in range(len(path) - 1):
                    step_u, step_v = path[i], path[i + 1]
                    self.assertTrue(graph.has_edge(step_u, step_v))
                    cost += graph.get_weight(step_u, step_v)

                self.assertAlmostEqual(cost, dm.get_distance(u, v), places=7)

    def test_disconnected_graph_handling(self):
        """6. Disconnected graph raises DisconnectedGraphError."""
        graph = CityGraph(num_nodes=4)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(2, 3, 1.0)

        with self.assertRaises(DisconnectedGraphError):
            compute_all_pairs_shortest_paths(graph)

    def test_two_node_graph(self):
        """7. 2-node graph behaves correctly."""
        graph = CityGraph(num_nodes=2)
        graph.add_edge(0, 1, 4.25)

        dm = compute_all_pairs_shortest_paths(graph)

        self.assertEqual(dm.num_nodes, 2)
        self.assertEqual(dm.get_distance(0, 0), 0.0)
        self.assertEqual(dm.get_distance(1, 1), 0.0)
        self.assertEqual(dm.get_distance(0, 1), 4.25)
        self.assertEqual(dm.get_distance(1, 0), 4.25)
        self.assertEqual(dm.get_path(0, 1), [0, 1])
        self.assertEqual(dm.get_path(1, 0), [1, 0])

    def test_triangle_inequality_small_graphs(self):
        """8. Triangle inequality D[u][w] <= D[u][v] + D[v][w] holds across all triples."""
        graph = self.grid_gen.create(num_nodes=9, rows=3, cols=3)
        apply_edge_weights(graph, seed=512, min_weight=1.0, max_weight=10.0)
        dm = compute_all_pairs_shortest_paths(graph)

        nodes = graph.nodes()
        for u in nodes:
            for v in nodes:
                d_uv = dm.get_distance(u, v)
                for w in nodes:
                    d_uw = dm.get_distance(u, w)
                    d_vw = dm.get_distance(v, w)
                    # Triangle inequality: d(u, w) <= d(u, v) + d(v, w)
                    self.assertLessEqual(d_uw, d_uv + d_vw + 1e-9)

    def test_dijkstra_matches_floyd_warshall(self):
        """9. Compare Dijkstra APSP against Floyd-Warshall reference on various topologies."""
        # Test 1: Grid topology
        g_grid = self.grid_gen.create(num_nodes=9, rows=3, cols=3)
        apply_edge_weights(g_grid, seed=101, min_weight=1.0, max_weight=5.0)
        dm_grid = compute_all_pairs_shortest_paths(g_grid)
        fw_grid = floyd_warshall_reference(g_grid)
        for i in range(9):
            for j in range(9):
                self.assertAlmostEqual(dm_grid.get_distance(i, j), fw_grid[i][j], places=7)

        # Test 2: Hub-and-spoke topology
        g_hub = self.hub_gen.create(num_nodes=8, num_hubs=2, seed=202)
        apply_edge_weights(g_hub, seed=303, min_weight=1.0, max_weight=5.0)
        dm_hub = compute_all_pairs_shortest_paths(g_hub)
        fw_hub = floyd_warshall_reference(g_hub)
        for i in range(8):
            for j in range(8):
                self.assertAlmostEqual(dm_hub.get_distance(i, j), fw_hub[i][j], places=7)

        # Test 3: Scale-free topology
        g_sf = self.sf_gen.create(num_nodes=8, m=2, seed=404)
        apply_edge_weights(g_sf, seed=505, min_weight=1.0, max_weight=5.0)
        dm_sf = compute_all_pairs_shortest_paths(g_sf)
        fw_sf = floyd_warshall_reference(g_sf)
        for i in range(8):
            for j in range(8):
                self.assertAlmostEqual(dm_sf.get_distance(i, j), fw_sf[i][j], places=7)

    def test_reproducibility_deterministic_results(self):
        """10. Running APSP on identical graphs produces identical distance matrices."""
        g1 = self.grid_gen.create(num_nodes=12, rows=3, cols=4)
        g2 = self.grid_gen.create(num_nodes=12, rows=3, cols=4)

        apply_edge_weights(g1, seed=42, min_weight=1.0, max_weight=5.0)
        apply_edge_weights(g2, seed=42, min_weight=1.0, max_weight=5.0)

        dm1 = compute_all_pairs_shortest_paths(g1)
        dm2 = compute_all_pairs_shortest_paths(g2)

        self.assertEqual(dm1.to_list(), dm2.to_list())
        for u in g1.nodes():
            for v in g1.nodes():
                self.assertEqual(dm1.get_path(u, v), dm2.get_path(u, v))


if __name__ == "__main__":
    unittest.main()
