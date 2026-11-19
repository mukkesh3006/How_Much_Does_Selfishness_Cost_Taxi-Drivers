"""End-to-end integration tests for high-level generate_city_network API."""

import unittest

from src.network import generate_city_network


class TestIntegration(unittest.TestCase):
    def test_e2e_grid_generation(self):
        """End-to-end test for grid topology generation and queries."""
        city = generate_city_network(
            topology="grid",
            num_nodes=16,
            rows=4,
            cols=4,
            seed=42,
            min_weight=1.0,
            max_weight=5.0,
        )

        # Graph checks
        self.assertEqual(city.graph.num_nodes, 16)
        self.assertEqual(len(city.graph.edges()), 24)

        # API query checks
        cost = city.get_travel_cost(0, 15)
        self.assertGreater(cost, 0.0)

        route = city.get_route(0, 15)
        self.assertEqual(route[0], 0)
        self.assertEqual(route[-1], 15)
        self.assertGreaterEqual(len(route), 7)  # Manhattan distance is 6 steps -> 7 nodes

        neighbors_0 = city.get_neighbors(0)
        self.assertEqual(neighbors_0, [1, 4])

        coords_0 = city.get_coordinates(0)
        self.assertEqual(coords_0, (0.0, 0.0))

    def test_e2e_hub_spoke_generation(self):
        """End-to-end test for hub-and-spoke topology generation and queries."""
        city = generate_city_network(
            topology="hub_spoke",
            num_nodes=12,
            num_hubs=3,
            seed=101,
            min_weight=2.0,
            max_weight=6.0,
        )

        self.assertEqual(city.graph.num_nodes, 12)
        # Hub backbone has 3 edges; 9 spokes have 9 edges -> 12 edges
        self.assertEqual(len(city.graph.edges()), 12)

        cost = city.get_travel_cost(3, 11)
        self.assertGreater(cost, 0.0)

        route = city.get_route(3, 11)
        self.assertEqual(route[0], 3)
        self.assertEqual(route[-1], 11)

        self.assertIsNotNone(city.get_coordinates(0))

    def test_e2e_scale_free_generation(self):
        """End-to-end test for scale-free topology generation and queries."""
        city = generate_city_network(
            topology="scale_free",
            num_nodes=20,
            m=2,
            seed=202,
            min_weight=1.0,
            max_weight=8.0,
        )

        self.assertEqual(city.graph.num_nodes, 20)
        # Seed graph K3 has 3 edges; 17 nodes add 2 edges each -> 3 + 34 = 37 edges
        self.assertEqual(len(city.graph.edges()), 37)

        cost = city.get_travel_cost(0, 19)
        self.assertGreater(cost, 0.0)

        route = city.get_route(0, 19)
        self.assertEqual(route[0], 0)
        self.assertEqual(route[-1], 19)

    def test_unknown_topology_raises_error(self):
        """Invalid topology name raises ValueError."""
        with self.assertRaises(ValueError):
            generate_city_network(topology="non_existent_topology", num_nodes=10)

    def test_reproducibility_across_runs(self):
        """Identical seed produces identical graph, travel costs, and routes."""
        city1 = generate_city_network("grid", num_nodes=16, seed=42)
        city2 = generate_city_network("grid", num_nodes=16, seed=42)

        self.assertEqual(city1.graph.edges(), city2.graph.edges())
        for u in range(16):
            for v in range(16):
                self.assertEqual(city1.get_travel_cost(u, v), city2.get_travel_cost(u, v))
                self.assertEqual(city1.get_route(u, v), city2.get_route(u, v))

    def test_custom_weight_bounds(self):
        """Edge weights respect custom min_weight and max_weight boundaries."""
        city = generate_city_network(
            "grid",
            num_nodes=9,
            rows=3,
            cols=3,
            min_weight=4.0,
            max_weight=7.0,
            seed=55,
        )
        for _, _, w in city.graph.edges():
            self.assertGreaterEqual(w, 4.0)
            self.assertLessEqual(w, 7.0)


if __name__ == "__main__":
    unittest.main()
