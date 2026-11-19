"""
Comprehensive Unit Tests for the Centralized Welfare Optimizer.

Tests cover:
- Mathematical equivalence between sum(U_i) and Eq. 4 welfare formulation
- 1 driver / 1 zone
- 1 driver / multiple zones
- Multiple drivers / 1 zone
- Multiple drivers / multiple zones
- Equal demand zones
- Strongly concentrated demand
- Zero-demand zones
- Zero travel costs / zero lambda
- High lambda (travel cost dominance)
- Deterministic tie-breaking on tied optimal assignments
- Heterogeneous driver locations
- More drivers than zones (n > |Z|)
- More zones than drivers (|Z| > n)
- Global welfare optimization vs selfish myopic choice
- Independent brute-force validation oracle
- Flexible cost interfaces (callables, tuple-dicts, nested-dicts)
- Multi-start local search heuristic performance
"""

import itertools
import math
import unittest

from demand import (
    generate_demand,
    generate_uniform_demand,
    generate_concentrated_demand,
    generate_degree_correlated_demand,
)
from optimizer import (
    OptimizationResult,
    OptimizationStatus,
    CentralizedOptimizer,
    evaluate_welfare,
    optimize_welfare,
    optimize_welfare_exact,
    optimize_welfare_heuristic,
    create_mock_cost_matrix,
    create_euclidean_cost_fn,
)


class TestWelfareEquivalence(unittest.TestCase):
    """Verify that sum(U_i) and Eq. (4) welfare definition are mathematically identical."""

    def test_mathematical_identity_random_assignments(self):
        zones = ["Z0", "Z1", "Z2", "Z3"]
        demand = {"Z0": 45.0, "Z1": 30.0, "Z2": 15.0, "Z3": 10.0}
        drivers = [f"d{i}" for i in range(5)]
        locs = {"d0": "L0", "d1": "L1", "d2": "L2", "d3": "L0", "d4": "L3"}
        
        cost_matrix = create_mock_cost_matrix(
            locations=["L0", "L1", "L2", "L3"],
            zones=zones,
            seed=42
        )

        # Test across various arbitrary joint action profiles
        test_profiles = [
            {"d0": "Z0", "d1": "Z0", "d2": "Z1", "d3": "Z2", "d4": "Z3"},
            {"d0": "Z0", "d1": "Z0", "d2": "Z0", "d3": "Z0", "d4": "Z0"},
            {"d0": "Z3", "d1": "Z2", "d2": "Z1", "d3": "Z0", "d4": "Z0"},
            {"d0": "Z1", "d1": "Z1", "d2": "Z2", "d3": "Z2", "d4": "Z1"},
        ]

        for lambda_c in [0.0, 0.5, 1.0, 2.5]:
            for prof in test_profiles:
                welfare, dem_cap, travel_cost, utils = evaluate_welfare(
                    assignment=prof,
                    driver_locations=locs,
                    demand=demand,
                    cost_source=cost_matrix,
                    lambda_cost=lambda_c,
                )
                sum_utilities = sum(utils.values())
                self.assertAlmostEqual(
                    welfare,
                    sum_utilities,
                    places=9,
                    msg=f"Welfare mismatch for lambda={lambda_c}, profile={prof}",
                )


class TestSmallInstanceExactOptimizer(unittest.TestCase):
    """Test exact optimizer on foundational cases."""

    def test_one_driver_one_zone(self):
        drivers = ["d0"]
        locs = {"d0": "L0"}
        zones = ["Z0"]
        demand = {"Z0": 50.0}
        cost_matrix = {("L0", "Z0"): 5.0}

        res = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=2.0)
        self.assertEqual(res.assignment, {"d0": "Z0"})
        # Welfare: D(Z0) - 2.0 * 5.0 = 50 - 10 = 40.0
        self.assertAlmostEqual(res.welfare, 40.0)
        self.assertEqual(res.status, OptimizationStatus.EXACT_OPTIMAL)

    def test_one_driver_multiple_zones(self):
        drivers = ["d0"]
        locs = {"d0": "L0"}
        zones = ["Z0", "Z1", "Z2"]
        demand = {"Z0": 10.0, "Z1": 30.0, "Z2": 25.0}
        # Net welfare:
        # Z0: 10 - 1*2 = 8
        # Z1: 30 - 1*10 = 20
        # Z2: 25 - 1*4 = 21  <- Optimal
        cost_matrix = {("L0", "Z0"): 2.0, ("L0", "Z1"): 10.0, ("L0", "Z2"): 4.0}

        res = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0)
        self.assertEqual(res.assignment, {"d0": "Z2"})
        self.assertAlmostEqual(res.welfare, 21.0)

    def test_multiple_drivers_one_zone(self):
        drivers = ["d0", "d1", "d2"]
        locs = {"d0": "L0", "d1": "L1", "d2": "L2"}
        zones = ["Z0"]
        demand = {"Z0": 60.0}
        cost_matrix = {("L0", "Z0"): 2.0, ("L1", "Z0"): 3.0, ("L2", "Z0"): 5.0}

        res = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0)
        self.assertEqual(res.assignment, {"d0": "Z0", "d1": "Z0", "d2": "Z0"})
        # Total travel cost = 2 + 3 + 5 = 10
        # Welfare = 60 - 10 = 50.0
        self.assertAlmostEqual(res.welfare, 50.0)
        # Individual utilities: 60/3 - 2 = 18, 60/3 - 3 = 17, 60/3 - 5 = 15 -> sum = 50
        self.assertAlmostEqual(res.driver_utilities["d0"], 18.0)
        self.assertAlmostEqual(res.driver_utilities["d1"], 17.0)
        self.assertAlmostEqual(res.driver_utilities["d2"], 15.0)

    def test_equal_demand_dispersion(self):
        """When demand is equal, optimizer should disperse drivers to minimize travel costs."""
        drivers = ["d0", "d1"]
        locs = {"d0": "L0", "d1": "L1"}
        zones = ["Z0", "Z1"]
        demand = {"Z0": 20.0, "Z1": 20.0}
        # d0 is close to Z0, d1 is close to Z1
        cost_matrix = {
            ("L0", "Z0"): 1.0, ("L0", "Z1"): 10.0,
            ("L1", "Z0"): 10.0, ("L1", "Z1"): 1.0,
        }

        res = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0)
        self.assertEqual(res.assignment, {"d0": "Z0", "d1": "Z1"})
        # Both zones active: 20 + 20 - (1 + 1) = 38.0
        self.assertAlmostEqual(res.welfare, 38.0)

    def test_strongly_concentrated_hotspot(self):
        """When one hotspot has huge demand, welfare optimizer may group drivers if travel is justified."""
        drivers = ["d0", "d1"]
        locs = {"d0": "L0", "d1": "L1"}
        zones = ["Z_hot", "Z_cold"]
        demand = {"Z_hot": 100.0, "Z_cold": 2.0}
        cost_matrix = {
            ("L0", "Z_hot"): 2.0, ("L0", "Z_cold"): 1.0,
            ("L1", "Z_hot"): 3.0, ("L1", "Z_cold"): 1.0,
        }

        # Case A: Separate zones -> W = 100 + 2 - (2 + 1) = 99.0
        # Case B: Both in Z_hot -> W = 100 - (2 + 3) = 95.0
        # Separate is better because activating Z_cold gives +2 at travel cost +1 (net +1)
        res = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0)
        self.assertEqual(res.assignment, {"d0": "Z_hot", "d1": "Z_cold"})
        self.assertAlmostEqual(res.welfare, 99.0)

    def test_zero_demand_zone(self):
        """Optimizer will not send driver to zero-demand zone if positive-demand zone is available at equal/better cost."""
        drivers = ["d0"]
        locs = {"d0": "L0"}
        zones = ["Z_active", "Z_dead"]
        demand = {"Z_active": 20.0, "Z_dead": 0.0}
        cost_matrix = {("L0", "Z_active"): 2.0, ("L0", "Z_dead"): 2.0}

        res = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0)
        self.assertEqual(res.assignment, {"d0": "Z_active"})
        self.assertAlmostEqual(res.welfare, 18.0)

    def test_zero_travel_cost(self):
        """When lambda = 0, optimizer activates top demand zones."""
        drivers = ["d0", "d1"]
        locs = {"d0": "L0", "d1": "L1"}
        zones = ["Z0", "Z1", "Z2"]
        demand = {"Z0": 10.0, "Z1": 30.0, "Z2": 25.0}
        cost_matrix = create_mock_cost_matrix(["L0", "L1"], zones, seed=1)

        res = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=0.0)
        # 2 drivers should activate the 2 highest demand zones: Z1 (30) and Z2 (25) -> W = 55.0
        self.assertAlmostEqual(res.welfare, 55.0)
        self.assertEqual(set(res.assignment.values()), {"Z1", "Z2"})

    def test_high_lambda_local_dominance(self):
        """When lambda is very high, drivers prioritize nearest zone over demand."""
        drivers = ["d0"]
        locs = {"d0": "L0"}
        zones = ["Z_far_high", "Z_near_low"]
        demand = {"Z_far_high": 100.0, "Z_near_low": 10.0}
        # Z_far cost is 10, Z_near cost is 1
        # For lambda=20:
        # Far: 100 - 20*10 = -100
        # Near: 10 - 20*1 = -10  <- Optimal
        cost_matrix = {("L0", "Z_far_high"): 10.0, ("L0", "Z_near_low"): 1.0}

        res = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=20.0)
        self.assertEqual(res.assignment, {"d0": "Z_near_low"})
        self.assertAlmostEqual(res.welfare, -10.0)

    def test_deterministic_tie_breaking(self):
        """Symmetric problem with equal welfare must break ties deterministically."""
        drivers = ["d0", "d1"]
        locs = {"d0": "L0", "d1": "L0"}
        zones = ["Z0", "Z1"]
        demand = {"Z0": 20.0, "Z1": 20.0}
        # Completely symmetric cost
        cost_matrix = {
            ("L0", "Z0"): 1.0, ("L0", "Z1"): 1.0,
        }

        res1 = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0)
        res2 = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0)

        self.assertEqual(res1.assignment, res2.assignment)
        self.assertEqual(res1.welfare, res2.welfare)

    def test_more_drivers_than_zones(self):
        """n = 5 drivers, |Z| = 2 zones."""
        drivers = [f"d{i}" for i in range(5)]
        locs = {d: f"L{i}" for i, d in enumerate(drivers)}
        zones = ["Z0", "Z1"]
        demand = {"Z0": 50.0, "Z1": 40.0}
        cost_matrix = create_mock_cost_matrix([f"L{i}" for i in range(5)], zones, seed=12)

        res = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0)
        self.assertEqual(len(res.assignment), 5)
        self.assertTrue(set(res.assignment.values()).issubset(set(zones)))


class TestIndependentValidationOracle(unittest.TestCase):
    """Verify optimizer against an independent brute-force oracle implemented directly in the test."""

    def test_against_independent_brute_force(self):
        zones = ["Z0", "Z1", "Z2"]
        demand = {"Z0": 18.0, "Z1": 25.0, "Z2": 12.0}
        drivers = ["d0", "d1", "d2"]
        locs = {"d0": "A", "d1": "B", "d2": "C"}
        cost_matrix = {
            ("A", "Z0"): 1.0, ("A", "Z1"): 4.0, ("A", "Z2"): 3.0,
            ("B", "Z0"): 5.0, ("B", "Z1"): 2.0, ("B", "Z2"): 2.0,
            ("C", "Z0"): 2.0, ("C", "Z1"): 3.0, ("C", "Z2"): 1.0,
        }
        lambda_val = 1.5

        # Independent brute-force in test
        true_best_w = float("-inf")
        true_best_a = None

        for combo in itertools.product(zones, repeat=len(drivers)):
            a = {d: z for d, z in zip(drivers, combo)}
            active = set(a.values())
            dem_sum = sum(demand[z] for z in active)
            t_cost = sum(cost_matrix[(locs[d], a[d])] for d in drivers)
            w = dem_sum - lambda_val * t_cost
            if w > true_best_w + 1e-9:
                true_best_w = w
                true_best_a = a

        # Run optimizer
        result = optimize_welfare(drivers, locs, zones, demand, cost_matrix, lambda_cost=lambda_val)
        self.assertAlmostEqual(result.welfare, true_best_w, places=7)


class TestCostInterfaces(unittest.TestCase):
    """Verify flexibility of cost sources (nested dicts, callables, Euclidean)."""

    def test_nested_dict_cost_source(self):
        drivers = ["d0", "d1"]
        locs = {"d0": "A", "d1": "B"}
        zones = ["Z0", "Z1"]
        demand = {"Z0": 15.0, "Z1": 15.0}
        nested_cost = {
            "A": {"Z0": 1.0, "Z1": 5.0},
            "B": {"Z0": 5.0, "Z1": 1.0},
        }

        res = optimize_welfare(drivers, locs, zones, demand, nested_cost)
        self.assertEqual(res.assignment, {"d0": "Z0", "d1": "Z1"})
        self.assertAlmostEqual(res.welfare, 28.0)

    def test_callable_cost_source(self):
        drivers = ["d0", "d1"]
        locs = {"d0": (0, 0), "d1": (10, 10)}
        zones = ["Z_origin", "Z_corner"]
        demand = {"Z_origin": 20.0, "Z_corner": 20.0}
        
        coords = {
            (0, 0): (0.0, 0.0),
            (10, 10): (10.0, 10.0),
            "Z_origin": (0.0, 0.0),
            "Z_corner": (10.0, 10.0),
        }
        euclidean_fn = create_euclidean_cost_fn(coords)

        res = optimize_welfare(drivers, locs, zones, demand, euclidean_fn, lambda_cost=1.0)
        self.assertEqual(res.assignment, {"d0": "Z_origin", "d1": "Z_corner"})
        self.assertAlmostEqual(res.welfare, 40.0)


class TestHeuristicOptimizer(unittest.TestCase):
    """Test heuristic optimizer on medium instances and verify quality against exact solver."""

    def test_heuristic_matches_exact_on_small_medium(self):
        zones = ["Z0", "Z1", "Z2", "Z3"]
        demand = {"Z0": 50.0, "Z1": 30.0, "Z2": 20.0, "Z3": 10.0}
        drivers = [f"d{i}" for i in range(4)]
        locs = {d: f"L{i}" for i, d in enumerate(drivers)}
        cost_matrix = create_mock_cost_matrix([f"L{i}" for i in range(4)], zones, seed=99)

        exact_res = optimize_welfare_exact(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0)
        heur_res = optimize_welfare_heuristic(
            drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0, n_restarts=5, seed=42
        )

        self.assertEqual(heur_res.status, OptimizationStatus.HEURISTIC)
        self.assertAlmostEqual(heur_res.welfare, exact_res.welfare, places=6)

    def test_heuristic_scales_to_larger_instance(self):
        """Test with 50 drivers and 20 zones."""
        drivers = [f"d_{i}" for i in range(50)]
        locs = {d: f"L_{idx % 10}" for idx, d in enumerate(drivers)}
        zones = [f"Z_{j}" for j in range(20)]
        demand = {z: 10.0 + j * 2.0 for j, z in enumerate(zones)}
        cost_matrix = create_mock_cost_matrix([f"L_{i}" for i in range(10)], zones, seed=123)

        res = optimize_welfare(
            drivers=drivers,
            driver_locations=locs,
            zones=zones,
            demand=demand,
            cost_source=cost_matrix,
            method="heuristic",
            seed=42,
        )
        self.assertEqual(len(res.assignment), 50)
        self.assertEqual(res.status, OptimizationStatus.HEURISTIC)
        self.assertGreater(res.welfare, 0.0)


if __name__ == "__main__":
    unittest.main()
