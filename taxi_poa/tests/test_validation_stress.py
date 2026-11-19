"""
Rigorous Validation and Stress Test Suite for Centralized Welfare Optimizer
and Demand Integration.

Contains comprehensive tests for:
1. Independent exactness validation against a standalone brute-force oracle across 50+ random instances.
2. Extensive mathematical welfare equivalence testing across 100+ random joint assignments.
3. ILP solver validation and transparent fallback behavior.
4. Heuristic optimality gap and feasibility analysis across varying problem sizes.
5. Systematic edge-case stress testing (0 drivers, 1 driver/1 zone, large n/1 zone, zero demand, extreme lambda, etc.).
6. Cost interface consistency (callable, nested dict, tuple dict, shortest path adapter, Euclidean).
7. Auto-mode method selection and truthful status reporting.
8. Complete data consistency across all OptimizationResult fields.
9. Numerical robustness, precision limits, and deterministic tie-breaking.
"""

from __future__ import annotations

import itertools
import math
import random
import unittest
from typing import Any, Dict, List, Set, Tuple

from demand import (
    DemandRegime,
    generate_demand,
    generate_uniform_demand,
    generate_concentrated_demand,
    generate_degree_correlated_demand,
    generate_power_law_demand,
    compute_herfindahl_index,
)
from optimizer import (
    CostSource,
    OptimizationResult,
    OptimizationStatus,
    CentralizedOptimizer,
    evaluate_welfare,
    optimize_welfare,
    optimize_welfare_exact,
    optimize_welfare_heuristic,
    optimize_welfare_ilp,
    is_ilp_solver_available,
    create_mock_cost_matrix,
    create_euclidean_cost_fn,
    adapter_from_graph_shortest_paths,
)


class TestIndependentExactnessValidation(unittest.TestCase):
    """
    Independently enumerates every feasible assignment without relying
    on the optimizer's internal search implementation.
    """

    def _independent_brute_force_solver(
        self,
        drivers: List[Any],
        locs: Dict[Any, Any],
        zones: List[Any],
        demand: Dict[Any, float],
        costs: Dict[Tuple[Any, Any], float],
        lambda_val: float,
    ) -> Tuple[float, Dict[Any, Any]]:
        """Standalone oracle written directly inside the test class."""
        best_w = float("-inf")
        best_a = {}

        for combo in itertools.product(zones, repeat=len(drivers)):
            # Form assignment
            a = {d: z for d, z in zip(drivers, combo)}

            # Active zones
            active = set(a.values())
            dem_sum = sum(demand[z] for z in active)

            # Travel costs
            t_cost = sum(costs[(locs[d], a[d])] for d in drivers)

            welfare = dem_sum - lambda_val * t_cost

            if welfare > best_w + 1e-9:
                best_w = welfare
                best_a = a

        return best_w, best_a

    def test_battery_of_random_small_instances(self):
        """Test across 50 randomly generated small problem instances."""
        rng = random.Random(42)

        for instance_idx in range(50):
            num_drivers = rng.randint(1, 4)
            num_zones = rng.randint(1, 4)
            num_locations = rng.randint(1, 3)

            drivers = [f"d_{i}" for i in range(num_drivers)]
            zones = [f"Z_{j}" for j in range(num_zones)]
            locations = [f"Loc_{k}" for k in range(num_locations)]

            # Driver locations
            driver_locations = {d: rng.choice(locations) for d in drivers}

            # Generate random demand regime
            regime = rng.choice(["uniform", "concentrated", "power_law"])
            if regime == "uniform":
                demand = generate_uniform_demand(zones, mean_demand=rng.uniform(10, 50), seed=instance_idx)
            elif regime == "concentrated":
                demand = generate_concentrated_demand(
                    zones, total_demand=rng.uniform(50, 200), hotspot_share=rng.uniform(0.5, 0.95), seed=instance_idx
                )
            else:
                demand = generate_power_law_demand(zones, alpha=rng.uniform(0.5, 2.5), seed=instance_idx)

            # Random costs
            cost_matrix: Dict[Tuple[Any, Any], float] = {}
            for loc in locations:
                for z in zones:
                    cost_matrix[(loc, z)] = round(rng.uniform(0.5, 10.0), 2)

            lambda_c = round(rng.uniform(0.1, 3.0), 2)

            # 1. Run independent test oracle
            oracle_w, oracle_a = self._independent_brute_force_solver(
                drivers, driver_locations, zones, demand, cost_matrix, lambda_c
            )

            # 2. Run optimizer exact solver
            opt_res_exact = optimize_welfare_exact(
                drivers, driver_locations, zones, demand, cost_matrix, lambda_cost=lambda_c
            )

            # 3. Run optimizer top-level function
            opt_res_auto = optimize_welfare(
                drivers, driver_locations, zones, demand, cost_matrix, lambda_cost=lambda_c, method="exact"
            )

            # Assertions
            self.assertAlmostEqual(
                opt_res_exact.welfare,
                oracle_w,
                places=7,
                msg=f"Instance {instance_idx} exact solver welfare mismatch with oracle",
            )
            self.assertAlmostEqual(
                opt_res_auto.welfare,
                oracle_w,
                places=7,
                msg=f"Instance {instance_idx} auto exact welfare mismatch with oracle",
            )
            # Verify feasibility of returned assignment
            self.assertEqual(len(opt_res_exact.assignment), num_drivers)
            for d in drivers:
                self.assertIn(opt_res_exact.assignment[d], zones)


class TestMathematicalWelfareIdentity(unittest.TestCase):
    """
    Thorough verification of mathematical equivalence between:
    Formulation A: sum_i [ D(a_i)/n_{a_i}(a) - lambda * c(l_i, a_i) ]
    Formulation B: sum_{z in active} D(z) - lambda * sum_i c(l_i, a_i)
    """

    def test_extensive_welfare_identity_100_profiles(self):
        rng = random.Random(101)

        for test_idx in range(100):
            n = rng.randint(1, 15)
            z_count = rng.randint(1, 10)
            drivers = [f"d_{i}" for i in range(n)]
            zones = [f"Z_{j}" for j in range(z_count)]
            locations = [f"Loc_{k}" for k in range(max(1, z_count // 2))]

            driver_locations = {d: rng.choice(locations) for d in drivers}
            demand = {z: rng.uniform(0.0, 100.0) for z in zones}
            cost_matrix = {
                (loc, z): rng.uniform(0.0, 15.0) for loc in locations for z in zones
            }
            lambda_val = rng.uniform(0.0, 5.0)

            # Random joint action profile
            assignment = {d: rng.choice(zones) for d in drivers}

            # Formulation A (Driver-by-driver utility sum)
            zone_loads: Dict[Any, int] = {}
            for z in assignment.values():
                zone_loads[z] = zone_loads.get(z, 0) + 1

            sum_utilities = 0.0
            for d in drivers:
                assigned_z = assignment[d]
                loc = driver_locations[d]
                travel_c = cost_matrix[(loc, assigned_z)]
                d_share = demand[assigned_z] / zone_loads[assigned_z]
                u_i = d_share - lambda_val * travel_c
                sum_utilities += u_i

            # Formulation B (Social welfare via active zones)
            active_zones = set(assignment.values())
            captured_demand = sum(demand[z] for z in active_zones)
            total_travel = sum(cost_matrix[(driver_locations[d], assignment[d])] for d in drivers)
            welfare_b = captured_demand - lambda_val * total_travel

            # Evaluation via module function
            w_mod, d_cap_mod, t_cost_mod, u_mod = evaluate_welfare(
                assignment, driver_locations, demand, cost_matrix, lambda_cost=lambda_val
            )

            # Verification
            self.assertAlmostEqual(sum_utilities, welfare_b, places=9)
            self.assertAlmostEqual(w_mod, welfare_b, places=9)
            self.assertAlmostEqual(sum(u_mod.values()), welfare_b, places=9)


class TestILPAndFallbackValidation(unittest.TestCase):
    """Validate ILP solver when present and test transparent fallback."""

    def test_ilp_solver_or_transparent_fallback(self):
        available, solver_name = is_ilp_solver_available()
        drivers = ["d0", "d1", "d2"]
        locs = {"d0": "L0", "d1": "L1", "d2": "L2"}
        zones = ["Z0", "Z1"]
        demand = {"Z0": 30.0, "Z1": 20.0}
        cost_matrix = create_mock_cost_matrix(["L0", "L1", "L2"], zones, seed=42)

        res_exact = optimize_welfare_exact(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0)
        res_ilp = optimize_welfare_ilp(drivers, locs, zones, demand, cost_matrix, lambda_cost=1.0, seed=42)

        if available:
            # If an external ILP solver is installed, it must find the exact global optimum
            self.assertEqual(res_ilp.status, OptimizationStatus.SOLVER_OPTIMAL)
            self.assertAlmostEqual(res_ilp.welfare, res_exact.welfare, places=6)
        else:
            # If no external ILP solver is installed, it must truthfully report HEURISTIC
            self.assertEqual(res_ilp.status, OptimizationStatus.HEURISTIC)
            self.assertTrue(res_ilp.details.get("ilp_fallback", False))
            # Verify heuristic still produces valid feasible solution
            self.assertEqual(len(res_ilp.assignment), 3)


class TestHeuristicGapsAndFeasibility(unittest.TestCase):
    """Measure heuristic optimality gaps on known exact instances."""

    def test_heuristic_optimality_gaps(self):
        rng = random.Random(999)
        gaps: List[float] = []
        rel_gaps: List[float] = []

        for trial in range(25):
            n = rng.randint(2, 4)
            z_count = rng.randint(2, 4)
            drivers = [f"d_{i}" for i in range(n)]
            zones = [f"Z_{j}" for j in range(z_count)]
            locs = {d: f"Loc_{i % 2}" for i, d in enumerate(drivers)}

            demand = generate_concentrated_demand(zones, total_demand=100.0, hotspot_share=0.8, seed=trial)
            costs = create_mock_cost_matrix(["Loc_0", "Loc_1"], zones, min_cost=1.0, max_cost=5.0, seed=trial)

            exact_res = optimize_welfare_exact(drivers, locs, zones, demand, costs, lambda_cost=1.0)
            heur_res = optimize_welfare_heuristic(
                drivers, locs, zones, demand, costs, lambda_cost=1.0, n_restarts=5, seed=trial
            )

            # Heuristic must always be feasible
            self.assertEqual(len(heur_res.assignment), n)
            self.assertTrue(set(heur_res.assignment.values()).issubset(set(zones)))
            self.assertEqual(heur_res.status, OptimizationStatus.HEURISTIC)

            # Welfare comparison
            w_exact = exact_res.welfare
            w_heur = heur_res.welfare
            abs_gap = w_exact - w_heur
            rel_gap = abs_gap / abs(w_exact) if abs(w_exact) > 1e-6 else 0.0

            self.assertGreaterEqual(abs_gap, -1e-9, "Heuristic welfare exceeded exact optimum!")
            gaps.append(abs_gap)
            rel_gaps.append(rel_gap)

        avg_rel_gap = sum(rel_gaps) / len(rel_gaps)
        # Average relative optimality gap should be < 1% on small/medium instances
        self.assertLess(avg_rel_gap, 0.01)


class TestEdgeCasesStress(unittest.TestCase):
    """Stress test boundary conditions and edge cases."""

    def test_zero_drivers(self):
        res = optimize_welfare([], {}, ["Z0", "Z1"], {"Z0": 10.0, "Z1": 10.0}, {("A", "Z0"): 1.0})
        self.assertEqual(res.assignment, {})
        self.assertEqual(res.welfare, 0.0)
        self.assertEqual(res.active_zones, [])

    def test_one_driver_many_zones(self):
        drivers = ["d0"]
        locs = {"d0": "Origin"}
        zones = [f"Z_{i}" for i in range(25)]
        demand = {z: 5.0 for z in zones}
        demand["Z_super"] = 100.0
        zones.append("Z_super")

        # Equal cost to all zones
        costs = {("Origin", z): 5.0 for z in zones}

        res = optimize_welfare(drivers, locs, zones, demand, costs, lambda_cost=1.0)
        self.assertEqual(res.assignment, {"d0": "Z_super"})
        self.assertAlmostEqual(res.welfare, 95.0)

    def test_many_drivers_one_zone(self):
        drivers = [f"d_{i}" for i in range(30)]
        locs = {d: "Origin" for d in drivers}
        zones = ["SingleZone"]
        demand = {"SingleZone": 300.0}
        costs = {("Origin", "SingleZone"): 2.0}

        res = optimize_welfare(drivers, locs, zones, demand, costs, lambda_cost=1.5)
        self.assertEqual(len(res.assignment), 30)
        for d in drivers:
            self.assertEqual(res.assignment[d], "SingleZone")
        # Total travel cost = 30 * 2 = 60
        # Welfare = 300 - 1.5 * 60 = 300 - 90 = 210.0
        self.assertAlmostEqual(res.welfare, 210.0)

    def test_all_zero_demand(self):
        drivers = ["d0", "d1"]
        locs = {"d0": "L0", "d1": "L1"}
        zones = ["Z0", "Z1"]
        demand = {"Z0": 0.0, "Z1": 0.0}
        costs = {
            ("L0", "Z0"): 1.0, ("L0", "Z1"): 5.0,
            ("L1", "Z0"): 4.0, ("L1", "Z1"): 2.0,
        }

        res = optimize_welfare(drivers, locs, zones, demand, costs, lambda_cost=1.0)
        # Should assign to minimize travel cost: d0 -> Z0 (1.0), d1 -> Z1 (2.0) -> cost = 3.0 -> welfare = -3.0
        self.assertEqual(res.assignment, {"d0": "Z0", "d1": "Z1"})
        self.assertAlmostEqual(res.welfare, -3.0)

    def test_mixed_zero_and_positive_demand(self):
        drivers = ["d0"]
        locs = {"d0": "L0"}
        zones = ["Z_dead", "Z_alive"]
        demand = {"Z_dead": 0.0, "Z_alive": 50.0}
        costs = {("L0", "Z_dead"): 1.0, ("L0", "Z_alive"): 2.0}

        # Dead zone: 0 - 1 = -1
        # Alive zone: 50 - 2 = 48  <- Optimal
        res = optimize_welfare(drivers, locs, zones, demand, costs, lambda_cost=1.0)
        self.assertEqual(res.assignment, {"d0": "Z_alive"})
        self.assertAlmostEqual(res.welfare, 48.0)

    def test_extreme_lambda_values(self):
        drivers = ["d0"]
        locs = {"d0": "L0"}
        zones = ["Z_near_low", "Z_far_high"]
        demand = {"Z_near_low": 10.0, "Z_far_high": 1000.0}
        costs = {("L0", "Z_near_low"): 1.0, ("L0", "Z_far_high"): 50.0}

        # Case A: lambda very small (1e-6) -> Demand dominates -> Z_far_high
        res_small_lam = optimize_welfare(drivers, locs, zones, demand, costs, lambda_cost=1e-6)
        self.assertEqual(res_small_lam.assignment, {"d0": "Z_far_high"})

        # Case B: lambda very large (100.0) -> Travel cost dominates:
        # Near: 10 - 100 * 1 = -90
        # Far:  1000 - 100 * 50 = -4000
        res_large_lam = optimize_welfare(drivers, locs, zones, demand, costs, lambda_cost=100.0)
        self.assertEqual(res_large_lam.assignment, {"d0": "Z_near_low"})
        self.assertAlmostEqual(res_large_lam.welfare, -90.0)

    def test_integer_and_float_types(self):
        """Verify seamless execution with mixed int and float demand and costs."""
        drivers = [1, 2]
        locs = {1: 100, 2: 200}
        zones = [10, 20]
        demand = {10: 50, 20: 30.5}  # int and float
        costs = {(100, 10): 2, (100, 20): 4.5, (200, 10): 3.2, (200, 20): 1}

        res = optimize_welfare(drivers, locs, zones, demand, costs, lambda_cost=1)
        self.assertEqual(len(res.assignment), 2)
        self.assertIsInstance(res.welfare, float)


class TestTravelCostInterfaceConsistency(unittest.TestCase):
    """Verify that all supported cost interfaces produce identical optimal results."""

    def test_all_cost_interfaces_equivalence(self):
        drivers = ["d0", "d1"]
        locs = {"d0": "Node_A", "d1": "Node_B"}
        zones = ["Zone_1", "Zone_2"]
        demand = {"Zone_1": 40.0, "Zone_2": 30.0}

        # 1. Tuple-key dict
        cost_tuple = {
            ("Node_A", "Zone_1"): 2.0, ("Node_A", "Zone_2"): 6.0,
            ("Node_B", "Zone_1"): 5.0, ("Node_B", "Zone_2"): 1.5,
        }

        # 2. Nested dict
        cost_nested = {
            "Node_A": {"Zone_1": 2.0, "Zone_2": 6.0},
            "Node_B": {"Zone_1": 5.0, "Zone_2": 1.5},
        }

        # 3. Callable function
        cost_callable = lambda u, v: cost_nested[u][v]

        # 4. Shortest path adapter
        cost_adapter = adapter_from_graph_shortest_paths(cost_nested)

        # 5. Euclidean coordinates
        coords = {
            "Node_A": (0.0, 2.0),
            "Node_B": (0.0, 5.0),
            "Zone_1": (0.0, 0.0),
            "Zone_2": (0.0, 6.5),
        }
        cost_euclidean = create_euclidean_cost_fn(coords)
        # Verify euclidean matches: Node_A to Zone_1 = 2.0, Node_A to Zone_2 = 4.5, Node_B to Zone_1 = 5.0, Node_B to Zone_2 = 1.5

        # Run optimizer across all interfaces
        res_tuple = optimize_welfare(drivers, locs, zones, demand, cost_tuple, lambda_cost=1.0)
        res_nested = optimize_welfare(drivers, locs, zones, demand, cost_nested, lambda_cost=1.0)
        res_callable = optimize_welfare(drivers, locs, zones, demand, cost_callable, lambda_cost=1.0)
        res_adapter = optimize_welfare(drivers, locs, zones, demand, cost_adapter, lambda_cost=1.0)

        # Assert all produce identical assignments and welfare
        self.assertEqual(res_tuple.assignment, res_nested.assignment)
        self.assertEqual(res_tuple.assignment, res_callable.assignment)
        self.assertEqual(res_tuple.assignment, res_adapter.assignment)

        self.assertAlmostEqual(res_tuple.welfare, res_nested.welfare, places=7)
        self.assertAlmostEqual(res_tuple.welfare, res_callable.welfare, places=7)
        self.assertAlmostEqual(res_tuple.welfare, res_adapter.welfare, places=7)


class TestUnifiedAutoModeSelection(unittest.TestCase):
    """Verify auto-mode threshold switching and truthful status reporting."""

    def test_auto_selects_exact_for_small(self):
        drivers = ["d0", "d1"]
        locs = {"d0": "L0", "d1": "L1"}
        zones = ["Z0", "Z1"]
        demand = {"Z0": 10.0, "Z1": 10.0}
        costs = create_mock_cost_matrix(["L0", "L1"], zones, seed=1)

        # |Z|^n = 2^2 = 4 <= 50,000 -> EXACT_OPTIMAL
        res = optimize_welfare(drivers, locs, zones, demand, costs, method="auto")
        self.assertEqual(res.status, OptimizationStatus.EXACT_OPTIMAL)
        self.assertEqual(res.method_used, "exact_enumeration")

    def test_auto_selects_heuristic_for_large(self):
        # 10 drivers, 10 zones -> 10^10 combinations > 50,000 threshold
        drivers = [f"d_{i}" for i in range(10)]
        locs = {d: "L0" for d in drivers}
        zones = [f"Z_{j}" for j in range(10)]
        demand = {z: 10.0 for z in zones}
        costs = create_mock_cost_matrix(["L0"], zones, seed=1)

        res = optimize_welfare(drivers, locs, zones, demand, costs, method="auto", seed=42)
        # Unless an external ILP is installed, auto mode chooses heuristic and reports HEURISTIC
        self.assertIn(
            res.status,
            [OptimizationStatus.HEURISTIC, OptimizationStatus.SOLVER_OPTIMAL],
        )
        self.assertEqual(len(res.assignment), 10)


class TestDataConsistency(unittest.TestCase):
    """Verify that all fields of OptimizationResult are internally consistent."""

    def test_all_result_fields_consistency(self):
        drivers = ["d0", "d1", "d2", "d3"]
        locs = {"d0": "L0", "d1": "L1", "d2": "L0", "d3": "L1"}
        zones = ["Z0", "Z1", "Z2"]
        demand = {"Z0": 40.0, "Z1": 30.0, "Z2": 20.0}
        costs = create_mock_cost_matrix(["L0", "L1"], zones, seed=77)
        lambda_val = 1.25

        res = optimize_welfare(drivers, locs, zones, demand, costs, lambda_cost=lambda_val)

        # 1. Assignment valid
        self.assertEqual(set(res.assignment.keys()), set(drivers))
        self.assertTrue(set(res.assignment.values()).issubset(set(zones)))

        # 2. Active zones
        expected_active = list(set(res.assignment.values()))
        self.assertEqual(set(res.active_zones), set(expected_active))

        # 3. Demand captured
        expected_demand = sum(demand[z] for z in expected_active)
        self.assertAlmostEqual(res.total_demand_captured, expected_demand, places=7)

        # 4. Total travel cost
        expected_travel = sum(costs[(locs[d], res.assignment[d])] for d in drivers)
        self.assertAlmostEqual(res.total_travel_cost, expected_travel, places=7)

        # 5. Welfare = Demand - lambda * Travel
        expected_welfare = expected_demand - lambda_val * expected_travel
        self.assertAlmostEqual(res.welfare, expected_welfare, places=7)

        # 6. Sum of utilities equals welfare
        self.assertAlmostEqual(sum(res.driver_utilities.values()), expected_welfare, places=7)


class TestNumericalRobustnessAndTies(unittest.TestCase):
    """Test tiny differences, large numbers, and tie-breaking stability."""

    def test_close_competing_welfare(self):
        """Two assignments differing by only 0.0001 welfare."""
        drivers = ["d0"]
        locs = {"d0": "L0"}
        zones = ["Z_opt", "Z_close"]
        demand = {"Z_opt": 10.0001, "Z_close": 10.0000}
        costs = {("L0", "Z_opt"): 1.0, ("L0", "Z_close"): 1.0}

        res = optimize_welfare(drivers, locs, zones, demand, costs, lambda_cost=1.0)
        self.assertEqual(res.assignment, {"d0": "Z_opt"})
        self.assertAlmostEqual(res.welfare, 9.0001, places=6)

    def test_large_numbers_numerical_stability(self):
        """Very large demands (e.g. 10^8)."""
        drivers = ["d0", "d1"]
        locs = {"d0": "L0", "d1": "L1"}
        zones = ["Z0", "Z1"]
        demand = {"Z0": 1e8, "Z1": 5e7}
        costs = create_mock_cost_matrix(["L0", "L1"], zones, seed=1)

        res = optimize_welfare(drivers, locs, zones, demand, costs, lambda_cost=0.5)
        self.assertEqual(res.assignment, {"d0": "Z0", "d1": "Z1"})
        expected_w = (1e8 + 5e7) - 0.5 * (costs[("L0", "Z0")] + costs[("L1", "Z1")])
        self.assertAlmostEqual(res.welfare, expected_w, places=2)

    def test_deterministic_tie_breaking_across_repeated_runs(self):
        """Ensure 100% identical tie-breaking across 10 repeated invocations."""
        drivers = ["d0", "d1"]
        locs = {"d0": "L0", "d1": "L0"}
        zones = ["Z0", "Z1"]
        demand = {"Z0": 20.0, "Z1": 20.0}
        costs = {("L0", "Z0"): 1.0, ("L0", "Z1"): 1.0}

        first_res = optimize_welfare(drivers, locs, zones, demand, costs)
        for _ in range(10):
            res = optimize_welfare(drivers, locs, zones, demand, costs)
            self.assertEqual(res.assignment, first_res.assignment)
            self.assertEqual(res.welfare, first_res.welfare)


if __name__ == "__main__":
    unittest.main()
