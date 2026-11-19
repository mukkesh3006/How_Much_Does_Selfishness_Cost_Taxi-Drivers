"""Regression test: ILP must not credit demand for zones with no driver.

Bug: the ILP had x_{i,z} <= y_z but not y_z <= sum_i x_{i,z}, so y_z could be 1
for an unserved zone. The solver then reported an assignment that was far from
optimal (e.g. all drivers in one zone, welfare 18.9 vs exact 48.9).
"""
import random
import unittest

from optimizer import (
    OptimizationStatus,
    evaluate_welfare,
    is_ilp_solver_available,
    optimize_welfare_exact,
    optimize_welfare_ilp,
)


class TestILPMatchesExact(unittest.TestCase):
    def test_ilp_equals_exact_on_random_instances(self):
        available, _ = is_ilp_solver_available()
        if not available:
            self.skipTest("no ILP solver installed")
        for seed in range(60):
            rng = random.Random(seed)
            n_z, n_d = rng.randint(2, 5), rng.randint(1, 5)
            zones = [f"Z{i}" for i in range(n_z)]
            drivers = [f"d{i}" for i in range(n_d)]
            locs = {d: f"L{i}" for i, d in enumerate(drivers)}
            demand = {z: rng.uniform(1, 40) for z in zones}
            cost = {(l, z): rng.uniform(0, 12) for l in locs.values() for z in zones}
            lam = rng.choice([0.1, 0.5, 1.0, 2.0])

            ex = optimize_welfare_exact(drivers, locs, zones, demand, cost, lambda_cost=lam)
            il = optimize_welfare_ilp(drivers, locs, zones, demand, cost, lambda_cost=lam, fallback_to_heuristic=False)

            self.assertEqual(il.status, OptimizationStatus.SOLVER_OPTIMAL)
            # Welfare is recomputed from the returned assignment, so this also
            # catches a solver whose objective disagrees with evaluate_welfare.
            w, *_ = evaluate_welfare(il.assignment, locs, demand, cost, lam)
            self.assertAlmostEqual(w, il.welfare, places=9)
            self.assertAlmostEqual(il.welfare, ex.welfare, places=6, msg=f"seed={seed}")

    def test_known_failing_case(self):
        available, _ = is_ilp_solver_available()
        if not available:
            self.skipTest("no ILP solver installed")
        rng = random.Random(1)
        zones = [f"Z{i}" for i in range(4)]
        drivers = [f"d{i}" for i in range(3)]
        locs = {d: f"L{i}" for i, d in enumerate(drivers)}
        demand = {z: rng.uniform(5, 30) for z in zones}
        cost = {(l, z): rng.uniform(1, 10) for l in locs.values() for z in zones}
        ex = optimize_welfare_exact(drivers, locs, zones, demand, cost, lambda_cost=1.0)
        il = optimize_welfare_ilp(drivers, locs, zones, demand, cost, lambda_cost=1.0, fallback_to_heuristic=False)
        self.assertAlmostEqual(il.welfare, ex.welfare, places=6)
        self.assertEqual(len(set(il.assignment.values())), 3)  # one driver per zone


if __name__ == "__main__":
    unittest.main()
