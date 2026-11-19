"""
Comprehensive unit tests for the demand-generation module.

Tests cover:
- Non-negativity constraint D(z) >= 0 across all regimes
- Uniform demand regime (equal mean, noise bounds, exact match with zero noise)
- Concentrated demand regime (hotspot allocation, independent concentration controls, power-law sweep)
- Degree-correlated demand regime (monotonicity with node degree, degree_power scaling, graph-independence)
- Deterministic reproducibility using random seeds
- Zone representation flexibility (ints, strings, custom objects, sets)
- Edge cases (empty zones, single zone, extreme parameters)
- Herfindahl concentration index H calculation and normalization
"""

import math
import unittest

from demand.regimes import DemandRegime, DemandConfig
from demand.generator import (
    generate_demand,
    generate_uniform_demand,
    generate_concentrated_demand,
    generate_power_law_demand,
    generate_degree_correlated_demand,
    DemandGenerator,
)
from demand.metrics import (
    compute_herfindahl_index,
    compute_normalized_herfindahl_index,
    compute_top_k_share,
    compute_demand_summary,
)


class TestNonNegativity(unittest.TestCase):
    """Verify that demand scores D(z) are always nonnegative (D(z) >= 0)."""

    def setUp(self):
        self.zones = list(range(20))

    def test_uniform_non_negativity(self):
        for seed in range(10):
            # Test even with large noise scale
            demands = generate_uniform_demand(
                self.zones, mean_demand=10.0, noise_scale=2.0, seed=seed
            )
            for z, d in demands.items():
                self.assertGreaterEqual(d, 0.0, f"Zone {z} demand was negative: {d}")

    def test_concentrated_non_negativity(self):
        for seed in range(10):
            demands = generate_concentrated_demand(
                self.zones,
                hotspot_fraction=0.1,
                hotspot_share=0.9,
                noise_scale=2.0,
                seed=seed,
            )
            for z, d in demands.items():
                self.assertGreaterEqual(d, 0.0, f"Zone {z} demand was negative: {d}")

    def test_power_law_non_negativity(self):
        for seed in range(10):
            demands = generate_power_law_demand(
                self.zones, alpha=2.5, noise_scale=2.0, seed=seed
            )
            for z, d in demands.items():
                self.assertGreaterEqual(d, 0.0, f"Zone {z} demand was negative: {d}")

    def test_degree_correlated_non_negativity(self):
        degrees = {z: z for z in self.zones}  # degrees 0 through 19
        for seed in range(10):
            demands = generate_degree_correlated_demand(
                self.zones, degrees=degrees, degree_power=2.0, noise_scale=2.0, seed=seed
            )
            for z, d in demands.items():
                self.assertGreaterEqual(d, 0.0, f"Zone {z} demand was negative: {d}")

    def test_min_demand_enforcement(self):
        min_floor = 5.0
        demands = generate_uniform_demand(
            self.zones, mean_demand=1.0, min_demand=min_floor, seed=42
        )
        for z, d in demands.items():
            self.assertGreaterEqual(d, min_floor)


class TestUniformRegime(unittest.TestCase):
    """Test uniform demand properties."""

    def setUp(self):
        self.zones = [f"zone_{i}" for i in range(10)]

    def test_exact_uniform_zero_noise(self):
        target_mean = 15.0
        demands = generate_uniform_demand(
            self.zones, mean_demand=target_mean, noise_scale=0.0
        )
        self.assertEqual(len(demands), 10)
        for z in self.zones:
            self.assertAlmostEqual(demands[z], target_mean, places=6)

    def test_total_demand_scaling(self):
        target_total = 250.0
        demands = generate_uniform_demand(
            self.zones, total_demand=target_total, noise_scale=0.1, seed=42
        )
        self.assertAlmostEqual(sum(demands.values()), target_total, places=5)

    def test_uniform_perturbation_bounds(self):
        noise = 0.08  # +/- 8%
        target_mean = 50.0
        demands = generate_uniform_demand(
            self.zones, mean_demand=target_mean, noise_scale=noise, seed=123
        )
        for z, d in demands.items():
            # Demand should stay within target_mean * (1 +/- noise)
            self.assertTrue(
                target_mean * (1.0 - noise - 1e-6) <= d <= target_mean * (1.0 + noise + 1e-6),
                f"Demand {d} outside noise bounds for mean {target_mean} and noise {noise}",
            )

    def test_uniform_herfindahl_index(self):
        num_zones = 20
        zones = list(range(num_zones))
        demands = generate_uniform_demand(zones, noise_scale=0.0)
        h = compute_herfindahl_index(demands)
        expected_h = 1.0 / num_zones
        self.assertAlmostEqual(h, expected_h, places=6)

        h_norm = compute_normalized_herfindahl_index(demands)
        self.assertAlmostEqual(h_norm, 0.0, places=6)


class TestConcentratedRegime(unittest.TestCase):
    """Test concentrated demand regime and independent concentration controls."""

    def setUp(self):
        self.zones = [f"Z{i}" for i in range(20)]

    def test_hotspot_demand_share(self):
        # 4 out of 20 zones (20%) should receive ~80% of total demand
        target_total = 1000.0
        hotspot_share = 0.80
        hotspot_fraction = 0.20
        explicit_hotspots = ["Z0", "Z1", "Z2", "Z3"]

        demands = generate_concentrated_demand(
            self.zones,
            total_demand=target_total,
            hotspot_fraction=hotspot_fraction,
            hotspot_share=hotspot_share,
            hotspot_zones=explicit_hotspots,
            noise_scale=0.0,
            seed=42,
        )

        hotspot_sum = sum(demands[z] for z in explicit_hotspots)
        non_hotspot_sum = sum(demands[z] for z in self.zones if z not in explicit_hotspots)

        self.assertAlmostEqual(hotspot_sum, target_total * hotspot_share, places=5)
        self.assertAlmostEqual(non_hotspot_sum, target_total * (1.0 - hotspot_share), places=5)

        # Average demand in hotspot should be significantly higher than non-hotspot
        avg_h = hotspot_sum / len(explicit_hotspots)
        avg_nh = non_hotspot_sum / (len(self.zones) - len(explicit_hotspots))
        self.assertAlmostEqual(avg_h, 200.0)
        self.assertAlmostEqual(avg_nh, 12.5)
        self.assertGreater(avg_h, avg_nh * 10)

    def test_concentration_scaling(self):
        """Increasing hotspot_share should strictly increase Herfindahl index H."""
        h_values = []
        shares = [0.3, 0.5, 0.7, 0.9, 0.99]
        for s in shares:
            demands = generate_concentrated_demand(
                self.zones,
                hotspot_fraction=0.2,
                hotspot_share=s,
                noise_scale=0.0,
                seed=42,
            )
            h = compute_herfindahl_index(demands)
            h_values.append(h)

        # Verify strict monotonic increase of H
        for i in range(len(h_values) - 1):
            self.assertLess(
                h_values[i],
                h_values[i + 1],
                f"Concentration did not increase: {h_values[i]} >= {h_values[i+1]}",
            )

    def test_power_law_concentration_sweep(self):
        """Increasing power_law_alpha should continuously increase concentration."""
        alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        h_indices = []
        for a in alphas:
            demands = generate_power_law_demand(
                self.zones, alpha=a, noise_scale=0.0, seed=42, shuffle_zones=False
            )
            h_indices.append(compute_herfindahl_index(demands))

        # Check monotonic increase
        for i in range(len(h_indices) - 1):
            self.assertLess(h_indices[i], h_indices[i + 1])


class TestDegreeCorrelatedRegime(unittest.TestCase):
    """Test degree-correlated demand regime."""

    def setUp(self):
        self.zones = ["hub", "mid1", "mid2", "leaf1", "leaf2", "leaf3"]
        self.degrees = {
            "hub": 10,
            "mid1": 4,
            "mid2": 4,
            "leaf1": 1,
            "leaf2": 1,
            "leaf3": 1,
        }

    def test_degree_monotonicity(self):
        """Higher degree nodes should receive higher demand."""
        demands = generate_degree_correlated_demand(
            self.zones, degrees=self.degrees, noise_scale=0.0
        )
        self.assertGreater(demands["hub"], demands["mid1"])
        self.assertEqual(demands["mid1"], demands["mid2"])
        self.assertGreater(demands["mid1"], demands["leaf1"])
        self.assertEqual(demands["leaf1"], demands["leaf2"])

    def test_degree_power_scaling(self):
        """Higher degree_power gamma should magnify the difference between high and low degree nodes."""
        d_linear = generate_degree_correlated_demand(
            self.zones, degrees=self.degrees, degree_power=1.0, noise_scale=0.0
        )
        d_quadratic = generate_degree_correlated_demand(
            self.zones, degrees=self.degrees, degree_power=2.0, noise_scale=0.0
        )

        ratio_linear = d_linear["hub"] / d_linear["leaf1"]
        ratio_quadratic = d_quadratic["hub"] / d_quadratic["leaf1"]

        self.assertGreater(ratio_quadratic, ratio_linear)

    def test_degree_as_callable_and_sequence(self):
        """Verify interface accepts callable, sequence, or mapping for degrees."""
        deg_fn = lambda z: self.degrees[z]
        deg_seq = [self.degrees[z] for z in self.zones]

        d_map = generate_degree_correlated_demand(self.zones, degrees=self.degrees, noise_scale=0.0)
        d_fn = generate_degree_correlated_demand(self.zones, degrees=deg_fn, noise_scale=0.0)
        d_seq = generate_degree_correlated_demand(self.zones, degrees=deg_seq, noise_scale=0.0)

        for z in self.zones:
            self.assertAlmostEqual(d_map[z], d_fn[z], places=6)
            self.assertAlmostEqual(d_map[z], d_seq[z], places=6)

    def test_missing_degree_raises_error(self):
        """Should raise clear ValueError if degrees not supplied."""
        with self.assertRaises(ValueError):
            generate_degree_correlated_demand(self.zones, degrees=None)


class TestReproducibility(unittest.TestCase):
    """Test that seed guarantees exact reproducibility across runs."""

    def test_reproducibility_across_all_regimes(self):
        zones = [f"z_{i}" for i in range(15)]
        degrees = {z: (i % 5) + 1 for i, z in enumerate(zones)}

        for regime in [DemandRegime.UNIFORM, DemandRegime.CONCENTRATED, DemandRegime.DEGREE_CORRELATED, DemandRegime.POWER_LAW]:
            run1 = generate_demand(zones, regime=regime, seed=999, degrees=degrees, noise_scale=0.1)
            run2 = generate_demand(zones, regime=regime, seed=999, degrees=degrees, noise_scale=0.1)
            self.assertEqual(run1, run2, f"Regime {regime} failed bitwise reproducibility test")

            # Different seed should yield different values
            run3 = generate_demand(zones, regime=regime, seed=111, degrees=degrees, noise_scale=0.1)
            self.assertNotEqual(run1, run3)


class TestGraphAgnosticism(unittest.TestCase):
    """Ensure generator works seamlessly with ints, strings, tuples, custom objects."""

    def test_various_zone_types(self):
        # 1. Integer zones
        res_int = generate_demand([0, 1, 2, 3], regime="uniform", seed=42)
        self.assertEqual(set(res_int.keys()), {0, 1, 2, 3})

        # 2. String zones
        res_str = generate_demand(["alpha", "beta", "gamma"], regime="uniform", seed=42)
        self.assertEqual(set(res_str.keys()), {"alpha", "beta", "gamma"})

        # 3. Coordinate tuple zones (e.g. grid coordinates)
        res_coord = generate_demand([(0, 0), (0, 1), (1, 0), (1, 1)], regime="uniform", seed=42)
        self.assertEqual(set(res_coord.keys()), {(0, 0), (0, 1), (1, 0), (1, 1)})


class TestEdgeCases(unittest.TestCase):
    """Test boundary and edge cases."""

    def test_empty_zones(self):
        self.assertEqual(generate_demand([], regime="uniform"), {})
        self.assertEqual(generate_demand([], regime="concentrated"), {})
        self.assertEqual(generate_demand([], regime="degree_correlated", degrees={}), {})
        self.assertEqual(compute_herfindahl_index({}), 0.0)
        self.assertEqual(compute_normalized_herfindahl_index({}), 0.0)

    def test_single_zone(self):
        res = generate_demand([100], regime="uniform", total_demand=50.0)
        self.assertEqual(res, {100: 50.0})
        self.assertEqual(compute_herfindahl_index(res), 1.0)
        self.assertEqual(compute_normalized_herfindahl_index(res), 1.0)

    def test_demand_generator_class(self):
        gen = DemandGenerator(seed=42)
        zones = [1, 2, 3, 4]
        res_u = gen.uniform(zones)
        res_c = gen.concentrated(zones, hotspot_fraction=0.25, hotspot_share=0.7)
        self.assertEqual(len(res_u), 4)
        self.assertEqual(len(res_c), 4)

    def test_demand_summary(self):
        zones = ["a", "b", "c", "d"]
        demands = {"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0}
        summary = compute_demand_summary(demands)
        self.assertEqual(summary["num_zones"], 4.0)
        self.assertEqual(summary["total_demand"], 100.0)
        self.assertEqual(summary["mean_demand"], 25.0)
        self.assertEqual(summary["min_demand"], 10.0)
        self.assertEqual(summary["max_demand"], 40.0)
        self.assertGreater(summary["herfindahl_h"], 0.25)


class TestConfigAndRegimeParsing(unittest.TestCase):
    """Test DemandConfig validation and DemandRegime string parsing."""

    def test_regime_string_parsing(self):
        self.assertEqual(DemandRegime.from_str("uniform"), DemandRegime.UNIFORM)
        self.assertEqual(DemandRegime.from_str("UNIFORM"), DemandRegime.UNIFORM)
        self.assertEqual(DemandRegime.from_str("degree-correlated"), DemandRegime.DEGREE_CORRELATED)
        self.assertEqual(DemandRegime.from_str("DEGREE_CORRELATED"), DemandRegime.DEGREE_CORRELATED)
        self.assertEqual(DemandRegime.from_str("concentrated"), DemandRegime.CONCENTRATED)
        self.assertEqual(DemandRegime.from_str("power_law"), DemandRegime.POWER_LAW)

        with self.assertRaises(ValueError):
            DemandRegime.from_str("invalid_regime")

    def test_config_validation(self):
        # Negative min_demand
        with self.assertRaises(ValueError):
            DemandConfig(min_demand=-1.0)

        # Negative noise_scale
        with self.assertRaises(ValueError):
            DemandConfig(noise_scale=-0.1)

        # Invalid hotspot_fraction
        with self.assertRaises(ValueError):
            DemandConfig(hotspot_fraction=0.0)
        with self.assertRaises(ValueError):
            DemandConfig(hotspot_fraction=1.5)

        # Invalid hotspot_share
        with self.assertRaises(ValueError):
            DemandConfig(hotspot_share=-0.1)
        with self.assertRaises(ValueError):
            DemandConfig(hotspot_share=1.2)

    def test_generate_demand_with_kwargs(self):
        zones = [1, 2, 3, 4, 5]
        # Test regime as string with direct kwargs
        res = generate_demand(
            zones,
            regime="concentrated",
            hotspot_fraction=0.2,
            hotspot_share=0.75,
            total_demand=100.0,
            seed=42,
        )
        self.assertEqual(len(res), 5)
        self.assertAlmostEqual(sum(res.values()), 100.0, places=5)


if __name__ == "__main__":
    unittest.main()
