"""
Demand Generator Module for Spatial Taxi Repositioning Games.

Generates nonnegative demand scores D(z) >= 0 for a collection of spatial zones
under three primary regimes:
1. Uniform: approximately equal demand with small random perturbations.
2. Concentrated: a small subset of zones receives a large share of total demand.
3. Degree-correlated: demand increases monotonically with node connectivity.

Also supports smooth power-law / Pareto concentration sweeps for empirical experiments.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from demand.regimes import DemandConfig, DemandRegime


def _get_rng(seed: Optional[int]) -> random.Random:
    """Return an isolated random.Random instance for deterministic reproducibility."""
    return random.Random(seed)


def _extract_zones(zones: Iterable[Any]) -> List[Any]:
    """Extract a list of unique zone identifiers preserving iteration order."""
    if not isinstance(zones, (list, tuple)):
        # Preserve unique zones in encounter order
        seen = set()
        zone_list = []
        for z in zones:
            if z not in seen:
                seen.add(z)
                zone_list.append(z)
        return zone_list
    return list(zones)


def generate_uniform_demand(
    zones: Iterable[Any],
    total_demand: Optional[float] = None,
    mean_demand: float = 10.0,
    noise_scale: float = 0.05,
    seed: Optional[int] = None,
    min_demand: float = 0.0,
) -> Dict[Any, float]:
    """
    Generate approximately equal demand across all zones with small random perturbations.

    For each zone z:
        D(z) = mean_demand * (1 + epsilon_z)
    where epsilon_z ~ Uniform(-noise_scale, noise_scale), clamped to D(z) >= min_demand.
    If total_demand is specified, values are proportionally scaled so sum(D(z)) == total_demand.

    Args:
        zones: Iterable of zone identifiers (ints, strings, custom objects).
        total_demand: Optional aggregate total demand across all zones.
        mean_demand: Base average demand per zone (default: 10.0).
        noise_scale: Relative magnitude of random uniform perturbation (default: 0.05, i.e., +/-5%).
        seed: Random seed for deterministic reproducibility.
        min_demand: Floor for demand score (default: 0.0, ensuring non-negativity).

    Returns:
        Dict[Any, float]: Mapping of zone_id -> nonnegative demand score D(z).
    """
    zone_list = _extract_zones(zones)
    if not zone_list:
        return {}

    rng = _get_rng(seed)
    num_zones = len(zone_list)
    target_total = total_demand if total_demand is not None else mean_demand * num_zones

    if num_zones == 1:
        return {zone_list[0]: max(min_demand, float(target_total))}

    # Generate raw values with uniform perturbation
    raw_demands: Dict[Any, float] = {}
    for z in zone_list:
        noise = rng.uniform(-noise_scale, noise_scale) if noise_scale > 0 else 0.0
        val = max(min_demand, mean_demand * (1.0 + noise))
        raw_demands[z] = val

    # If total_demand is explicitly specified, scale so sum matches total_demand
    if total_demand is not None:
        raw_sum = sum(raw_demands.values())
        if raw_sum > 0:
            scale = total_demand / raw_sum
            return {z: max(min_demand, float(val * scale)) for z, val in raw_demands.items()}

    return {z: float(val) for z, val in raw_demands.items()}


def generate_concentrated_demand(
    zones: Iterable[Any],
    total_demand: Optional[float] = None,
    mean_demand: float = 10.0,
    hotspot_fraction: float = 0.2,
    hotspot_share: float = 0.8,
    hotspot_zones: Optional[Sequence[Any]] = None,
    noise_scale: float = 0.05,
    seed: Optional[int] = None,
    min_demand: float = 0.0,
) -> Dict[Any, float]:
    """
    Generate demand where a small subset of zones (hotspots) receives a large share of total demand.

    Demand concentration is independently controllable from network topology via:
    - hotspot_fraction (alpha in (0, 1]): fraction of zones designated as high-demand hotspots.
    - hotspot_share (beta in [0, 1]): fraction of total demand allocated to hotspot zones.

    Args:
        zones: Iterable of zone identifiers.
        total_demand: Optional aggregate total demand across all zones.
        mean_demand: Average demand per zone used when total_demand is None (default: 10.0).
        hotspot_fraction: Fraction of zones that are hotspots (default: 0.2, e.g. 20%).
        hotspot_share: Proportion of total demand given to hotspots (default: 0.8, e.g. 80%).
        hotspot_zones: Optional explicit list/set of zone IDs to use as hotspots.
                       If None, hotspots are chosen uniformly at random using seed.
        noise_scale: Relative magnitude of random perturbation within each tier.
        seed: Random seed for deterministic reproducibility.
        min_demand: Minimum allowed demand score per zone (default: 0.0).

    Returns:
        Dict[Any, float]: Mapping of zone_id -> nonnegative demand score D(z).
    """
    zone_list = _extract_zones(zones)
    if not zone_list:
        return {}

    num_zones = len(zone_list)
    target_total = total_demand if total_demand is not None else mean_demand * num_zones

    if num_zones == 1:
        return {zone_list[0]: max(min_demand, float(target_total))}

    rng = _get_rng(seed)

    # Determine hotspot zones
    if hotspot_zones is not None:
        hotspot_set = set(hotspot_zones).intersection(zone_list)
        if not hotspot_set:
            # Fallback: pick randomly if explicit list had no overlap
            k = max(1, round(num_zones * hotspot_fraction))
            hotspot_set = set(rng.sample(zone_list, min(k, num_zones)))
    else:
        k = max(1, min(num_zones - 1, round(num_zones * hotspot_fraction)))
        if k == 0 and num_zones > 1:
            k = 1
        hotspot_set = set(rng.sample(zone_list, k))

    non_hotspot_list = [z for z in zone_list if z not in hotspot_set]
    hotspot_list = [z for z in zone_list if z in hotspot_set]

    num_h = len(hotspot_list)
    num_nh = len(non_hotspot_list)

    # Calculate allocated demand pools
    if num_nh == 0:
        # All zones are hotspots
        pool_h = target_total
        pool_nh = 0.0
    else:
        pool_h = target_total * hotspot_share
        pool_nh = target_total * (1.0 - hotspot_share)

    base_h = pool_h / num_h if num_h > 0 else 0.0
    base_nh = pool_nh / num_nh if num_nh > 0 else 0.0

    raw_demands: Dict[Any, float] = {}

    # Assign hotspots with perturbation
    for z in hotspot_list:
        noise = rng.uniform(-noise_scale, noise_scale) if noise_scale > 0 else 0.0
        val = max(min_demand, base_h * (1.0 + noise))
        raw_demands[z] = val

    # Assign non-hotspots with perturbation
    for z in non_hotspot_list:
        noise = rng.uniform(-noise_scale, noise_scale) if noise_scale > 0 else 0.0
        val = max(min_demand, base_nh * (1.0 + noise))
        raw_demands[z] = val

    # Normalize within tiers to preserve exact pool totals if possible
    h_sum = sum(raw_demands[z] for z in hotspot_list)
    if h_sum > 0 and pool_h > 0:
        scale_h = pool_h / h_sum
        for z in hotspot_list:
            raw_demands[z] = max(min_demand, raw_demands[z] * scale_h)

    nh_sum = sum(raw_demands[z] for z in non_hotspot_list)
    if nh_sum > 0 and pool_nh > 0:
        scale_nh = pool_nh / nh_sum
        for z in non_hotspot_list:
            raw_demands[z] = max(min_demand, raw_demands[z] * scale_nh)

    # Return in original zone order
    return {z: float(raw_demands[z]) for z in zone_list}


def generate_power_law_demand(
    zones: Iterable[Any],
    alpha: float = 1.0,
    total_demand: Optional[float] = None,
    mean_demand: float = 10.0,
    noise_scale: float = 0.0,
    seed: Optional[int] = None,
    min_demand: float = 0.0,
    shuffle_zones: bool = True,
) -> Dict[Any, float]:
    """
    Generate demand using a power-law / Zipf decay ranking w_i = (i + 1)^(-alpha).

    Allows continuous parametric control of demand concentration (Herfindahl index H):
    - alpha = 0.0: Uniform demand.
    - alpha = 0.5: Mild concentration.
    - alpha = 1.0: Moderate (Zipfian) concentration.
    - alpha = 2.0+: Heavy concentration in top 1-2 zones.

    Args:
        zones: Iterable of zone identifiers.
        alpha: Power-law decay exponent alpha >= 0.
        total_demand: Optional aggregate total demand across all zones.
        mean_demand: Average demand per zone used when total_demand is None.
        noise_scale: Optional relative random perturbation applied to weights.
        seed: Random seed for deterministic zone rank shuffling and noise.
        min_demand: Minimum demand floor (default: 0.0).
        shuffle_zones: If True, randomly permute zone ranks using seed.

    Returns:
        Dict[Any, float]: Mapping of zone_id -> nonnegative demand score D(z).
    """
    zone_list = _extract_zones(zones)
    if not zone_list:
        return {}

    num_zones = len(zone_list)
    target_total = total_demand if total_demand is not None else mean_demand * num_zones

    if num_zones == 1:
        return {zone_list[0]: max(min_demand, float(target_total))}

    rng = _get_rng(seed)

    # Order zones
    ranked_zones = list(zone_list)
    if shuffle_zones:
        rng.shuffle(ranked_zones)

    # Compute raw power law weights w_i = (i + 1)^(-alpha)
    raw_weights: Dict[Any, float] = {}
    for rank, z in enumerate(ranked_zones):
        base_w = (rank + 1) ** (-alpha)
        noise = rng.uniform(-noise_scale, noise_scale) if noise_scale > 0 else 0.0
        w = max(0.0, base_w * (1.0 + noise))
        raw_weights[z] = w

    sum_w = sum(raw_weights.values())
    if sum_w > 0:
        scale = target_total / sum_w
        return {
            z: max(min_demand, float(raw_weights[z] * scale))
            for z in zone_list
        }

    # Fallback to equal split if sum_w == 0
    equal_val = target_total / num_zones
    return {z: max(min_demand, float(equal_val)) for z in zone_list}


def generate_degree_correlated_demand(
    zones: Iterable[Any],
    degrees: Optional[Union[Mapping[Any, float], Sequence[float], Callable[[Any], float]]] = None,
    total_demand: Optional[float] = None,
    mean_demand: float = 10.0,
    degree_power: float = 1.0,
    degree_base: float = 1.0,
    noise_scale: float = 0.05,
    seed: Optional[int] = None,
    min_demand: float = 0.0,
) -> Dict[Any, float]:
    """
    Generate demand that increases with node connectivity (degree), creating a relationship
    between network centrality and zone attractiveness.

    For each zone z:
        D(z) proportional to (degree_base + degree(z))^degree_power * (1 + epsilon_z)

    The interface makes zero assumptions about the graph data structure. Node degrees can be
    supplied as:
    - A dictionary mapping zone_id -> degree (e.g., dict(G.degree()))
    - A callable degree_fn(zone_id) -> float
    - A sequence of degree values corresponding to the order of zones
    - Zone objects having a `.degree` attribute or `['degree']` mapping

    Args:
        zones: Iterable of zone identifiers.
        degrees: Degree mapping, callable, or sequence.
        total_demand: Optional aggregate total demand across all zones.
        mean_demand: Target mean demand per zone if total_demand is None.
        degree_power: Exponent gamma for scaling with degree (default: 1.0, linear).
        degree_base: Positive baseline added before degree scaling (default: 1.0).
        noise_scale: Relative magnitude of random perturbation.
        seed: Random seed for deterministic reproducibility.
        min_demand: Minimum allowed demand score (default: 0.0).

    Returns:
        Dict[Any, float]: Mapping of zone_id -> nonnegative demand score D(z).
    """
    zone_list = _extract_zones(zones)
    if not zone_list:
        return {}

    num_zones = len(zone_list)
    target_total = total_demand if total_demand is not None else mean_demand * num_zones

    if num_zones == 1:
        return {zone_list[0]: max(min_demand, float(target_total))}

    rng = _get_rng(seed)

    # Resolve degree for each zone
    zone_degrees: Dict[Any, float] = {}
    if degrees is None:
        # Check if zones themselves have degree attributes or fallback
        for idx, z in enumerate(zone_list):
            deg = None
            if hasattr(z, "degree"):
                deg_attr = getattr(z, "degree")
                deg = deg_attr() if callable(deg_attr) else deg_attr
            elif isinstance(z, Mapping) and "degree" in z:
                deg = z["degree"]

            if deg is not None:
                zone_degrees[z] = float(deg)
            else:
                raise ValueError(
                    f"Degree information must be provided for degree-correlated demand regime. "
                    f"Pass a dict mapping zone_id -> degree, a callable, or a sequence to 'degrees'."
                )
    elif isinstance(degrees, Mapping):
        for z in zone_list:
            if z not in degrees:
                raise KeyError(f"Zone '{z}' not found in provided degrees mapping.")
            zone_degrees[z] = float(degrees[z])
    elif callable(degrees):
        for z in zone_list:
            zone_degrees[z] = float(degrees(z))
    elif isinstance(degrees, Sequence):
        if len(degrees) != len(zone_list):
            raise ValueError(
                f"Length of degrees sequence ({len(degrees)}) must match number of zones ({len(zone_list)})."
            )
        for z, deg in zip(zone_list, degrees):
            zone_degrees[z] = float(deg)
    else:
        raise TypeError(
            f"Unsupported type for degrees: {type(degrees)}. Expected Mapping, Sequence, or Callable."
        )

    # Compute raw demand scores: (degree_base + max(0, deg))^degree_power
    raw_demands: Dict[Any, float] = {}
    for z in zone_list:
        deg = max(0.0, zone_degrees[z])
        base_val = (degree_base + deg) ** degree_power
        noise = rng.uniform(-noise_scale, noise_scale) if noise_scale > 0 else 0.0
        val = max(min_demand, base_val * (1.0 + noise))
        raw_demands[z] = val

    # Scale to match target total demand
    raw_sum = sum(raw_demands.values())
    if raw_sum > 0 and target_total is not None:
        scale = target_total / raw_sum
        return {z: max(min_demand, float(raw_demands[z] * scale)) for z in zone_list}

    return {z: float(raw_demands[z]) for z in zone_list}


def generate_demand(
    zones: Iterable[Any],
    regime: Union[DemandRegime, str] = DemandRegime.UNIFORM,
    config: Optional[DemandConfig] = None,
    **kwargs: Any,
) -> Dict[Any, float]:
    """
    Unified public entry point to generate demand scores D(z) >= 0 across zones.

    Supports all three primary demand regimes (Uniform, Concentrated, Degree-correlated)
    as well as Power-law concentration sweeps. Parameters can be passed via a `DemandConfig`
    object or directly as keyword arguments.

    Args:
        zones: Collection/iterable of zone identifiers (e.g., list of ints, node IDs, etc.).
        regime: Demand regime ('uniform', 'concentrated', 'degree_correlated', 'power_law').
                Defaults to 'uniform' if not specified.
        config: Optional DemandConfig dataclass containing generation parameters.
        **kwargs: Parameter overrides (e.g. seed=42, total_demand=100.0, hotspot_share=0.85).

    Returns:
        Dict[Any, float]: Mapping of zone_id -> nonnegative demand score D(z).

    Examples:
        >>> # 1. Uniform demand
        >>> generate_demand([1, 2, 3, 4], regime="uniform", mean_demand=10.0, seed=42)
        {1: 10.18, 2: 9.94, 3: 10.02, 4: 9.86}

        >>> # 2. Concentrated demand (e.g. 20% zones get 80% demand)
        >>> generate_demand(['A', 'B', 'C', 'D', 'E'], regime="concentrated",
        ...                 hotspot_fraction=0.2, hotspot_share=0.8, seed=42)
        {'A': 2.5, 'B': 40.0, 'C': 2.5, 'D': 2.5, 'E': 2.5}

        >>> # 3. Degree-correlated demand
        >>> degrees = {'A': 2, 'B': 8, 'C': 3}
        >>> generate_demand(['A', 'B', 'C'], regime="degree_correlated", degrees=degrees, seed=42)
        {'A': 6.0, 'B': 18.0, 'C': 8.0}
    """
    # Merge config and kwargs
    if config is None:
        cfg = DemandConfig(regime=regime, **kwargs)
    else:
        # Clone config and apply overrides
        cfg_dict = {
            "regime": config.regime,
            "total_demand": config.total_demand,
            "mean_demand": config.mean_demand,
            "noise_scale": config.noise_scale,
            "seed": config.seed,
            "min_demand": config.min_demand,
            "hotspot_fraction": config.hotspot_fraction,
            "hotspot_share": config.hotspot_share,
            "hotspot_zones": config.hotspot_zones,
            "power_law_alpha": config.power_law_alpha,
            "degree_power": config.degree_power,
            "degree_base": config.degree_base,
            "degrees": config.degrees,
        }
        if regime != DemandRegime.UNIFORM or "regime" in kwargs:
            cfg_dict["regime"] = kwargs.pop("regime", regime)
        cfg_dict.update(kwargs)
        cfg = DemandConfig(**cfg_dict)

    active_regime = DemandRegime.from_str(cfg.regime)

    if active_regime == DemandRegime.UNIFORM:
        return generate_uniform_demand(
            zones=zones,
            total_demand=cfg.total_demand,
            mean_demand=cfg.mean_demand,
            noise_scale=cfg.noise_scale,
            seed=cfg.seed,
            min_demand=cfg.min_demand,
        )
    elif active_regime == DemandRegime.CONCENTRATED:
        return generate_concentrated_demand(
            zones=zones,
            total_demand=cfg.total_demand,
            mean_demand=cfg.mean_demand,
            hotspot_fraction=cfg.hotspot_fraction,
            hotspot_share=cfg.hotspot_share,
            hotspot_zones=cfg.hotspot_zones,
            noise_scale=cfg.noise_scale,
            seed=cfg.seed,
            min_demand=cfg.min_demand,
        )
    elif active_regime == DemandRegime.DEGREE_CORRELATED:
        return generate_degree_correlated_demand(
            zones=zones,
            degrees=cfg.degrees,
            total_demand=cfg.total_demand,
            mean_demand=cfg.mean_demand,
            degree_power=cfg.degree_power,
            degree_base=cfg.degree_base,
            noise_scale=cfg.noise_scale,
            seed=cfg.seed,
            min_demand=cfg.min_demand,
        )
    elif active_regime == DemandRegime.POWER_LAW:
        return generate_power_law_demand(
            zones=zones,
            alpha=cfg.power_law_alpha,
            total_demand=cfg.total_demand,
            mean_demand=cfg.mean_demand,
            noise_scale=cfg.noise_scale,
            seed=cfg.seed,
            min_demand=cfg.min_demand,
        )
    else:
        raise ValueError(f"Unhandled demand regime: {active_regime}")


class DemandGenerator:
    """
    Class-based interface for demand generation with shared or persistent configuration.
    """

    def __init__(self, default_config: Optional[DemandConfig] = None, seed: Optional[int] = None):
        self.config = default_config or DemandConfig()
        if seed is not None:
            self.config.seed = seed

    def generate(
        self,
        zones: Iterable[Any],
        regime: Optional[Union[DemandRegime, str]] = None,
        **kwargs: Any,
    ) -> Dict[Any, float]:
        """Generate demand using stored configuration with optional overrides."""
        target_regime = regime if regime is not None else self.config.regime
        return generate_demand(zones=zones, regime=target_regime, config=self.config, **kwargs)

    def uniform(self, zones: Iterable[Any], **kwargs: Any) -> Dict[Any, float]:
        """Convenience method for uniform demand generation."""
        return self.generate(zones, regime=DemandRegime.UNIFORM, **kwargs)

    def concentrated(self, zones: Iterable[Any], **kwargs: Any) -> Dict[Any, float]:
        """Convenience method for concentrated demand generation."""
        return self.generate(zones, regime=DemandRegime.CONCENTRATED, **kwargs)

    def degree_correlated(
        self,
        zones: Iterable[Any],
        degrees: Optional[Union[Mapping[Any, float], Sequence[float], Callable[[Any], float]]] = None,
        **kwargs: Any,
    ) -> Dict[Any, float]:
        """Convenience method for degree-correlated demand generation."""
        return self.generate(zones, regime=DemandRegime.DEGREE_CORRELATED, degrees=degrees, **kwargs)

    def power_law(self, zones: Iterable[Any], alpha: float = 1.0, **kwargs: Any) -> Dict[Any, float]:
        """Convenience method for power-law concentration demand generation."""
        return self.generate(zones, regime=DemandRegime.POWER_LAW, power_law_alpha=alpha, **kwargs)
