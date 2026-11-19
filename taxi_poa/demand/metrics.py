"""
Demand metrics and concentration statistics.

Implements the normalized Herfindahl index (H) as defined in Section 5.3
of the project report, along with other demand distribution statistics.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Dict


def compute_herfindahl_index(demand: Mapping[Any, float] | Sequence[float]) -> float:
    """
    Compute the Herfindahl concentration index H for a demand distribution:

        H = sum_{z in Z} ( D(z) / sum_{y in Z} D(y) )^2

    Properties:
        - If demand is distributed perfectly equally across |Z| zones, H = 1 / |Z|.
        - If all demand is concentrated in a single zone, H = 1.0.
        - Higher H indicates stronger concentration in fewer zones.

    Args:
        demand: Mapping of zone_id -> D(z) or sequence of demand scores D(z).

    Returns:
        float: Herfindahl index H in [1 / |Z|, 1.0] (or 0.0 if empty / total demand is 0).
    """
    values = list(demand.values()) if isinstance(demand, Mapping) else list(demand)
    if not values:
        return 0.0

    total_demand = sum(values)
    if total_demand <= 0.0:
        return 0.0

    # Share s_z = D(z) / sum D(y)
    shares = [v / total_demand for v in values]
    h_index = sum(s * s for s in shares)
    return float(h_index)


def compute_normalized_herfindahl_index(
    demand: Mapping[Any, float] | Sequence[float]
) -> float:
    """
    Compute the normalized Herfindahl index H* scaled to [0, 1]:

        H* = (H - 1/|Z|) / (1 - 1/|Z|)  for |Z| > 1

    Properties:
        - H* = 0.0 indicates perfectly uniform demand.
        - H* = 1.0 indicates maximum concentration (all demand in one zone).

    Args:
        demand: Mapping of zone_id -> D(z) or sequence of demand scores.

    Returns:
        float: Normalized Herfindahl index H* in [0.0, 1.0].
    """
    values = list(demand.values()) if isinstance(demand, Mapping) else list(demand)
    num_zones = len(values)
    if num_zones <= 1:
        return 1.0 if num_zones == 1 and sum(values) > 0 else 0.0

    h = compute_herfindahl_index(values)
    min_h = 1.0 / num_zones
    max_h = 1.0

    if max_h <= min_h:
        return 0.0

    h_norm = (h - min_h) / (max_h - min_h)
    return float(max(0.0, min(1.0, h_norm)))


def compute_top_k_share(
    demand: Mapping[Any, float] | Sequence[float],
    top_fraction: float = 0.2
) -> float:
    """
    Compute the fraction of total demand captured by the top top_fraction of zones.

    Args:
        demand: Mapping of zone_id -> D(z) or sequence of demand scores.
        top_fraction: Fraction of zones in (0, 1], e.g. 0.2 for top 20%.

    Returns:
        float: Proportion of total demand in the top zones in [0.0, 1.0].
    """
    values = sorted(
        list(demand.values()) if isinstance(demand, Mapping) else list(demand),
        reverse=True
    )
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0

    k = max(1, math.ceil(len(values) * top_fraction))
    top_k_sum = sum(values[:k])
    return float(top_k_sum / total)


def compute_demand_summary(
    demand: Mapping[Any, float] | Sequence[float]
) -> Dict[str, float]:
    """
    Compute summary statistics of a generated demand distribution.

    Args:
        demand: Mapping of zone_id -> D(z) or sequence of demand scores.

    Returns:
        Dict with keys:
            - num_zones: Number of demand zones |Z|
            - total_demand: Sum of all demand scores
            - mean_demand: Average demand per zone
            - min_demand: Minimum zone demand
            - max_demand: Maximum zone demand
            - std_demand: Standard deviation of zone demand
            - herfindahl_h: Standard Herfindahl index H
            - herfindahl_normalized: Scaled Herfindahl index H* in [0, 1]
            - top_20_share: Fraction of total demand held by top 20% of zones
    """
    values = list(demand.values()) if isinstance(demand, Mapping) else list(demand)
    num_zones = len(values)
    if num_zones == 0:
        return {
            "num_zones": 0,
            "total_demand": 0.0,
            "mean_demand": 0.0,
            "min_demand": 0.0,
            "max_demand": 0.0,
            "std_demand": 0.0,
            "herfindahl_h": 0.0,
            "herfindahl_normalized": 0.0,
            "top_20_share": 0.0,
        }

    total = sum(values)
    mean = total / num_zones
    variance = sum((v - mean) ** 2 for v in values) / num_zones
    std = math.sqrt(variance)

    return {
        "num_zones": float(num_zones),
        "total_demand": float(total),
        "mean_demand": float(mean),
        "min_demand": float(min(values)),
        "max_demand": float(max(values)),
        "std_demand": float(std),
        "herfindahl_h": compute_herfindahl_index(values),
        "herfindahl_normalized": compute_normalized_herfindahl_index(values),
        "top_20_share": compute_top_k_share(values, 0.2),
    }
