"""
Helpers to compute the competitive ratio between two motion-planning solutions.

The functions here deliberately stay lightweight: they accept simple numeric
inputs (scalars or sequences) and return floats so they can be reused both in
experiments and ad-hoc notebook analyses.
"""

from __future__ import annotations

from typing import Iterable, Sequence

EPS = 1e-9


def compute_competitive_ratio(online_cost: float, offline_cost: float) -> float:
    """
    Return the competitive ratio ``online_cost / offline_cost``.

    Args:
        online_cost: Objective value achieved by the online algorithm.
        offline_cost: Objective value of the offline optimum.

    The routine guards against division by zero so that experiments never crash
    mid-run. When ``offline_cost`` is extremely small we fall back to 1.0.
    """

    denom = offline_cost if abs(offline_cost) > EPS else EPS
    ratio = online_cost / denom
    return max(ratio, 0.0)


def batch_competitive_ratio(online_costs: Sequence[float], offline_costs: Sequence[float]) -> float:
    """
    Aggregate the element-wise competitive ratios across multiple runs.

    Args:
        online_costs: Sequence with the online objective values.
        offline_costs: Sequence with the offline objective values.

    Returns:
        Average competitive ratio across the provided runs.
    """

    if len(online_costs) != len(offline_costs):
        raise ValueError("online_costs and offline_costs must have identical lengths.")
    if not online_costs:
        return 0.0
    ratios = [compute_competitive_ratio(o, f) for o, f in zip(online_costs, offline_costs)]
    return sum(ratios) / len(ratios)


def max_competitive_ratio(ratios: Iterable[float]) -> float:
    """Convenience helper to highlight the worst-case ratio inside a batch."""

    best = 0.0
    for value in ratios:
        if value > best:
            best = value
    return best


__all__ = ["compute_competitive_ratio", "batch_competitive_ratio", "max_competitive_ratio"]
