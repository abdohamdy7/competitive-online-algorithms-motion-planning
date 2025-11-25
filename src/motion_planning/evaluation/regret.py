"""
Simple regret calculations for online motion-planning experiments.

Regret is defined as the difference between the online cost and the offline
optimum. Positive regret implies the online policy under-performed the oracle.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def compute_regret(online_cost: float, offline_cost: float) -> float:
    """Single-run regret."""

    return online_cost - offline_cost


def cumulative_regret(online_costs: Sequence[float], offline_costs: Sequence[float]) -> float:
    """Sum of regrets across multiple runs."""

    if len(online_costs) != len(offline_costs):
        raise ValueError("online_costs and offline_costs must have identical lengths.")
    return sum(compute_regret(o, f) for o, f in zip(online_costs, offline_costs))


def regret_curve(regrets: Iterable[float]) -> list[float]:
    """
    Build a cumulative regret curve from an iterable of per-episode regrets.

    This is useful when plotting regret as a function of time/episodes.
    """

    curve = []
    total = 0.0
    for value in regrets:
        total += value
        curve.append(total)
    return curve


__all__ = ["compute_regret", "cumulative_regret", "regret_curve"]
