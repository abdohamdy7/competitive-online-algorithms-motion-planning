"""
Risk consumption bookkeeping utilities.

The offline/online planners frequently track an aggregate risk budget. These
helpers simplify turning per-edge risks into interpretable statistics such as
total consumption and residual budget.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def total_risk_consumed(risks: Sequence[float]) -> float:
    """Return the sum of all risk values in the provided sequence."""

    return float(sum(risks))


def risk_utilization(risks: Sequence[float], budget: float) -> float:
    """
    Report the fraction of the budget that was consumed.

    Args:
        risks: Sequence with the per-edge (or per-path) risks.
        budget: Total budget allocated for the planning episode.
    """

    if budget <= 0:
        return 0.0
    return total_risk_consumed(risks) / budget


def residual_budget(risks: Sequence[float], budget: float) -> float:
    """How much risk budget remained after executing the trajectory."""

    return max(budget - total_risk_consumed(risks), 0.0)


def risk_profile(risks: Iterable[float]) -> Tuple[list[float], list[float]]:
    """
    Cumulative risk profile for visualization purposes.

    Returns:
        Two lists: ``indices`` (0-based steps) and ``cumulative_risk``.
    """

    cumulative = []
    running = 0.0
    for idx, value in enumerate(risks):
        running += value
        cumulative.append(running)
    steps = list(range(len(cumulative)))
    return steps, cumulative


__all__ = ["total_risk_consumed", "risk_utilization", "residual_budget", "risk_profile"]
