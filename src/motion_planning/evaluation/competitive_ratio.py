"""
Helpers to compute the competitive ratio between two motion-planning solutions.

The functions here deliberately stay lightweight: they accept simple numeric
inputs (scalars or sequences) and return floats so they can be reused both in
experiments and ad-hoc notebook analyses.
"""

from __future__ import annotations

from typing import Iterable, Sequence

EPS = 1e-9


def compute_competitive_ratio(
    online_value: float,
    offline_value: float,
    *,
    objective: str = "cost",
) -> float:
    """
    Compute the competitive ratio with a configurable objective interpretation.

    Args:
        online_value: Objective value achieved by the online algorithm.
        offline_value: Objective value of the offline optimum.
        objective: Either \"cost\" (minimization) or \"utility\" (maximization).

    For cost:    CR = online / offline (lower is better).
    For utility: CR = offline / online (higher is better).

    Guards against division by zero; if the denominator is ~0, uses EPS.
    """
    if objective not in {"cost", "utility"}:
        raise ValueError("objective must be 'cost' or 'utility'.")

    if objective == "cost":
        denom = offline_value if abs(offline_value) > EPS else EPS
        ratio = online_value / denom
    else:  # utility
        denom = online_value if abs(online_value) > EPS else EPS
        ratio = offline_value / denom

    return max(ratio, 0.0)


def batch_competitive_ratio(
    online_vals: Sequence[float],
    offline_vals: Sequence[float],
    *,
    objective: str = "cost",
) -> float:
    """
    Aggregate element-wise competitive ratios across multiple runs.

    Args:
        online_vals: Sequence with the online objective values.
        offline_vals: Sequence with the offline objective values.
        objective: \"cost\" or \"utility\" to match the metric being compared.
    """

    if len(online_vals) != len(offline_vals):
        raise ValueError("online_vals and offline_vals must have identical lengths.")
    if not online_vals:
        return 0.0
    ratios = [compute_competitive_ratio(o, f, objective=objective) for o, f in zip(online_vals, offline_vals)]
    return sum(ratios) / len(ratios)


def max_competitive_ratio(ratios: Iterable[float]) -> float:
    """Convenience helper to highlight the worst-case ratio inside a batch."""

    best = 0.0
    for value in ratios:
        if value > best:
            best = value
    return best


__all__ = ["compute_competitive_ratio", "batch_competitive_ratio", "max_competitive_ratio"]
