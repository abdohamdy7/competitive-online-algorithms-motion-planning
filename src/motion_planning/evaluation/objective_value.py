"""
Utility functions to build scalar objective values from motion-planning metrics.

The helpers allow experiments to tweak weights applied to travel time, comfort,
and risk when comparing different policies.
"""

from __future__ import annotations

from typing import Mapping, MutableMapping, Optional


def compute_objective_value(
    *,
    travel_time: float,
    comfort_cost: float,
    risk_cost: float,
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """
    Combine the core metrics into a single scalar objective value.

    Args:
        travel_time: Total time (seconds) for the solution.
        comfort_cost: Proxy for lateral acceleration, jerk, etc.
        risk_cost: Aggregate risk consumed by the plan.
        weights: Optional dict with scaling factors for each component.
                 Supported keys: ``travel_time``, ``comfort``, ``risk``.
    """

    weight_lookup: MutableMapping[str, float] = {"travel_time": 1.0, "comfort": 1.0, "risk": 1.0}
    if weights:
        weight_lookup.update(weights)

    return (
        travel_time * weight_lookup["travel_time"]
        + comfort_cost * weight_lookup["comfort"]
        + risk_cost * weight_lookup["risk"]
    )


def normalize_objective(value: float, reference: float) -> float:
    """
    Normalize an objective value by a reference scalar (e.g., best-known solution).

    This is handy when plotting metrics from different scenarios on a single scale.
    """

    if reference == 0:
        return value
    return value / reference


__all__ = ["compute_objective_value", "normalize_objective"]
