"""
Travel-time aggregation helpers.

Provides simple routines to compute total travel time from per-segment samples
and compare different solutions.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def total_travel_time(segment_times: Sequence[float]) -> float:
    """Return the total travel time (seconds) of a trajectory."""

    return float(sum(segment_times))


def average_speed(distance_m: float, travel_time_s: float) -> float:
    """
    Compute the average speed for a route.

    Args:
        distance_m: Total covered distance in meters.
        travel_time_s: Total time in seconds.
    """

    if travel_time_s <= 0:
        return 0.0
    return distance_m / travel_time_s


def compare_travel_times(online_time: float, offline_time: float) -> float:
    """Difference between online and offline travel times (positive means slower)."""

    return online_time - offline_time


def cumulative_time_profile(segment_times: Iterable[float]) -> list[float]:
    """
    Build a cumulative travel time profile along the route.

    Useful when plotting how long each partial trajectory took.
    """

    profile = []
    elapsed = 0.0
    for value in segment_times:
        elapsed += value
        profile.append(elapsed)
    return profile


__all__ = ["total_travel_time", "average_speed", "compare_travel_times", "cumulative_time_profile"]
