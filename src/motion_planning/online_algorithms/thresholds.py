"""
Threshold utilities shared by CZL-ORB, BAT-ORB, ITM, and future variants.

These helpers compute threshold parameters directly from offline artifacts
(candidates-based CSVs, etc.) so online algorithms can remain lean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from motion_planning.online_algorithms.load_problem_to_solve import load_candidates_problem


def _iter_candidate_files(files: Sequence[Path | str] | Path | str) -> Iterable[Path]:
    if isinstance(files, (str, Path)):
        yield Path(files)
    else:
        for f in files:
            yield Path(f)


def czl_thresholds(
    candidates_files: Sequence[Path | str] | Path | str,
    *,
    capacity_override: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Return (rho_lower, rho_upper, rho_mid) where rho = utility / risk, computed
    across all candidates with risk > 0 from the provided candidates CSV(s).
    Accepts a single path or a list of paths; compute global min/max.
    """
    rho_min = float("inf")
    rho_max = float("-inf")
    for file_path in _iter_candidate_files(candidates_files):
        problem = load_candidates_problem(file_path, capacity_override=capacity_override)
        for group in problem.groups:
            for cand in group.candidates:
                risk = float(cand.get("risk", 0.0))
                util = float(cand.get("utility", 0.0))
                if risk <= 0:
                    continue
                rho = util / risk
                rho_min = min(rho_min, rho)
                rho_max = max(rho_max, rho)
    if rho_min == float("inf") or rho_max == float("-inf"):
        raise ValueError("No valid rho values found (risk <= 0 for all candidates).")
    rho_mid = 0.5 * (rho_min + rho_max)
    return rho_min, rho_max, rho_mid


def czl_psi(z: float, rho_min: float, rho_max: float) -> float:
    """
    CZL threshold function:
        Psi_czl(t) = ((rho_max * e) / rho_min)^z * (rho_min / e)
    """
    import math

    return (((rho_max * math.e) / rho_min) ** z) * (rho_min / math.e)


def bat_threshold(
    candidates_files: Sequence[Path | str] | Path | str,
    *,
    capacity_override: Optional[float] = None,
) -> float:
    """
    Return risk_min = minimum positive risk across all provided offline candidates.
    """
    rmin = float("inf")
    for file_path in _iter_candidate_files(candidates_files):
        problem = load_candidates_problem(file_path, capacity_override=capacity_override)
        for group in problem.groups:
            for cand in group.candidates:
                risk = float(cand.get("risk", 0.0))
                if 0 < risk < rmin:
                    rmin = risk
    if rmin == float("inf"):
        raise ValueError("No positive risks found in candidates.")
    return rmin


__all__ = ["czl_thresholds", "czl_psi", "bat_threshold"]
