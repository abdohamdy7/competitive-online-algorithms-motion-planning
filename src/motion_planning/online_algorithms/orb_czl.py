"""
CZL-ORB online selector on candidates-based problems.

The algorithm relies on thresholds derived from the offline candidates:
    - rho_lower, rho_upper: bounds on utility / risk across all candidates.
It then applies a simple thresholding policy per epoch using only current data.

This implementation provides:
    * helpers to scan offline candidates and extract rho bounds.
    * a baseline online policy that picks the highest-rho candidate that fits
      the remaining budget, with an optional threshold gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from motion_planning.online_algorithms.load_problem_to_solve import (
    CandidatesProblem,
    load_candidates_problem,
    write_online_solution_candidates,
)
from motion_planning.online_algorithms.thresholds import czl_thresholds, czl_psi


def czl_orb_policy(
    problem: CandidatesProblem,
    *,
    rho_min: float,
    rho_max: float,
    total_budget: Optional[float] = None,
) -> Tuple[List[Optional[int]], List[float], List[float], List[float]]:
    """
    CZL-ORB as specified:
      z = 1 - Delta_t / Delta_0
      Psi_czl(t) = ((rho_max * e) / rho_min)^z * (rho_min / e)
      Feasible set: risk <= Delta_t and rho >= Psi_czl(t)
      Pick argmax rho among feasible; if none, return None for that epoch.
    """
    Delta_0 = total_budget if total_budget is not None else problem.capacity
    if Delta_0 is None or Delta_0 <= 0:
        raise ValueError("Total budget (Delta_0) must be positive.")

    remaining = float(Delta_0)
    selections: List[Optional[int]] = []
    remaining_budget: List[float] = []
    remaining_before: List[float] = []
    psi_list: List[float] = []

    for group in problem.groups:
        remaining_before.append(remaining)
        z = 1.0 - (remaining / Delta_0)
        if rho_max > 10000:
            rho_max = 10000  # cap to avoid overflow in exp
            
        psi_t = czl_psi(z, rho_min, rho_max)
        # psi_t = psi_t  # scale down for CZL-ORB
        psi_list.append(psi_t)

        best_idx = None
        best_rho = float("-inf")
        best_util = 0.0
        for idx, cand in enumerate(group.candidates):
            risk = float(cand.get("risk"))
            util = float(cand.get("utility"))
            if risk <= 0:
                continue

            rho = util / risk
            print(f"Candidate {idx}: utility={util}, risk={risk}, rho={rho:.4f}")
            print(f"Threshold psi_t={psi_t:.4f}, remaining budget={remaining:.4f}")
            if risk > remaining:
                print("  Rejected: risk exceeds remaining budget." )
                continue
            if rho < psi_t:
                print("  Rejected--: rho below threshold." )
                continue
            # if rho > best_rho:
            #     print(f'Accepted as current best: {idx}' )
            #     best_rho = rho
            #     best_idx = idx

            if util > best_util:
                best_util = util
                best_idx = idx

        if best_idx is not None:
            remaining -= float(group.candidates[best_idx].get("risk", 0.0))
            remaining = max(0.0, remaining)
        else:
            raise NotImplementedError("CZL-ORB requires a feasible candidate at each epoch.")
        
        print(f"Selected candidate {best_idx} with remaining budget {remaining:.4f}\n"  )
        selections.append(best_idx)
        remaining_budget.append(remaining)

    return selections, remaining_budget, remaining_before, psi_list


def run_czl_orb(
    candidates_csv: Path | str,
    *,
    capacity_override: Optional[float] = None,
    rho_min: Optional[float] = None,
    rho_max: Optional[float] = None,
    all_candidate_files: Optional[List[Path]] = None,
    output_root: Path | str = Path("results/data/online solutions/candidates"),
) -> Path:
    """
    Run CZL-ORB on a candidates CSV and write the online solution CSV.
    If rho_min/rho_max are None, compute from provided files (global scan).
    """
    problem = load_candidates_problem(candidates_csv, capacity_override=capacity_override)
    # files = all_candidate_files if all_candidate_files else [candidates_csv]
    if rho_min is None or rho_max is None:
        raise NotImplementedError("Temporary disable rho_min/rho_max auto-computation.")
        return None
        # rho_min_val, rho_max_val, _ = czl_thresholds(files, capacity_override=capacity_override)
        # rho_min = rho_min if rho_min is not None else rho_min_val
        # rho_max = rho_max if rho_max is not None else rho_max_val

    selections, remaining, remaining_before, psi_list = czl_orb_policy(
        problem,
        rho_min=rho_min,
        rho_max=rho_max,
        total_budget=capacity_override if capacity_override is not None else problem.capacity,
    )
    return write_online_solution_candidates(
        problem,
        selections,
        algorithm="CZL-ORB",
        remaining_budget=remaining,
        remaining_before=remaining_before,
        psi_values=psi_list,
        total_budget=capacity_override if capacity_override is not None else problem.capacity,
        output_root=output_root,
        suffix="online",
    )
