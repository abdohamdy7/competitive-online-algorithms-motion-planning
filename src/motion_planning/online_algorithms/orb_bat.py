"""
BAT-ORB online selector on candidates-based problems.

This variant uses the minimum risk across offline candidates (risk_min) as a
thresholding primitive. We provide helpers to extract risk_min and a baseline
policy that prioritizes low-risk candidates to preserve budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from motion_planning.online_algorithms.load_problem_to_solve import (
    CandidatesProblem,
    load_candidates_problem,
    write_online_solution_candidates,
)
from motion_planning.online_algorithms.thresholds import bat_threshold
import math


def bat_orb_policy(
    problem: CandidatesProblem,
    *,
    delta_min: float,
    rho_min: float,
    total_budget: Optional[float] = None,
) -> Tuple[List[Optional[int]], List[float], List[float], List[float]]:
    """
    BAT-ORB as specified:
      Psi_bat(t) = (Delta0 / Delta_t) * ln(1 + Delta0 / delta_min)
      Feasible: risk <= Delta_t and rho >= Psi_bat(t)
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
        if remaining <= 0:
            selections.append(None)
            remaining_budget.append(0.0)
            psi_list.append(0.0)
            continue
        # psi_t = (Delta_0 / remaining) * math.log(1.0 + (Delta_0 / delta_min))
        psi_t = (Delta_0 / remaining) * rho_min
        psi_list.append(psi_t)

        best_idx = None
        best_rho = float("-inf")
        best_util = 0.0
        
        minimum_risk_candidate = min (float(cand.get("risk")) for cand in group.candidates)
        print(f"Epoch with remaining budget {remaining:.4f}, psi_t={psi_t:.4f}, min risk candidate={minimum_risk_candidate:.4f}")

        for idx, cand in enumerate(group.candidates):
            
            risk = float(cand.get("risk", 0.0))
            util = float(cand.get("utility", 0.0))
            
            
            rho = util / risk
            print(f"Candidate {idx}: utility={util}, risk={risk}, rho={rho:.4f}")
            print(f"Threshold psi_t={psi_t:.4f}, remaining budget={remaining:.4f}")

            if risk <= 0 or risk > remaining:
                print("  Rejected: risk exceeds remaining budget." )
                continue
            if risk > Delta_0/8 and risk > minimum_risk_candidate:
                print("  Rejected--: will overshoot the budget by ." )
                continue
            if rho < psi_t:
                print("  Rejected--: rho below threshold." )
                continue
                
            if util > best_util:
                best_util = util
                best_idx = idx
            
        if best_idx is not None:
            remaining -= float(group.candidates[best_idx].get("risk", 0.0))
            remaining = max(0.0, remaining)
        else:
            print("No feasible candidate selected for this epoch.")
            raise NotImplementedError("BAT-ORB requires a feasible candidate at each epoch.")
        
        print(f"Selected candidate {best_idx}, updated remaining budget: {remaining:.4f}")
        selections.append(best_idx)
        remaining_budget.append(remaining)

    return selections, remaining_budget, remaining_before, psi_list


def run_bat_orb(
    candidates_csv: Path | str,
    *,
    capacity_override: Optional[float] = None,
    delta_min: Optional[float] = None,
    rho_min: Optional[float] = None,
    all_candidate_files: Optional[List[Path]] = None,
    output_root: Path | str = Path("results/data/online solutions/candidates"),
) -> Path:
    """
    Run BAT-ORB on a candidates CSV and write the online solution CSV.
    """
    problem = load_candidates_problem(candidates_csv, capacity_override=capacity_override)
    # files = all_candidate_files if all_candidate_files else [candidates_csv]
    if delta_min is None:
        # delta_min = bat_threshold(files, capacity_override=capacity_override)
        raise ValueError("delta_min must be provided to run BAT-ORB.")
    
    selections, remaining, remaining_before, psi_list = bat_orb_policy(
        problem,
        delta_min=delta_min,
        rho_min=rho_min,
        total_budget=capacity_override if capacity_override is not None else problem.capacity,
    )
    return write_online_solution_candidates(
        problem,
        selections,
        algorithm="BAT-ORB",
        remaining_budget=remaining,
        remaining_before=remaining_before,
        psi_values=psi_list,
        total_budget=capacity_override if capacity_override is not None else problem.capacity,
        output_root=output_root,
        suffix="online",
    )
