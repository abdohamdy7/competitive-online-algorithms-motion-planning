"""
ITM-ORB online selector for graph-based problems.

Implements the MILP in Eq. (7) of the formulation. Decision variables:
  - x_{ij}^ν: binary selection of edge (i, j) at speed ν
  - δ_t:      risk allocated to the current epoch

Constraints (7a)–(7e):
  7a: flow conservation from start (v_s) to goal (v_g)
  7b: sum risk * x <= δ_t
  7c: sum utility * x >= Ψ_t * δ_t           (Ψ_t is a user-supplied threshold)
  7d: sum_ν x_{ij}^ν <= 1 for every edge (i,j)
  7e: 0 <= δ_t <= Δ_t, x ∈ {0,1}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from motion_planning.online_algorithms.load_problem_to_solve import GraphProblem, load_graph_problem
from motion_planning.utils.paths import ONLINE_RESULTS_DIR
from motion_planning.online_algorithms.thresholds import czl_thresholds, czl_psi


def _nodes_from_edges(edges_df: pd.DataFrame) -> List[str]:
    nodes = set(edges_df["start_node"].astype(str)).union(edges_df["end_node"].astype(str))
    return sorted(nodes)


def _parse_risk_budget_from_stem(path: Path) -> Optional[float]:
    """
    Heuristic parser: filenames often look like
    {ts}_{scenario}_{budget}_{risk}_{epochs}_edge_values.csv
    or ..._offline_graph_solution_edges.csv.
    We try to detect the token preceding a known risk label (mid/high/low/etc.).
    """
    parts = path.stem.split("_")
    risk_labels = {"low", "mid", "medium", "high", "veryhigh", "very_high", "very"}
    # find risk token
    for idx, token in enumerate(parts):
        if token.lower() in risk_labels and idx > 0:
            try:
                return float(parts[idx - 1])
            except Exception:
                break
    # fallback: pick the last numeric token (skipping timestamps if present)
    # numeric_tokens = []
    # for tok in parts:
    #     try:
    #         numeric_tokens.append(float(tok))
    #     except Exception:
    #         continue
    # if numeric_tokens:
    #     return numeric_tokens[-1]
    # return None


def _write_itm_online_solution(
    edge_values_csv: Path,
    graph_id: str,
    edges_df: pd.DataFrame,
    results: List[Dict[str, Any]],
    total_budget: float,
    output_root: Path | str = ONLINE_RESULTS_DIR / "graph-based",
) -> Path:
    """
    Persist ITM online selections to CSV (graph-based online solution).
    Columns mirror the offline solution edges with per-epoch metadata.
    """
    rows = []
    for res in results:
        epoch = res.get("epoch")
        delta = res.get("delta")
        obj = res.get("objective")
        remaining = res.get("remaining_after")
        for seg_idx, entry in enumerate(res.get("selected_edges", [])):
            u, v, speed, risk_val, util_val = entry
            cost_val = None
            match = edges_df[
                (edges_df["start_node"].astype(str) == str(u))
                & (edges_df["end_node"].astype(str) == str(v))
                & (edges_df["speed"] == speed)
            ]
            if not match.empty and "cost" in match.columns:
                cost_val = float(match.iloc[0].get("cost", None))
            rows.append(
                {
                    "graph_id": graph_id,
                    "epoch": epoch,
                    "segment_index": seg_idx,
                    "start_node": u,
                    "end_node": v,
                    "speed": speed,
                    "risk": risk_val,
                    "cost": cost_val,
                    "utility": util_val,
                    "delta": delta,
                    "objective": obj,
                    "remaining_after": remaining,
                    "total_budget": total_budget,
                    "psi": res.get("psi"),
                }
            )
    if not rows:
        raise ValueError("No ITM selections to write.")
    df = pd.DataFrame(rows)
    # Sort by epoch then start_node, then segment_index
    df = df.sort_values(["epoch", "start_node", "segment_index"]).reset_index(drop=True)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = Path(edge_values_csv).stem.replace("_edge_values", "_online_ITM-ORB")
    out_path = output_root / f"{stem}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def solve_itm_epoch(
    edges_df: pd.DataFrame,
    start_node: str,
    goal_node: str,
    *,
    Delta_t: float,
    Delta_total: Optional[float] = None,
    epoch_size: Optional[float] = None,
    graph_size: Optional[float] = None,
    psi_t: float,
    gurobi_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Solve ITM-ORB for a single epoch between (start_node, goal_node).
    Returns selected edges with speeds, allocated risk δ_t, and objective.
    """

    print(f"Solving ITM-ORB epoch from {start_node} to {goal_node} with Delta_t={Delta_t}, psi_t={psi_t} with epoch_size={epoch_size}, graph_size={graph_size}")
    print(f"  Edges available: {len(edges_df)}")

    if edges_df.empty:
        raise ValueError("edges_df is empty.")

    env = None
    if gurobi_params:
        # Build a dedicated environment with licensing params; avoid setting WLS params after model creation.
        env = gp.Env(params=gurobi_params)
    model = gp.Model("ITM_ORB_epoch", env=env)
    model.Params.OutputFlag = 0
    if gurobi_params:
        for k, v in gurobi_params.items():
            if str(k).upper() in ("WLSACCESSID", "WLSSECRET", "LICENSEID"):
                continue  # already applied at Env creation
            try:
                setattr(model.Params, k, v)
            except Exception:
                model.setParam(k, v)

    # Decision vars: x[(u,v,nu)] and delta
    x: Dict[Tuple[str, str, Any], gp.Var] = {}
    for idx, row in edges_df.iterrows():
        u = str(row["start_node"])
        v = str(row["end_node"])
        nu = row.get("speed")
        x[(u, v, nu)] = model.addVar(vtype=GRB.BINARY, name=f"x[{u},{v},{nu}]")


    # compute the ub for delta for each decision epoch

    # Upper bound for delta: scale by epoch_size / graph_size * Delta_total if provided; otherwise remaining Delta_t.
    ub_delta = float(Delta_t)
    if (
        Delta_total is not None
        and epoch_size is not None
        and graph_size is not None
        and Delta_total > 0
        and epoch_size > 0
        and graph_size > 0
    ):
        scaled = (epoch_size / graph_size) * float(Delta_total)
        ub_delta = min(ub_delta, scaled)

    print(f"  Setting upper bound for delta: {ub_delta:.4f}")
    delta = model.addVar(lb=0.0, ub=ub_delta, name="delta_t")

    # Objective: maximize sum utility * x
    model.setObjective(
        gp.quicksum(float(row["utility"]) * x[(str(row["start_node"]), str(row["end_node"]), row["speed"])]
                    for _, row in edges_df.iterrows()),
        GRB.MAXIMIZE,
    )

    # Flow conservation (7a)
    nodes = _nodes_from_edges(edges_df)
    for k in nodes:
        incoming = []
        outgoing = []
        for (u, v, nu), var in x.items():
            if v == k:
                incoming.append(var)
            if u == k:
                outgoing.append(var)
        rhs = 0
        if k == str(start_node):
            rhs = 1
        elif k == str(goal_node):
            rhs = -1
        model.addConstr(gp.quicksum(outgoing) - gp.quicksum(incoming) == rhs, name=f"flow_{k}")

    # Risk budget (7b)
    model.addConstr(
        gp.quicksum(float(row["risk"]) * x[(str(row["start_node"]), str(row["end_node"]), row["speed"])]
                    for _, row in edges_df.iterrows()) <= delta,
        name="risk_budget",
    )

    # Utility threshold (7c)
    psi_val = psi_t
    if psi_val and psi_val > 0:
        model.addConstr(
            gp.quicksum(float(row["utility"]) * x[(str(row["start_node"]), str(row["end_node"]), row["speed"])]
                        for _, row in edges_df.iterrows()) >= psi_val * delta,
            name="utility_threshold",
        )

    # At most one speed per edge (7d)
    edge_groups: Dict[Tuple[str, str], List[gp.Var]] = {}
    for (u, v, nu), var in x.items():
        edge_groups.setdefault((u, v), []).append(var)
    for (u, v), vars_ in edge_groups.items():
        model.addConstr(gp.quicksum(vars_) <= 1, name=f"one_speed_{u}_{v}")

    # delta bounds handled by var lb/ub (7e)

    model.optimize()

    status = model.Status
    if status != GRB.OPTIMAL:
        return {"status": status, "selected_edges": [], "delta": None, "objective": None}

    selected_edges = []
    total_risk = 0.0
    total_utility = 0.0
    for (u, v, nu), var in x.items():
        if var.X > 0.5:
            row_mask = (edges_df["start_node"].astype(str) == u) & (edges_df["end_node"].astype(str) == v) & (edges_df["speed"] == nu)
            row = edges_df[row_mask].iloc[0]
            risk_val = float(row["risk"])
            util_val = float(row["utility"])
            selected_edges.append((u, v, nu, risk_val, util_val))
            total_risk += risk_val
            total_utility += util_val

    return {
        "status": status,
        "selected_edges": selected_edges,
        "delta": delta.X,
        "objective": float(model.objVal),
        "risk_used": total_risk,
        "utility_used": total_utility,
    }

from motion_planning.constrained_shortest_path.gurobi_license import GUROBI_OPTIONS

def run_itm_online(
    edge_values_csv: Path | str,
    decision_timeline_csv: Path | str,
    *,
    capacity: Optional[float] = None,
    psi_t: Optional[float] = None,
    rho_min_override: Optional[float] = None,
    rho_max_override: Optional[float] = None,
    gurobi_params: Optional[Dict[str, Any]] = None,
    graph_pickle_path: Optional[Path | str] = None,
    output_root: Path | str = ONLINE_RESULTS_DIR / "graph-based",
    write_csv: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run ITM-ORB across all decision epochs defined by the timeline.
    For each epoch t, solve the MILP between decision_nodes[t] -> decision_nodes[t+1].
    """
    problem = load_graph_problem(edge_values_csv, decision_timeline_csv, capacity=capacity, graph_pickle_path=graph_pickle_path)
    Delta_total = capacity
    if Delta_total is None:
        Delta_total = _parse_risk_budget_from_stem(Path(edge_values_csv))

    # if Delta_total is None:
    #     # fallback: use mean risk * 5 as a crude estimate
    #     Delta_total = float(problem.edges_df["risk"].mean() * 5)

    # Compute rho bounds for CZL-based psi if not provided
    rho_min = rho_max = None
    if psi_t is None:
        # Expect candidates files next to edge_values (offline candidates opt not needed)
        if rho_min_override is not None and rho_max_override is not None:
            rho_min, rho_max = float(rho_min_override), float(rho_max_override)
        else:
            raise NotImplementedError("Temporary disable rho_min/rho_max auto-computation.")
            # cand_dir = Path(edge_values_csv).parent.parent / "problem details"
            # cand_files = sorted(cand_dir.glob("*_candidates.csv"))
            # if cand_files:
            #     rho_min, rho_max, _ = czl_thresholds(cand_files)
            # else:
            #     raise ValueError("No candidates files found to compute rho_min/rho_max for ITM.")

    # Reconstruct ordered decision nodes from timeline (index order).
    timeline = problem.timeline_df.sort_values("index")
    nodes_seq = timeline["decision_node"].astype(str).tolist()
    if len(nodes_seq) < 2:
        raise ValueError("Timeline must contain at least two decision nodes for ITM online execution.")
        # return []
    remaining = float(Delta_total)
    decision_layers = timeline['decision_epoch'].tolist() if 'decision_epoch' in timeline.columns else None
    graph_size = max(decision_layers) if decision_layers else None
    results: List[Dict[str, Any]] = []
    for idx in range(len(nodes_seq) - 1):
        vs = nodes_seq[idx]
        vg = nodes_seq[idx + 1]
        params = gurobi_params if gurobi_params is not None else GUROBI_OPTIONS
        epoch_size = None
        if decision_layers and idx < len(decision_layers) - 1:
            try:
                epoch_size = float(decision_layers[idx + 1] - decision_layers[idx])
            except Exception:
                epoch_size = None
        # compute psi for this epoch if not fixed
        psi_epoch = psi_t
        if psi_epoch is None and rho_min is not None and rho_max is not None:
            z = 1.0 - (remaining / Delta_total)
            if rho_max > 10000:
                rho_max = 10000  # cap to avoid overflow in exp
                print(f"  Capping rho_max to {rho_max} to avoid overflow.")

            psi_epoch = czl_psi(z, rho_min, rho_max)
            # psi_epoch = 1000 * 0.1  # scale down for ITM
        
        sol = solve_itm_epoch(
            problem.edges_df,
            start_node=vs,
            goal_node=vg,
            Delta_t=remaining,
            Delta_total=Delta_total,
            epoch_size=epoch_size,
            graph_size=graph_size,
            psi_t=psi_epoch,
            gurobi_params=params,
        )
        # Update remaining budget using allocated delta when available, else risk_used.
        if sol.get("delta") is not None:
            remaining = max(0.0, remaining - float(sol["delta"]))

        elif sol.get("risk_used") is not None:
            remaining = max(0.0, remaining - float(sol["risk_used"]))
        else:
            raise ValueError("Solution missing both 'delta' and 'risk_used' to update remaining budget.")
        
        results.append(
            {
                "epoch": idx,
                "start": vs,
                "goal": vg,
                "psi": psi_epoch,
                **sol,
                "remaining_after": remaining,
                "delta_used": sol.get("delta"),
            }
        )
        if remaining <= 0:
            break

    if write_csv and results:
        _write_itm_online_solution(
            Path(edge_values_csv),
            graph_id=problem.graph_id,
            edges_df=problem.edges_df,
            results=results,
            total_budget=Delta_total,
            output_root=output_root,
        )
    return results