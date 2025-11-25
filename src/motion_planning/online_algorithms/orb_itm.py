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


def _nodes_from_edges(edges_df: pd.DataFrame) -> List[str]:
    nodes = set(edges_df["start_node"].astype(str)).union(edges_df["end_node"].astype(str))
    return sorted(nodes)


def _parse_risk_budget_from_stem(path: Path) -> Optional[float]:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    try:
        return float(parts[-4])
    except Exception:
        return None


def solve_itm_epoch(
    edges_df: pd.DataFrame,
    start_node: str,
    goal_node: str,
    *,
    Delta_t: float,
    threshold: float = 0.0,
    psi_t: Optional[float] = None,
    gurobi_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Solve ITM-ORB for a single epoch between (start_node, goal_node).
    Returns selected edges with speeds, allocated risk δ_t, and objective.
    """
    if edges_df.empty:
        raise ValueError("edges_df is empty.")

    model = gp.Model("ITM_ORB_epoch")
    model.Params.OutputFlag = 0
    if gurobi_params:
        for k, v in gurobi_params.items():
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

    delta = model.addVar(lb=0.0, ub=float(Delta_t), name="delta_t")

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
    psi_val = psi_t if psi_t is not None else threshold  # unify naming (threshold ≡ Psi)
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


def run_itm_online(
    edge_values_csv: Path | str,
    decision_timeline_csv: Path | str,
    *,
    capacity: Optional[float] = None,
    threshold: float = 0.0,
    psi_t: Optional[float] = None,
    gurobi_params: Optional[Dict[str, Any]] = None,
    graph_pickle_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Run ITM-ORB across all decision epochs defined by the timeline.
    For each epoch t, solve the MILP between decision_nodes[t] -> decision_nodes[t+1].
    """
    problem = load_graph_problem(edge_values_csv, decision_timeline_csv, capacity=capacity, graph_pickle_path=graph_pickle_path)
    Delta_total = capacity
    if Delta_total is None:
        Delta_total = _parse_risk_budget_from_stem(Path(edge_values_csv))
    if Delta_total is None:
        Delta_total = float(problem.edges_df["risk"].mean() * 5)  # fallback

    # Reconstruct ordered decision nodes from timeline (index order).
    timeline = problem.timeline_df.sort_values("index")
    nodes_seq = timeline["decision_node"].astype(str).tolist()
    if len(nodes_seq) < 2:
        return []

    remaining = float(Delta_total)
    results: List[Dict[str, Any]] = []
    for idx in range(len(nodes_seq) - 1):
        vs = nodes_seq[idx]
        vg = nodes_seq[idx + 1]
        sol = solve_itm_epoch(
            problem.edges_df,
            start_node=vs,
            goal_node=vg,
            Delta_t=remaining,
            threshold=threshold,
            psi_t=psi_t,
            gurobi_params=gurobi_params,
        )
        # Update remaining budget conservatively
        if sol.get("risk_used") is not None:
            remaining = max(0.0, remaining - float(sol["risk_used"]))
        results.append({"epoch": idx, "start": vs, "goal": vg, **sol, "remaining_after": remaining})
        if remaining <= 0:
            break

    return results
