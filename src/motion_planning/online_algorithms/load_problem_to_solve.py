"""
Helpers to load offline problem artifacts into in-memory structures that the
online algorithms (CZL-ORB, BAT-ORB, ITM) can consume.

The goal is to avoid re-reading multiple CSVs in each algorithm and to provide
uniform accessors for candidates-based and graph-based problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd

from motion_planning.offline_problems.solve_candidates_based_offline_problem import (
    load_candidate_groups_from_csv,
)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class CandidateGroup:
    epoch: int
    start_node: Optional[str]
    goal_node: Optional[str]
    candidates: List[Dict[str, Any]]


@dataclass
class CandidatesProblem:
    source_path: Path
    graph_id: str
    groups: List[CandidateGroup]
    capacity: float
    decision_nodes: List[str]
    raw_df: pd.DataFrame


@dataclass
class GraphProblem:
    source_edges_path: Path
    source_timeline_path: Path
    edges_df: pd.DataFrame
    timeline_df: pd.DataFrame
    capacity: Optional[float]
    graph_id: str
    graph: Optional[nx.DiGraph] = None


# --------------------------------------------------------------------------- #
# Candidates-based loader and writer helpers
# --------------------------------------------------------------------------- #

def load_candidates_problem(csv_path: Path | str, *, capacity_override: Optional[float] = None) -> CandidatesProblem:
    """
    Load a candidates-based offline problem from its candidates CSV.

    Returns a CandidatesProblem with grouped candidates per epoch.
    """
    utils, risks, ids, decision_nodes, capacity, df = load_candidate_groups_from_csv(
        csv_path, capacity_override=capacity_override
    )
    path = Path(csv_path)
    graph_id = str(df["graph_id"].iloc[0]) if "graph_id" in df.columns and not df.empty else path.stem

    groups: List[CandidateGroup] = []
    for epoch in sorted(df["epoch"].unique()):
        group_df = df[df["epoch"] == epoch]
        start_node = str(group_df["start_node"].dropna().iloc[0]) if "start_node" in group_df.columns else None
        goal_node = str(group_df["goal_node"].dropna().iloc[0]) if "goal_node" in group_df.columns else None
        groups.append(
            CandidateGroup(
                epoch=int(epoch),
                start_node=start_node,
                goal_node=goal_node,
                candidates=group_df.to_dict(orient="records"),
            )
        )

    return CandidatesProblem(
        source_path=path,
        graph_id=graph_id,
        groups=groups,
        capacity=float(capacity),
        decision_nodes=decision_nodes,
        raw_df=df,
    )


def write_online_solution_candidates(
    problem: CandidatesProblem,
    selections: Sequence[Optional[int]],
    algorithm: str,
    *,
    remaining_budget: Optional[Sequence[Optional[float]]] = None,
    output_root: Path | str = Path("results/data/online solutions/candidates"),
    suffix: str = "online",
) -> Path:
    """
    Persist the online selection for a candidates-based problem.

    selections: index per epoch (aligned with problem.groups), None to skip.
    remaining_budget: optional remaining budget after each choice.
    """
    rows: List[Dict[str, Any]] = []
    for idx, group in enumerate(problem.groups):
        sel = selections[idx] if idx < len(selections) else None
        if sel is None or sel >= len(group.candidates):
            continue
        row = dict(group.candidates[sel])  # copy
        row["algorithm"] = algorithm
        if remaining_budget is not None and idx < len(remaining_budget):
            row["remaining_budget"] = remaining_budget[idx]
        rows.append(row)

    if not rows:
        raise ValueError("No selections to write for online solution.")

    df = pd.DataFrame(rows)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = problem.source_path.stem.replace("_candidates", f"_{suffix}_{algorithm}")
    out_path = output_root / f"{stem}.csv"
    df.to_csv(out_path, index=False)
    return out_path


# --------------------------------------------------------------------------- #
# Graph-based loader (skeleton)
# --------------------------------------------------------------------------- #

def load_graph_problem(
    edge_values_csv: Path | str,
    decision_timeline_csv: Path | str,
    *,
    capacity: Optional[float] = None,
    graph_pickle_path: Optional[Path | str] = None,
) -> GraphProblem:
    """
    Load a graph-based offline problem (edges + decision timeline) and, if
    provided, the underlying pickled NetworkX graph.
    """
    edges_df = pd.read_csv(edge_values_csv)
    timeline_df = pd.read_csv(decision_timeline_csv)
    graph_id = str(edges_df["graph_id"].iloc[0]) if "graph_id" in edges_df.columns and not edges_df.empty else Path(edge_values_csv).stem
    graph_obj = None
    if graph_pickle_path is not None:
        graph_obj = nx.read_gpickle(graph_pickle_path)
    return GraphProblem(
        source_edges_path=Path(edge_values_csv),
        source_timeline_path=Path(decision_timeline_csv),
        edges_df=edges_df,
        timeline_df=timeline_df,
        capacity=capacity,
        graph_id=graph_id,
        graph=graph_obj,
    )


# --------------------------------------------------------------------------- #
# Lightweight runner stubs / examples
# --------------------------------------------------------------------------- #

def greedy_utility_policy(problem: CandidatesProblem) -> Tuple[List[int], List[float]]:
    """
    Simple example policy for candidates-based problems:
    pick the highest-utility candidate that fits the remaining budget.
    """
    remaining = problem.capacity
    selections: List[int] = []
    remaining_budget: List[float] = []

    for group in problem.groups:
        best_idx = None
        best_util = float("-inf")
        for idx, cand in enumerate(group.candidates):
            risk = float(cand.get("risk", 0.0))
            util = float(cand.get("utility", 0.0))
            if risk <= remaining and util > best_util:
                best_util = util
                best_idx = idx
        if best_idx is None:
            best_idx = 0  # fallback: take first if none fit
            risk = float(group.candidates[best_idx].get("risk", 0.0))
        else:
            risk = float(group.candidates[best_idx].get("risk", 0.0))
        remaining -= risk
        selections.append(best_idx)
        remaining_budget.append(remaining)

    return selections, remaining_budget


def run_candidates_policy(
    candidates_csv: Path | str,
    algorithm: str = "greedy_utility",
    *,
    capacity_override: Optional[float] = None,
    policy_fn=greedy_utility_policy,
    output_root: Path | str = Path("results/data/online solutions/candidates"),
) -> Path:
    """
    Convenience runner: load a candidates problem, apply a policy, and write output.
    """
    problem = load_candidates_problem(candidates_csv, capacity_override=capacity_override)
    selections, remaining = policy_fn(problem)
    return write_online_solution_candidates(
        problem,
        selections,
        algorithm=algorithm,
        remaining_budget=remaining,
        output_root=output_root,
    )
