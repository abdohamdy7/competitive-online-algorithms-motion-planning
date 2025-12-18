"""
Candidate-path generation utilities for offline motion-planning problems.

This module exposes two entry points:

* ``generate_all_candidate_paths`` enumerates all (simple) node paths between a
  start/subgoal pair and converts them into :class:`CANDIDATE_PATH` objects.
* ``generate_risk_bounded_candidate_paths`` repeatedly solves the full CSP
  while tightening the risk budget to gather diverse feasible solutions.

Both helpers share common aggregation logic so that every candidate exposes
risk, cost, utility, and Frenet progress in a consistent fashion.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Hashable, List, Optional, Tuple, Set, Sequence, Mapping

import networkx as nx
import numpy as np

from motion_planning.constrained_shortest_path.csp_full import CSP_FULL
from motion_planning.utils.candidate_path import CANDIDATE_PATH
from motion_planning.graph_construction.main_graph_construction import SPEED_SET
from motion_planning.constrained_shortest_path.gurobi_license import GUROBI_OPTIONS
from motion_planning.constrained_shortest_path.costs_definitions import EDGE_COST
import numpy as np
from motion_planning.offline_problems.utils import BIG_M


__all__ = [
    "generate_all_candidate_paths",
    "generate_risk_bounded_candidate_paths",
]

EdgeSpeedKey = Tuple[Hashable, Hashable, Any]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _speed_options() -> Tuple[Any, ...]:
    """Return a tuple with the configured speed options (defaults to a dummy 0)."""
    speeds = SPEED_SET
    if speeds is None:
        return (0,)
    try:
        options = tuple(speeds)  # type: ignore[arg-type]
    except TypeError:
        options = (speeds,)  # single numeric value
    return options or (0,)


def _speed_index(speed: Any) -> Optional[int]:
    speeds = _speed_options()
    try:
        return speeds.index(speed)
    except ValueError:
        return None


def _resolve_edge_cost(attr: Dict[str, Any], edge_cost_key: str, speed: Any) -> float:
    """Pick the cost associated with ``speed`` (fallback to the first entry)."""
    cost_list = attr.get(edge_cost_key)
    if isinstance(cost_list, (list, tuple)) and cost_list:
        idx = _speed_index(speed)
        if idx is not None and idx < len(cost_list):
            return float(cost_list[idx])
        return float(cost_list[0])
    return float(attr.get("edge_cost", 0.0))


def _safe_ratio(utility: float, risk: float) -> Optional[float]:
    return utility / risk if risk and abs(risk) > 1e-9 else None


def _edge_attr(graph: nx.Graph, u: Hashable, v: Hashable) -> Dict[str, Any]:
    try:
        return graph.edges[u, v]
    except KeyError as exc:  # pragma: no cover - sanity guard
        raise KeyError(f"Edge attributes missing for ({u}, {v})") from exc


def _build_candidate(
    *,
    graph: nx.Graph,
    start_node: Hashable,
    sub_goal_node: Hashable,
    solution_edge_speeds: List[EdgeSpeedKey],
    cost_matrix: Dict[EdgeSpeedKey, float],
    risk_matrix: Dict[EdgeSpeedKey, float],
    utility_matrix: Dict[EdgeSpeedKey, float],
) -> CANDIDATE_PATH:
    total_risk = sum(risk_matrix.get(edge, 0.0) for edge in solution_edge_speeds)
    total_cost = sum(cost_matrix.get(edge, 0.0) for edge in solution_edge_speeds)
    total_utility = sum(utility_matrix.get(edge, 0.0) for edge in solution_edge_speeds)

    total_frenet = 0.0
    total_time = 0.0
    for u, v, _ in solution_edge_speeds:
        attr = _edge_attr(graph, u, v)
        total_frenet += float(attr.get("frenet_progress", 0.0))
        total_time += float(attr.get("time_progress", 0.0))

    return CANDIDATE_PATH(
        risk=total_risk,
        cost=total_cost,
        utility=total_utility,
        start_node=start_node,
        subgoal_node=sub_goal_node,
        solution_edge_speeds=solution_edge_speeds,
        cost_matrix=cost_matrix,
        risk_matrix=risk_matrix,
        utility_matrix=utility_matrix,
        frenet_progress=total_frenet,
        time_progress=total_time,
        ratio=_safe_ratio(total_utility, total_risk),
    )


def _risk_levels(delta: float, step: float) -> List[float]:
    """Monotone list of risk budgets up to ``delta`` (inclusive)."""
    if delta <= 0:
        return []
    step = max(step, 1e-3)
    levels = [round(i * step, 5) for i in range(1, int(delta / step) + 1)]
    if not levels or not math.isclose(levels[-1], delta, rel_tol=1e-5, abs_tol=1e-5):
        levels.append(round(delta, 5))
    return levels


# --------------------------------------------------------------------------- #
# Public APIs
# --------------------------------------------------------------------------- #

def generate_all_candidate_paths(
    graph: nx.DiGraph,
    start_node: Hashable,
    sub_goal_node: Hashable,
    *,
    max_simple_paths: Optional[int] = 50,
    cutoff: Optional[int] = None,
    edge_cost_key: str = EDGE_COST.EDGE_TIME_LATERAL_COST_LIST,
) -> List[CANDIDATE_PATH]:
    """
    Enumerate simple paths from ``start_node`` to ``sub_goal_node``.

    Args:
        graph: Planning graph (directed).
        start_node: Start vertex.
        sub_goal_node: Destination vertex.
        max_simple_paths: Optional cap to avoid combinatorial explosion.
        cutoff: Optional length cutoff passed to ``nx.all_simple_paths``.
        edge_cost_key: Edge attribute storing the per-speed cost list.

    Returns:
        List of :class:`CANDIDATE_PATH` objects. The list is empty when no path
        exists. Speeds default to the first entry inside ``SPEED_SET``.
    """

    if start_node not in graph or sub_goal_node not in graph:
        raise ValueError("start_node and sub_goal_node must exist in the graph.")

    try:
        print(f'now trying to generating simple paths from  start node {start_node.id} to goal node {sub_goal_node.id}')
        layer_gap = sub_goal_node.layer  - start_node.layer 
        
        simple_paths = nx.all_simple_paths(graph, start_node, sub_goal_node, cutoff=layer_gap+1)
        print(f'done generated simple paths with gap of {layer_gap} ! number of paths: {(len(list(simple_paths)))}')
        for path in simple_paths:
            for u,v in simple_paths:
                print(f'edge from {u.id} to {v.id}')

                    
    except nx.NetworkXNoPath:
        return []
    candidates: List[CANDIDATE_PATH] = []
    default_speed = _speed_options()[0]

    for idx, path in enumerate(simple_paths):
        if max_simple_paths is not None and idx >= max_simple_paths:
            break
        if len(path) < 2:
            continue

        solution_edge_speeds: List[EdgeSpeedKey] = []
        cost_matrix: Dict[EdgeSpeedKey, float] = {}
        risk_matrix: Dict[EdgeSpeedKey, float] = {}
        utility_matrix: Dict[EdgeSpeedKey, float] = {}

        for u, v in zip(path[:-1], path[1:]):
            attr = _edge_attr(graph, u, v)
            edge_speed = default_speed
            cost = _resolve_edge_cost(attr, edge_cost_key, edge_speed)
            risk = float(attr.get("edge_risk", 0.0))
            utility = BIG_M - cost

            edge_key = (u, v, edge_speed)
            solution_edge_speeds.append(edge_key)
            cost_matrix[edge_key] = cost
            risk_matrix[edge_key] = risk
            utility_matrix[edge_key] = utility

        if solution_edge_speeds:
            candidates.append(
                _build_candidate(
                    graph=graph,
                    start_node=start_node,
                    sub_goal_node=sub_goal_node,
                    solution_edge_speeds=solution_edge_speeds,
                    cost_matrix=cost_matrix,
                    risk_matrix=risk_matrix,
                    utility_matrix=utility_matrix,
                )
            )

    return candidates


def _path_signature(solution_edge_speeds: Sequence[EdgeSpeedKey]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Return (node_ids, speeds) signature for deduplicating candidate paths."""
    node_ids: List[Any] = []
    speed_ids: List[Any] = []
    for idx, (u, v, speed) in enumerate(solution_edge_speeds):
        if idx == 0:
            node_ids.append(getattr(u, "id", u))
        node_ids.append(getattr(v, "id", v))
        speed_ids.append(speed)
    return (tuple(node_ids), tuple(speed_ids))


def generate_risk_bounded_candidate_paths(
    graph: nx.DiGraph,
    start_node: Hashable,
    sub_goal_node: Hashable,
    Delta: float,
    PHI_ij: Dict[EdgeSpeedKey, float],
    edge_cost_key: str = EDGE_COST.EDGE_TIME_LATERAL_COST_LIST,
    # risk_step: float = 1,
    risk_step: float = 0.2,
    max_candidates: Optional[int] = None,
    epoch_size: Optional[float] = None,
    graph_size: Optional[float] = None,
) -> Optional[List[CANDIDATE_PATH]]:
    """
    Generate candidate paths using CSP_FULL under incremental risk budgets.

    Args:
        graph: Planning graph.
        start_node: Start node.
        sub_goal_node: Destination node.
        Delta: Maximum admissible risk.
        PHI_ij: Risk matrix keyed by (u, v, speed).
        edge_cost_key: Attribute that stores per-speed cost lists.
        risk_step: Granularity of the risk budget sweep.
        max_candidates: Optional cap on the number of feasible paths returned.

    Returns:
        List of :class:`CANDIDATE_PATH` objects, or ``None`` if no feasible path
        exists for any risk level (the caller can regenerate risks in that case).
    """

    if Delta <= 0:
        return None

    candidates: List[CANDIDATE_PATH] = []
    seen_paths: Set[Tuple[Tuple[Any, ...], Tuple[Any, ...]]] = set()
    # Scale delta by epoch size relative to full graph size: eta = Delta / graph_size, modified_delta = ceil(eta * epoch_size)
    if epoch_size and epoch_size > 0 and graph_size and graph_size > 0:
        eta = Delta / graph_size
        modified_delta = float(np.round(eta * epoch_size, 2))
        print(f'scaling risk budget from {Delta} to {modified_delta} using epoch size {epoch_size} and graph size {graph_size}')
    # else:
    #     # fallback: original heuristic
    #     scale = epoch_size if epoch_size and epoch_size > 0 else 3.0
    #     modified_delta = Delta / scale
    else:
        raise ValueError("epoch_size and graph_size must be positive values.")
    
    
    for risk_cap in _risk_levels(modified_delta, risk_step):
        print(f'risk cap is {risk_cap}')

        csp_full = CSP_FULL(
            graph,
            start_node,
            sub_goal_node,
            edge_cost_list_key=edge_cost_key,
            speed_options=_speed_options(),
            gurobi_env_params=GUROBI_OPTIONS,
            risk_max=risk_cap,
            risk_matrix=PHI_ij,
            output=False,
        )

        solution_edge_speeds = csp_full.solve()
        if not solution_edge_speeds:
            print(
                f"[CSP_FULL] No feasible path found for risk bound "
                f"{risk_cap:.2f} from {getattr(start_node, 'id', start_node)} "
                f"to {getattr(sub_goal_node, 'id', sub_goal_node)}."
            )

            # return None
            continue

        cost_matrix: Dict[EdgeSpeedKey, float] = {}
        risk_matrix: Dict[EdgeSpeedKey, float] = {}
        utility_matrix: Dict[EdgeSpeedKey, float] = {}

        for edge in solution_edge_speeds:
            cost = csp_full._edge_costs[edge]
            cost_matrix[edge] = cost
            risk_matrix[edge] = PHI_ij.get(edge, 0.0)
            utility_matrix[edge] = BIG_M - cost

        path_signature = _path_signature(solution_edge_speeds)
        print(f'path signature: {path_signature}')
        if path_signature in seen_paths:
            continue
        seen_paths.add(path_signature)

        candidates.append(
            _build_candidate(
                graph=graph,
                start_node=start_node,
                sub_goal_node=sub_goal_node,
                solution_edge_speeds=solution_edge_speeds,
                cost_matrix=cost_matrix,
                risk_matrix=risk_matrix,
                utility_matrix=utility_matrix,
            )
        )

        if max_candidates is not None and len(candidates) >= max_candidates:
            break

    return candidates or None





# -------------- #
# new code 
#--------------- #

import math
import networkx as nx
from typing import Hashable, Optional, List, Dict, Tuple

# ---- Assumed available from your codebase ----
# CANDIDATE_PATH, EdgeSpeedKey, BIG_M
# _build_candidate(graph, start_node, sub_goal_node, solution_edge_speeds, cost_matrix, risk_matrix, utility_matrix)
# _edge_attr(graph, u, v)
# _resolve_edge_cost(attr, edge_cost_key, edge_speed)
# _speed_options()

def _ensure_dag(G: nx.DiGraph) -> None:
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG for this enumerator.")

def _min_remaining_risk_dag(
    G: nx.DiGraph,
    t: Hashable,
    edge_min_risk: Mapping[Tuple[Hashable, Hashable], float],
) -> Dict[Hashable, float]:
    """Exact lower bound: rmin[v] = minimum possible risk from v to t (single reverse-topo pass)."""
    order = list(reversed(list(nx.topological_sort(G))))
    rmin = {v: math.inf for v in G}
    rmin[t] = 0.0
    for v in order:
        if v == t:
            continue
        best = math.inf
        for _, w in G.out_edges(v):
            c = float(edge_min_risk.get((v, w), math.inf))
            if c + rmin[w] < best:
                best = c + rmin[w]
        rmin[v] = best
    return rmin

def _integerize_risks_floor(
    edge_min_risk: Mapping[Tuple[Hashable, Hashable], float],
    scale: int,
) -> Dict[Tuple[Hashable, Hashable], int]:
    """
    Return integer risks via SAFE flooring: c_int(u,v) = floor(scale * c(u,v)).
    Using these ONLY to detect 'no feasible suffix' is exact: zero in DP => truly zero in reals.
    """
    risk_int: Dict[Tuple[Hashable, Hashable], int] = {}
    for key, value in edge_min_risk.items():
        risk_int[key] = int(math.floor(scale * float(value) + 1e-12))
    return risk_int

def _dp_zero_feasible_suffix(
    G: nx.DiGraph,
    t: Hashable,
    Delta_int: int,
    risk_int: Dict[Tuple[Hashable, Hashable], int],
) -> Dict[Hashable, List[int]]:
    """
    cnt[v][B] = number of v->t suffixes with integer risk <= B.
    We only ever check '== 0' to prune; absolute values can be huge.
    Time O(m * Delta_int), memory O(n * Delta_int).
    """
    cnt = {v: [0] * (Delta_int + 1) for v in G}
    for B in range(Delta_int + 1):
        cnt[t][B] = 1
    order = list(reversed(list(nx.topological_sort(G))))
    for v in order:
        if v == t:
            continue
        outs = list(G.out_edges(v))
        for B in range(Delta_int + 1):
            s = 0
            for (u, w) in outs:
                c = risk_int[(u, w)]
                if c <= B:
                    s += cnt[w][B - c]
            cnt[v][B] = s
    return cnt

def _edge_min_risk_map(
    graph: nx.DiGraph,
    PHI_ij: Mapping[EdgeSpeedKey, float],
    edge_risk_key: str,
) -> Dict[Tuple[Hashable, Hashable], float]:
    """Return the minimum available risk per (u, v) across all speed options."""
    speeds = _speed_options()
    edge_min: Dict[Tuple[Hashable, Hashable], float] = {}
    for u, v in graph.edges():
        best = math.inf
        for speed in speeds:
            key = (u, v, speed)
            if key in PHI_ij:
                risk_val = float(PHI_ij[key])
                if risk_val < best:
                    best = risk_val
        if math.isinf(best):
            attr = _edge_attr(graph, u, v)
            best = float(attr.get(edge_risk_key, 0.0))
        edge_min[(u, v)] = best
    return edge_min

def generate_all_candidate_paths_under_budget(
    graph: nx.DiGraph,
    start_node: Hashable,
    sub_goal_node: Hashable,
    *,
    delta_risk: float,                                # Δ (float)
    PHI_ij: Mapping[EdgeSpeedKey, float],
    edge_risk_key: str = "edge_risk",
    edge_cost_key: str = EDGE_COST.EDGE_TIME_LATERAL_COST_LIST,
    # Enumeration controls
    max_simple_paths: Optional[int] = None,           # cap outputs if desired; None = enumerate all
    sort_successors: bool = True,                     # expand promising edges first
    # Integer-DP pruning controls (for float risks)
    enable_dp: bool = True,                           # allow DP-based zero-feasibility pruning
    risk_scale: int = 1000,                           # scale S (e.g., 3 decimal places)
    max_scaled_budget: int = 500_000,                 # build DP only if floor(Δ*S) <= this
) -> List[CANDIDATE_PATH]:
    """
    Enumerate ALL s->t paths with total risk <= Δ on a DAG across all speed choices.
    Risks are sourced from ``PHI_ij`` so every candidate is defined by specific (edge, speed)
    selections. Exact correctness with floats:
      - Always uses exact rmin pruning (float).
      - Optionally uses integer DP with SAFE flooring to prune branches that cannot finish under Δ.
        This never prunes a truly feasible float path.
    Complexity:
      Preproc: O(m) for rmin; optional O(m * floor(Δ*S)) for DP.
      Output:  O(sum_{printed paths} path_length) + small per-branch checks.
    """
    if start_node not in graph or sub_goal_node not in graph:
        raise ValueError("start_node and sub_goal_node must exist in the graph.")

    # first time only
    print("ensuring dag graph stage !!!!")
    _ensure_dag(graph)

    # first time only
    print("edge min risk map stage !!!!")
    edge_min_risk = _edge_min_risk_map(graph, PHI_ij, edge_risk_key)

    
    # 1) Exact lower bounds in floats
    rmin = _min_remaining_risk_dag(graph, sub_goal_node, edge_min_risk)
    if rmin[start_node] > float(delta_risk):
        return []

    # 2) Optional integer DP for stronger pruning (exactly safe with floor scaling)
    use_dp = False
    cnt = None
    Delta_int = None
    risk_int = None
    if enable_dp:
        Delta_int = int(math.floor(delta_risk * risk_scale + 1e-12))
        if Delta_int >= 0 and Delta_int <= max_scaled_budget:
            risk_int = _integerize_risks_floor(edge_min_risk, risk_scale)
            cnt = _dp_zero_feasible_suffix(graph, sub_goal_node, Delta_int, risk_int)
            use_dp = True
        # else: fall back to rmin-only (still exact)

    print("ordering by edge risk + rmin")
    # 3) Cache successors; optionally order by (edge risk + rmin[next]) to find/complete paths early
    out_cache: Dict[Hashable, List[Tuple[Hashable, Dict]]] = {
        v: [(w, graph[v][w]) for w in graph.successors(v)]
        for v in graph.nodes()
    }
    def _ordered_succ(v: Hashable) -> List[Tuple[Hashable, Dict]]:
        succ = out_cache.get(v, [])
        if not sort_successors:
            return succ
        return sorted(succ, key=lambda p: edge_min_risk[(v, p[0])] + rmin[p[0]])

    speed_options = _speed_options()

    # 4) DFS with strong pruning
    candidates: List[CANDIDATE_PATH] = []

    def _emit_candidate(path: List[Hashable], chosen_speeds: List[Any]) -> CANDIDATE_PATH:
        solution_edge_speeds: List[EdgeSpeedKey] = []
        cost_matrix: Dict[EdgeSpeedKey, float] = {}
        risk_matrix: Dict[EdgeSpeedKey, float] = {}
        utility_matrix: Dict[EdgeSpeedKey, float] = {}
        for idx in range(len(path) - 1):
            u = path[idx]
            v = path[idx + 1]
            edge_speed = chosen_speeds[idx]
            attr = _edge_attr(graph, u, v)
            cost = _resolve_edge_cost(attr, edge_cost_key, edge_speed)
            risk = float(PHI_ij.get((u, v, edge_speed), attr.get(edge_risk_key, 0.0)))
            utility = BIG_M - cost
            key = (u, v, edge_speed)
            solution_edge_speeds.append(key)
            cost_matrix[key] = cost
            risk_matrix[key] = risk
            utility_matrix[key] = utility
        return _build_candidate(
            graph=graph,
            start_node=start_node,
            sub_goal_node=sub_goal_node,
            solution_edge_speeds=solution_edge_speeds,
            cost_matrix=cost_matrix,
            risk_matrix=risk_matrix,
            utility_matrix=utility_matrix,
        )

    StackItem = Tuple[Hashable, List[Hashable], List[Any], float]
    stack: List[StackItem] = [(start_node, [start_node], [], 0.0)]

    print("now the stacking process!!!")
    while stack:
        v, path, chosen_speeds, g = stack.pop()

        if v == sub_goal_node:
            # g is guaranteed <= Δ by pruning
            candidates.append(_emit_candidate(path, chosen_speeds))
            if max_simple_paths is not None and len(candidates) >= max_simple_paths:
                break
            continue

        for w, data in _ordered_succ(v):
            for speed in speed_options:
                key = (v, w, speed)
                risk_val = PHI_ij.get(key)
                if risk_val is None:
                    continue
                risk_val = float(risk_val)
                ng = g + risk_val

                # Exact lower-bound prune in floats
                if ng + rmin[w] > float(delta_risk):
                    continue

                # Optional exact (safe) DP zero-feasibility prune
                if use_dp:
                    rem_int = int(math.floor((delta_risk - ng) * risk_scale + 1e-12))
                    if rem_int < 0 or cnt[w][rem_int] == 0:
                        # If cnt==0 with floored scaling, then truly no float-feasible suffix exists
                        continue

                stack.append((w, path + [w], chosen_speeds + [speed], ng))

    return candidates
