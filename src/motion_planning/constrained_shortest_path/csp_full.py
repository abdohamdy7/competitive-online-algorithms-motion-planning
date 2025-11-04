"""
Constrained shortest-path model that chooses both edges and speed options.

This solver extends :class:`CSP_BASIC` by introducing one binary variable per
edge-speed combination. The objective cost as well as the risk budget are
speed-aware, which allows the model to trade travelling faster (or slower)
against risk and time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Hashable, Iterable, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB, quicksum

try:  # pragma: no cover - convenience default
    from motion_planning.graph_construction.main_graph_construction import (
        SPEED_SET as DEFAULT_SPEED_SET,
    )
except Exception:  # pragma: no cover - fallback if import side-effects fail
    DEFAULT_SPEED_SET: Optional[Iterable[Any]] = None


class CSP_FULL:
    """
    Minimum-cost path from ``start_node`` to ``goal_node`` with speed-aware risks.

    Decision variables:
        x_(i,j,v) ∈ {0,1} — 1 if edge (i,j) is chosen at speed option ``v``.

    Objective:
        minimise Σ_(i,j,v) cost(i,j,v) * x_(i,j,v)

    Constraints:
        * Flow balance per node (net flow = supply/demand).
        * Optional risk budget across all (edge, speed) choices.
        * At most one speed can be selected for each directed edge.

    Parameters
    ----------
    Graph : networkx.Graph
        Directed graph containing edge attributes with cost lists.
    start_node, goal_node : Hashable
        Identifiers of the source and sink vertices.
    edge_cost_list_key : str, optional
        Edge attribute that stores a list of costs aligned with ``speed_options``.
        Defaults to ``"edge_time_cost_list"``.
    speed_options : Iterable[Any], optional
        Iterable of speed options. When omitted the solver tries to infer the
        values from the supplied ``risk_matrix`` or the global ``SPEED_SET``.
    risk_max : float, optional
        Maximum admissible aggregated risk. If omitted the risk budget constraint
        is not added.
    risk_matrix : dict[(u, v, speed), float], optional
        Risk values per (edge, speed). Missing entries default to zero risk.
    gurobi_env_params, model_name, time_limit, mip_gap, output
        Passed straight to Gurobi.
    """

    def __init__(
        self,
        Graph,
        start_node: Hashable,
        goal_node: Hashable,
        *,
        edge_cost_list_key: str = "edge_time_cost_list",
        speed_options: Optional[Iterable[Any]] = None,
        risk_max: Optional[float] = None,
        risk_matrix: Optional[Dict[Tuple[Hashable, Hashable, Any], float]] = None,
        gurobi_env_params: Optional[Dict[str, Any]] = None,
        model_name: str = "CSP_FULL",
        time_limit: Optional[int] = None,
        mip_gap: Optional[float] = None,
        output: bool = True,
    ) -> None:
        if start_node not in Graph or goal_node not in Graph:
            raise ValueError("start_node and goal_node must exist in Graph.")
        if start_node == goal_node:
            raise ValueError("start_node and goal_node must be different.")
        if risk_max is not None and risk_max < 0:
            raise ValueError("risk_max must be non-negative if provided.")

        self.G = Graph
        self._edges: List[Tuple[Hashable, Hashable, Dict[str, Any]]] = list(
            Graph.edges(data=True),
        )

        self.start = start_node
        self.goal = goal_node
        self.edge_cost_list_key = edge_cost_list_key

        self.risk_max = risk_max
        self.risk = risk_matrix or {}

        self.speed_options = self._resolve_speed_options(speed_options, self.risk)
        if not self.speed_options:
            raise ValueError(
                "speed_options could not be resolved. Provide them explicitly or "
                "ensure the risk matrix or SPEED_SET is available.",
            )

        self.gurobi_env_params = gurobi_env_params or {}
        self.model_name = model_name
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.output = output

        self.model: Optional[gp.Model] = None
        self.x: Dict[Tuple[Hashable, Hashable, Any], gp.Var] = {}
        self._edge_costs: Dict[Tuple[Hashable, Hashable, Any], float] = {}

        self._outgoing: DefaultDict[Hashable, List[Tuple[Hashable, Hashable, Any]]] = (
            defaultdict(list)
        )
        self._incoming: DefaultDict[Hashable, List[Tuple[Hashable, Hashable, Any]]] = (
            defaultdict(list)
        )

        self._built = False

    # ------------------------------------------------------------------ Utilities

    @staticmethod
    def _resolve_speed_options(
        explicit_speeds: Optional[Iterable[Any]],
        risk_matrix: Dict[Tuple[Hashable, Hashable, Any], float],
    ) -> Tuple[Any, ...]:
        if explicit_speeds is not None:
            speeds = list(explicit_speeds)
            if not speeds:
                raise ValueError("speed_options must contain at least one value.")
            return tuple(speeds)

        if risk_matrix:
            speeds = sorted({speed for _, _, speed in risk_matrix.keys()})
            if speeds:
                return tuple(speeds)

        if DEFAULT_SPEED_SET:
            speeds = list(DEFAULT_SPEED_SET)
            if speeds:
                return tuple(speeds)

        return tuple()

    # ------------------------------------------------------------------ Build model

    def _build(self) -> None:
        """Create variables, objective, and constraints."""
        with gp.Env(params=self.gurobi_env_params) as env:
            m = gp.Model(env=env, name=self.model_name)

            if self.time_limit is not None:
                m.Params.TimeLimit = self.time_limit
            if self.mip_gap is not None:
                m.Params.MIPGap = self.mip_gap
            m.Params.OutputFlag = 1 if self.output else 0

            for i, j, attr in self._edges:
                try:
                    cost_list = attr[self.edge_cost_list_key]
                except KeyError as exc:
                    raise KeyError(
                        f"Edge attribute '{self.edge_cost_list_key}' not found for edge "
                        f"{getattr(i, 'id', i)}->{getattr(j, 'id', j)}.",
                    ) from exc

                if len(cost_list) != len(self.speed_options):
                    raise ValueError(
                        f"Edge {getattr(i, 'id', i)}->{getattr(j, 'id', j)} provides "
                        f"{len(cost_list)} cost entries but {len(self.speed_options)} "
                        "speed options were supplied.",
                    )

                for speed, cost in zip(self.speed_options, cost_list):
                    key = (i, j, speed)
                    self.x[key] = m.addVar(
                        vtype=GRB.BINARY,
                        name=f"x[{getattr(i, 'id', i)}->{getattr(j, 'id', j)}@{speed}]",
                    )
                    self._edge_costs[key] = float(cost)
                    self._outgoing[i].append(key)
                    self._incoming[j].append(key)

            # Objective: minimise total cost over (edge, speed) selections
            m.setObjective(
                quicksum(self.x[key] * self._edge_costs[key] for key in self.x),
                GRB.MINIMIZE,
            )

            # Flow-balance constraints per node
            for v in self.G.nodes:
                out_vars = [self.x[key] for key in self._outgoing.get(v, [])]
                in_vars = [self.x[key] for key in self._incoming.get(v, [])]
                supply = 1 if v == self.start else (-1 if v == self.goal else 0)
                m.addConstr(
                    quicksum(out_vars) - quicksum(in_vars) == supply,
                    name=f"flow[{getattr(v, 'id', v)}]",
                )

            # Ensure each directed edge selects at most one speed
            seen_edges: Dict[Tuple[Hashable, Hashable], List[Tuple[Hashable, Hashable, Any]]] = defaultdict(list)
            for key in self.x:
                i, j, speed = key
                seen_edges[(i, j)].append(key)

            for (i, j), keys in seen_edges.items():
                m.addConstr(
                    quicksum(self.x[key] for key in keys) <= 1,
                    name=f"choose_speed[{getattr(i, 'id', i)}->{getattr(j, 'id', j)}]",
                )

            # Optional risk budget
            if self.risk_max is not None:
                m.addConstr(
                    quicksum(
                        self.x[key] * float(self.risk.get(key, 0.0))
                        for key in self.x
                    )
                    <= float(self.risk_max),
                    name="risk_budget",
                )

            self.model = m
            self._built = True

    # ---------------------------------------------------------------------- Solve

    def solve(self) -> Optional[List[Tuple[Hashable, Hashable, Any]]]:
        """Build (if needed) and solve the model, returning selected (edge, speed)."""
        if not self._built:
            self._build()

        assert self.model is not None
        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            return self._selected_edges_with_speed()
        elif self.model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            print("Model infeasible. Computing IIS...")
            try:
                self.model.computeIIS()
                self.model.write("csp_full_infeasible.ilp")
                print("IIS written to csp_full_infeasible.ilp")
                for constr in self.model.getConstrs():
                    if constr.IISConstr:
                        print(f"Infeasible: {constr.ConstrName}")
            except gp.GurobiError:
                print("Failed to compute IIS.")
            return None
        else:
            print(f"No optimal solution (status={self.model.Status}).")
            return None

    # -------------------------------------------------------------------- Helpers

    def _selected_edges_with_speed(self) -> List[Tuple[Hashable, Hashable, Any]]:
        return [key for key, var in self.x.items() if var.X > 0.5]

    def solution_path_nodes(self) -> Optional[List[Hashable]]:
        """
        Trace the node sequence from start to goal ignoring speed labels.
        Returns None if no solution or the path cannot be reconstructed.
        """
        selected = self._selected_edges_with_speed()
        if not selected:
            return None

        next_map: Dict[Hashable, Tuple[Hashable, Any]] = {}
        indeg: Dict[Hashable, int] = {}
        outdeg: Dict[Hashable, int] = {}

        for i, j, speed in selected:
            if i not in next_map:
                next_map[i] = (j, speed)
            outdeg[i] = outdeg.get(i, 0) + 1
            indeg[j] = indeg.get(j, 0) + 1
            outdeg.setdefault(j, 0)
            indeg.setdefault(i, 0)

        if outdeg.get(self.start, 0) == 0:
            return None

        path = [self.start]
        cur = self.start
        visited = {cur}
        while cur in next_map:
            nxt, _speed = next_map[cur]
            if nxt in visited:
                return None  # cycle detected
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
            if cur == self.goal:
                return path

        return None


__all__ = ["CSP_FULL"]

