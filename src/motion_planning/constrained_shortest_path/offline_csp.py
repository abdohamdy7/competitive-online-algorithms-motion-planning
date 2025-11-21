"""
Offline constrained shortest-path solver with required sub-goal visitation.

The model mirrors :class:`CSP_FULL` but augments it with constraints described in
the offline problem specification:

* Flow conservation at each node.
* Simple-path enforcement (at most one incoming/outgoing arc, except source/sink).
* Exactly one speed per physical edge.
* Every sub-goal node must be visited (exactly one incoming and one outgoing arc).
* Aggregate risk across selected (edge, speed) pairs must not exceed ``risk_budget``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB, quicksum

from motion_planning.constrained_shortest_path.costs_definitions import EDGE_COST

try:  # pragma: no cover - optional dependency from graph builder
    from motion_planning.graph_construction.main_graph_construction import (
        SPEED_SET as DEFAULT_SPEED_SET,
    )
except Exception:  # pragma: no cover - fallback to explicit input
    DEFAULT_SPEED_SET = None

EdgeSpeedKey = Tuple[Hashable, Hashable, Any]


class OfflineCSPModel:
    """Helper that builds and solves the offline CSP with sub-goal constraints."""

    def __init__(
        self,
        graph,
        start_node: Hashable,
        goal_node: Hashable,
        *,
        sub_goals: Iterable[Hashable],
        risk_matrix: Mapping[EdgeSpeedKey, float],
        risk_budget: float,
        edge_cost_list_key: str = EDGE_COST.EDGE_TIME_LATERAL_COST_LIST,
        speed_options: Optional[Iterable[Any]] = None,
        gurobi_env_params: Optional[Dict[str, Any]] = None,
        model_name: str = "OFFLINE_CSP",
        time_limit: Optional[int] = None,
        mip_gap: Optional[float] = None,
        output: bool = False,
    ) -> None:
        if start_node not in graph or goal_node not in graph:
            raise ValueError("start_node and goal_node must exist in the graph.")
        if start_node == goal_node:
            raise ValueError("start_node and goal_node must be different.")
        if risk_budget < 0:
            raise ValueError("risk_budget must be non-negative.")

        self.G = graph
        self.start = start_node
        self.goal = goal_node
        self.sub_goals = tuple(dict.fromkeys(sub_goals))
        if not self.sub_goals:
            raise ValueError("sub_goals must contain at least one node.")
        for node in self.sub_goals:
            if node not in graph:
                raise ValueError(
                    f"Sub-goal node {getattr(node, 'id', node)} not present in the graph.",
                )

        self.edge_cost_list_key = edge_cost_list_key
        self.risk_matrix = dict(risk_matrix)
        self.risk_budget = float(risk_budget)

        self.speed_options = self._resolve_speed_options(speed_options, self.risk_matrix)
        if not self.speed_options:
            raise ValueError(
                "speed_options could not be resolved. Provide them explicitly or "
                "ensure SPEED_SET is available.",
            )

        self.gurobi_env_params = gurobi_env_params or {}
        self.model_name = model_name
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.output = output

        self.model: Optional[gp.Model] = None
        self.x: Dict[EdgeSpeedKey, gp.Var] = {}
        self._edge_costs: Dict[EdgeSpeedKey, float] = {}
        self._outgoing = defaultdict(list)
        self._incoming = defaultdict(list)
        self._per_arc = defaultdict(list)
        self._built = False

    # ------------------------------------------------------------------ Utilities

    @staticmethod
    def _resolve_speed_options(
        explicit_speeds: Optional[Iterable[Any]],
        risk_matrix: Mapping[EdgeSpeedKey, float],
    ) -> Tuple[Any, ...]:
        if explicit_speeds is not None:
            speeds = list(explicit_speeds)
            if not speeds:
                raise ValueError("speed_options must contain at least one value.")
            return tuple(speeds)

        if risk_matrix:
            speeds = sorted({key[2] for key in risk_matrix.keys()})
            if speeds:
                return tuple(speeds)

        if DEFAULT_SPEED_SET:
            speeds = list(DEFAULT_SPEED_SET)
            if speeds:
                return tuple(speeds)

        return tuple()

    # ------------------------------------------------------------------ Build model

    def build(self) -> None:
        """Create all decision variables, the objective, and the constraints."""
        if self._built:
            return

        with gp.Env(params=self.gurobi_env_params) as env:
            model = gp.Model(env=env, name=self.model_name)
            if self.time_limit is not None:
                model.Params.TimeLimit = self.time_limit
            if self.mip_gap is not None:
                model.Params.MIPGap = self.mip_gap
            model.Params.OutputFlag = 1 if self.output else 0

            for i, j, attr in self.G.edges(data=True):
                try:
                    cost_list = attr[self.edge_cost_list_key]
                except KeyError as exc:
                    raise KeyError(
                        f"Edge attribute '{self.edge_cost_list_key}' missing for edge "
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
                    var = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x[{getattr(i, 'id', i)}->{getattr(j, 'id', j)}@{speed}]",
                    )
                    self.x[key] = var
                    self._edge_costs[key] = float(cost)
                    self._outgoing[i].append(key)
                    self._incoming[j].append(key)
                    self._per_arc[(i, j)].append(key)

            # Step 1: minimise travel cost across chosen (edge, speed) pairs.
            model.setObjective(
                quicksum(self.x[key] * self._edge_costs[key] for key in self.x),
                GRB.MINIMIZE,
            )

            for v in self.G.nodes:
                out_vars = [self.x[key] for key in self._outgoing.get(v, [])]
                in_vars = [self.x[key] for key in self._incoming.get(v, [])]
                supply = 1 if v == self.start else (-1 if v == self.goal else 0)
                # Step 2: classical flow conservation with source/sink supplies.
                model.addConstr(
                    quicksum(out_vars) - quicksum(in_vars) == supply,
                    name=f"flow[{getattr(v, 'id', v)}]",
                )

                # Step 3: simple-path enforcement (≤1 outgoing except sink, ≤1 incoming except source).
                if v != self.goal:
                    model.addConstr(
                        quicksum(out_vars) <= 1,
                        name=f"simple_out[{getattr(v, 'id', v)}]",
                    )
                if v != self.start:
                    model.addConstr(
                        quicksum(in_vars) <= 1,
                        name=f"simple_in[{getattr(v, 'id', v)}]",
                    )

            # Step 4: at most one speed per physical arc.
            for (i, j), keys in self._per_arc.items():
                model.addConstr(
                    quicksum(self.x[key] for key in keys) <= 1,
                    name=f"speed_unique[{getattr(i, 'id', i)}->{getattr(j, 'id', j)}]",
                )

            for r in self.sub_goals:
                in_vars = [self.x[key] for key in self._incoming.get(r, [])]
                out_vars = [self.x[key] for key in self._outgoing.get(r, [])]
                # Step 5: every required sub-goal must be visited exactly once.
                if r != self.start:
                    model.addConstr(
                        quicksum(in_vars) == 1,
                        name=f"visit_in[{getattr(r, 'id', r)}]",
                    )
                if r != self.goal:
                    model.addConstr(
                        quicksum(out_vars) == 1,
                        name=f"visit_out[{getattr(r, 'id', r)}]",
                    )

            # Step 6: keep total path risk within budget Δ.
            model.addConstr(
                quicksum(
                    self.x[key] * float(self.risk_matrix.get(key, 0.0))
                    for key in self.x
                )
                <= self.risk_budget,
                name="risk_budget",
            )

            self.model = model
            self._built = True

    # ---------------------------------------------------------------------- Solve

    def solve(self) -> Optional[List[EdgeSpeedKey]]:
        if not self._built:
            self.build()

        assert self.model is not None
        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            return self._selected_edges()
        elif self.model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            print("Offline CSP model infeasible. Attempting IIS export...")
            try:
                self.model.computeIIS()
                self.model.write("offline_csp_infeasible.ilp")
                for constr in self.model.getConstrs():
                    if constr.IISConstr:
                        print(f"Infeasible constraint: {constr.ConstrName}")
            except gp.GurobiError:
                print("Failed to compute IIS for offline CSP.")
            return None
        else:
            print(f"Offline CSP solver ended with status {self.model.Status}.")
            return None

    # -------------------------------------------------------------------- Helpers

    def _selected_edges(self) -> List[EdgeSpeedKey]:
        return [key for key, var in self.x.items() if var.X > 0.5]

    def solution_path_nodes(self) -> Optional[List[Hashable]]:
        selected = self._selected_edges()
        if not selected:
            return None

        next_node: Dict[Hashable, Hashable] = {}
        visited = set()
        for u, v, _ in selected:
            next_node[u] = v

        path = [self.start]
        cur = self.start
        visited.add(cur)
        while cur in next_node:
            nxt = next_node[cur]
            if nxt in visited:
                return None
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
            if cur == self.goal:
                return path
        return None


def get_optimal_offline_CSP(
    graph,
    start_node: Hashable,
    goal_node: Hashable,
    *,
    sub_goals: Iterable[Hashable],
    risk_matrix: Mapping[EdgeSpeedKey, float],
    risk_budget: float,
    edge_cost_list_key: str = EDGE_COST.EDGE_TIME_LATERAL_COST_LIST,
    speed_options: Optional[Iterable[Any]] = None,
    gurobi_env_params: Optional[Dict[str, Any]] = None,
    model_name: str = "OFFLINE_CSP",
    time_limit: Optional[int] = None,
    mip_gap: Optional[float] = None,
    output: bool = False,
) -> Tuple[Optional[List[EdgeSpeedKey]], OfflineCSPModel]:
    """
    Build and solve the offline CSP model returning the selected (edge, speed) pairs.

    The function instantiates :class:`OfflineCSPModel`, builds the Gurobi model,
    solves it, and returns both the solution and the solver instance so callers
    can inspect the model, objective, or call ``solution_path_nodes``.
    """

    solver = OfflineCSPModel(
        graph,
        start_node,
        goal_node,
        sub_goals=sub_goals,
        risk_matrix=risk_matrix,
        risk_budget=risk_budget,
        edge_cost_list_key=edge_cost_list_key,
        speed_options=speed_options,
        gurobi_env_params=gurobi_env_params,
        model_name=model_name,
        time_limit=time_limit,
        mip_gap=mip_gap,
        output=output,
    )
    solver.build()
    solution = solver.solve()
    return solution, solver


__all__ = ["OfflineCSPModel", "get_optimal_offline_CSP", "EdgeSpeedKey"]
