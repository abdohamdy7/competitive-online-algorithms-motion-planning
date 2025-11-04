# -*- coding: utf-8 -*-
from typing import Any, Dict, Hashable, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB, quicksum

class CSP_BASIC:
    """
    Minimum-cost path from start_node to goal_node with an optional additive risk budget.

    Variables:
        x_(i,j) ∈ {0,1}  — 1 if edge (i,j) is chosen in the path, else 0.

    Objective:
        minimize  Σ_(i,j) cost(i,j) * x_(i,j), where cost is read from an edge attribute.

    Constraints (flow-balance form):
        For each node v:
            Σ_(v,j) x_(v,j) - Σ_(i,v) x_(i,v) = b(v),
        where b(start_node)=+1, b(goal_node)=-1, and 0 otherwise.

    Optional risk budget:
        Σ_(i,j) risk(i,j) * x_(i,j) ≤ risk_max

    Notes:
      - Set `edge_cost_key` to the graph edge attribute you want to minimize
        (default: 'edge_cost').
      - Provide `risk_max` and `risk_matrix` {(i,j): risk} to activate the budget.
      - If you want to minimize travel *time*, precompute time per edge and store
        it in the attribute you set with `edge_cost_key`.
    """

    def __init__(
        self,
        Graph,
        start_node: Hashable,
        goal_node: Hashable,
        *,
        edge_cost_key: str = "edge_cost",
        risk_max: Optional[float] = None,
        risk_matrix: Optional[Dict[Tuple[Hashable, Hashable], float]] = None,
        gurobi_env_params: Optional[Dict[str, Any]] = None,
        model_name: str = "CSP",
        time_limit: Optional[int] = None,
        mip_gap: Optional[float] = None,
        output: bool = True,
    ) -> None:
        # Basic input checks
        if start_node not in Graph or goal_node not in Graph:
            raise ValueError("start_node and goal_node must exist in Graph.")
        if start_node == goal_node:
            raise ValueError("start_node and goal_node must be different.")
        if risk_max is not None and risk_max < 0:
            raise ValueError("risk_max must be non-negative if provided.")

        self.G = Graph
        # Snapshot edges with data once to avoid re-iterating NetworkX views
        self._edges: List[Tuple[Hashable, Hashable, Dict[str, Any]]] = list(Graph.edges(data=True))

        self.start = start_node
        self.goal = goal_node
        self.edge_cost_key = edge_cost_key

        self.risk_max = risk_max
        self.risk = risk_matrix or {}  # default risk 0 for missing keys

        self.gurobi_env_params = gurobi_env_params or {}
        self.model_name = model_name
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.output = output

        self.model: Optional[gp.Model] = None
        self.x: Dict[Tuple[Hashable, Hashable], gp.Var] = {}
        self._built = False

    # --------------------------- Build model ---------------------------

    def _build(self) -> None:
        """Create variables, objective, and constraints."""
        with gp.Env(params=self.gurobi_env_params) as env:
            m = gp.Model(env=env, name=self.model_name)

            # Optional solver controls
            if self.time_limit is not None:
                m.Params.TimeLimit = self.time_limit
            if self.mip_gap is not None:
                m.Params.MIPGap = self.mip_gap
            m.Params.OutputFlag = 1 if self.output else 0

            # Decision variables: one binary per directed edge
            for i, j, _ in self._edges:
                self.x[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f"x[{i}->{j}]")

            # Read costs from edge attribute
            try:
                edge_cost = {(i, j): attr[self.edge_cost_key] for i, j, attr in self._edges}
            except KeyError as e:
                missing_key = str(e).strip("'")
                raise KeyError(
                    f"Edge attribute '{missing_key}' not found on all edges. "
                    f"Set 'edge_cost_key' correctly or populate it on the graph."
                )

            # Objective: minimize total cost
            m.setObjective(quicksum(self.x[e] * edge_cost[e] for e in self.x), GRB.MINIMIZE)

            # Flow-balance constraints
            nodes = list(self.G.nodes)
            for v in nodes:
                out_vars = [self.x[(v, j)] for v2, j, _ in self._edges if v2 == v]
                in_vars  = [self.x[(i, v)] for i, v2, _ in self._edges if v2 == v]
                supply = 1 if v == self.start else (-1 if v == self.goal else 0)
                m.addConstr(quicksum(out_vars) - quicksum(in_vars) == supply,
                            name=f"flow[{getattr(v, 'id', v)}]")

            # Optional risk budget
            if self.risk_max is not None:
                m.addConstr(
                    quicksum(self.x[e] * float(self.risk.get(e, 0.0)) for e in self.x) <= float(self.risk_max),
                    name="risk_budget"
                )

            self.model = m
            self._built = True

    # ------------------------------ Solve ------------------------------

    def solve(self) -> Optional[List[Tuple[Hashable, Hashable]]]:
        """
        Build (if needed) and solve the model.
        Returns the chosen path as a list of edges [(i,j), ...] or None if no optimal solution.
        """
        if not self._built:
            self._build()

        m = self.model
        assert m is not None

        m.optimize()

        if m.Status == GRB.OPTIMAL:
            return self._selected_edges()
        elif m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            # Compute IIS to diagnose infeasibility
            print("Model infeasible. Computing IIS...")
            try:
                m.computeIIS()
                m.write("csp_infeasible.ilp")
                print("IIS written to csp_infeasible.ilp")
                for c in m.getConstrs():
                    if c.IISConstr:
                        print(f"Infeasible: {c.ConstrName}")
            except gp.GurobiError:
                print("Failed to compute IIS.")
            return None
        else:
            print(f"No optimal solution (status={m.Status}).")
            return None

    # ---------------------------- Utilities ----------------------------

    def _selected_edges(self) -> List[Tuple[Hashable, Hashable]]:
        """Edges with x_ij ≈ 1."""
        return [(i, j) for (i, j), var in self.x.items() if var.X > 0.5]

    def solution_path_nodes(self) -> Optional[List[Hashable]]:
        """
        Trace the node sequence from start to goal using selected edges.
        Returns None if no solution or the path can't be traced (e.g., cycle/branching).
        """
        sel = self._selected_edges()
        if not sel:
            return None

        next_map: Dict[Hashable, Hashable] = {}
        indeg: Dict[Hashable, int] = {}
        outdeg: Dict[Hashable, int] = {}

        for i, j in sel:
            next_map[i] = j if i not in next_map else next_map[i]  # keep first, tolerate degeneracy
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
            cur = next_map[cur]
            if cur in visited:
                return None  # cycle detected
            visited.add(cur)
            path.append(cur)
            if cur == self.goal:
                return path
        return None
