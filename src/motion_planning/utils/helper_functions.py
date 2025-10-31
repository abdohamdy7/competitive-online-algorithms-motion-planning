import math
import random
import numpy as np


def get_angle(wp1, wp2):
    x1= wp1[ 0]
    y1= wp1[ 1]
    x2= wp2[ 0]
    y2= wp2[ 1]
    angle_radians = math.atan2(y2 - y1, x2 - x1)
    
    # Calculate the angle in radians between the line from (x1, y1) to (x2, y2) and the x-axis
    angle_radians = math.atan2(y2 - y1, x2 - x1)
    
    # Convert the angle from radians to degrees
    # angle_degrees = math.degrees(angle_radians)
    
    return angle_radians

def define_goal_points_for_decision_epochs(G, decision_epochs, reference_path=None):

    # define the row index of the reference path at each decision epoch 
    pass
def find_start_and_goal_nodes(G, start_node_id='v_0_0'):
    start_node = next(node for node in G.nodes if node.id == start_node_id)
    goal_node = list(G.nodes)[-1]
    return start_node, goal_node
def select_random_edges(graph, num_edges):
    edges = list(graph.edges())
    if num_edges > len(edges):
        raise ValueError("Number of edges to select exceeds the number of edges in the graph")
    random_edges = random.sample(edges, num_edges)
    return random_edges

def generate_random_risks(graph, num_edges, risk_max):
    random_edges = select_random_edges(graph, num_edges)
    PHI_ij = {}
    for (u, v) in graph.edges:
        PHI_ij[u, v] = np.random.uniform(0.1, risk_max) if (u, v) in random_edges else 0.001
    return PHI_ij

def generate_zero_risks(graph):
    PHI_ij = {}
    for (u, v) in graph.edges:
        PHI_ij[u, v] = 0.001
    
    return PHI_ij


def get_random_decision_epochs(num_epochs, goal_indecies):
    print("goal indecies inside helper function: ", goal_indecies)
    possible_epochs = [i for i in range(2, len(goal_indecies)-2)]
    if num_epochs > len(possible_epochs):
        raise ValueError("Number of decision epochs exceeds the available epochs.")
    
    selected_epochs = random.sample(possible_epochs, num_epochs-1)
    selected_epochs.sort()
    
    # decision_epochs = [goal_indecies[i] for i in selected_epochs]
    return selected_epochs

def get_closeset_node(G, layer_nodes, carla_route):
    # this function takes layer nodes, and carla waypoints and find the layer node that is closest to any of the carla waypoints.


    min_distance = float('inf')
    closest_node = None
    
    for node in layer_nodes:
        node_x, node_y = node.globalFrame_state[0], node.globalFrame_state[1]
        for wp in carla_route:
            wp_x, wp_y = wp[0], wp[1]
            distance = math.sqrt((node_x - wp_x) ** 2 + (node_y - wp_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_node = node
    return closest_node 

def get_decision_nodes(G, decision_epochs, carla_route):
    # this function takes the decision epochs and find the corresponding decision nodes in the graph G.
    start_node = next(node for node in G.nodes if node.id == 'v_0_0')

    decision_nodes = []
    decision_nodes.append(start_node)
    layer_nodes = []
    
    for epoch in decision_epochs:
        for node in G.nodes:
            if node.layer == epoch:
                layer_nodes.append(node)
                # find the node that has the same coordinates as the one in carla_route
        closest_node = get_closeset_node(G, layer_nodes, carla_route)
        decision_nodes.append(closest_node)
        layer_nodes = [] # reset for the next epoch
    return decision_nodes

def get_start_goal_node_pairs(G,decision_nodes):
    start_goal_pairs = []
    previous_node = None
    for node in decision_nodes:
        if previous_node is not None:
            start_goal_pairs.append((previous_node, node))
        previous_node = node
    last_node = list(G.nodes)[-1]
    start_goal_pairs.append((previous_node, last_node))
    return start_goal_pairs

from typing import Sequence, Optional, Any, Dict, List
import gurobipy as gp
from gurobipy import GRB


def solve_offline_mckp_gurobi(
    utilities: Sequence[Sequence[float]],
    risks: Sequence[Sequence[float]],
    capacity: float,
    *,
    ids: Optional[Sequence[Sequence[Any]]] = None,
    mip_gap: Optional[float] = None,
    time_limit: Optional[float] = None,
    threads: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    # NEW: licensing / environment hooks
    env_options: Optional[Dict[str, Any]] = None,   # e.g., your WLS dict
    external_env: Optional[gp.Env] = None,          # reuse an env if you already opened one
    model_name: str = "offline_MCKP",
) -> Dict[str, Any]:
    """
    Offline MCKP with optional Gurobi WLS licensing and environment reuse.

    If `external_env` is given, the model is built on it (and we DO NOT close that env).
    Otherwise, we create a private environment; if `env_options` is provided, it is passed
    to `gp.Env(params=env_options)` (works with WLSACCESSID/WLSSECRET/LICENSEID).
    """

    # --- basic validation ---
    T = len(utilities)
    if T == 0:
        raise ValueError("`utilities` must contain at least one group.")
    if len(risks) != T:
        raise ValueError("`utilities` and `risks` must have the same number of groups.")
    if ids is not None and len(ids) != T:
        raise ValueError("`ids` must have the same shape as `utilities` if provided.")

    N_t: List[int] = []
    for t in range(T):
        if len(risks[t]) != len(utilities[t]):
            raise ValueError(f"Group {t}: utilities and risks lengths differ.")
        if ids is not None and len(ids[t]) != len(utilities[t]):
            raise ValueError(f"Group {t}: ids length does not match utilities.")
        if len(utilities[t]) == 0:
            raise ValueError(f"Group {t} is empty; each group must contain ≥1 item.")
        N_t.append(len(utilities[t]))

    # Fast infeasibility check
    min_risk_sum = sum(min(risks[t][n] for n in range(N_t[t])) for t in range(T))
    if min_risk_sum > capacity + 1e-12:
        return {
            "status": GRB.INFEASIBLE,
            "status_str": "INFEASIBLE (pre-check)",
            "obj_value": None,
            "risk_used": None,
            "selected_indices": None,
            "selected_ids": None,
            "y_values": {},
            "gap": None,
        }

    def _build_and_solve(_model: gp.Model) -> Dict[str, Any]:
        # params
        _model.Params.OutputFlag = 1 if verbose else 0
        if mip_gap is not None:
            _model.Params.MIPGap = float(mip_gap)
        if time_limit is not None:
            _model.Params.TimeLimit = float(time_limit)
        if threads is not None:
            _model.Params.Threads = int(threads)
        if seed is not None:
            _model.Params.Seed = int(seed)

        # variables
        y = {(t, n): _model.addVar(vtype=GRB.BINARY, name=f"y[{t},{n}]")
             for t in range(T) for n in range(N_t[t])}

        # objective: maximize total utility
        _model.setObjective(
            gp.quicksum(utilities[t][n] * y[(t, n)]
                        for t in range(T) for n in range(N_t[t])),
            GRB.MAXIMIZE,
        )

        # capacity
        _model.addConstr(
            gp.quicksum(risks[t][n] * y[(t, n)]
                        for t in range(T) for n in range(N_t[t]))
            <= capacity, name="risk_budget"
        )

        # exactly one per group
        for t in range(T):
            _model.addConstr(gp.quicksum(y[(t, n)] for n in range(N_t[t])) == 1,
                             name=f"one_of_group_{t}")

        # optimize
        _model.optimize()

        status = _model.Status
        status_str = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
        }.get(status, f"STATUS_{status}")

        has_solution = (status == GRB.OPTIMAL) or (_model.SolCount and _model.SolCount > 0)
        y_vals: Dict[tuple, float] = {}
        selected_indices: Optional[List[Optional[int]]] = None
        selected_ids: Optional[List[Any]] = None
        obj_value = None
        risk_used = None
        gap = None

        if has_solution:
            for key, var in y.items():
                y_vals[key] = var.X

            selected_indices = []
            selected_ids = [] if ids is not None else None
            for t in range(T):
                picked = None
                for n in range(N_t[t]):
                    if y_vals[(t, n)] > 0.5:
                        picked = n
                        break
                selected_indices.append(picked)
                if ids is not None:
                    selected_ids.append(ids[t][picked] if picked is not None else None)

            obj_value = float(_model.objVal)
            risk_used = (sum(risks[t][selected_indices[t]] for t in range(T))
                         if selected_indices and None not in selected_indices else None)
            try:
                gap = float(_model.MIPGap)
            except gp.GurobiError:
                gap = None

        return {
            "status": status,
            "status_str": status_str,
            "obj_value": obj_value,
            "risk_used": risk_used,
            "selected_indices": selected_indices,
            "selected_ids": selected_ids,
            "y_values": y_vals,
            "gap": gap,
        }

    # --- Environment handling ---
    if external_env is not None:
        # Use the caller's environment; do not close it here.
        model = gp.Model(env=external_env, name=model_name)
        return _build_and_solve(model)

    # Create a private env (WLS or otherwise) and close it automatically.
    # If you have WLS fields (WLSACCESSID, WLSSECRET, LICENSEID), pass them via `env_options`.
    try:
        if env_options is None:
            with gp.Env() as env:
                model = gp.Model(env=env, name=model_name)
                return _build_and_solve(model)
        else:
            with gp.Env(params=env_options) as env:
                model = gp.Model(env=env, name=model_name)
                return _build_and_solve(model)
    except gp.GurobiError as e:
        return {
            "status": -1,
            "status_str": f"Gurobi error {e.errno}: {e.message}",
            "obj_value": None,
            "risk_used": None,
            "selected_indices": None,
            "selected_ids": None,
            "y_values": {},
            "gap": None,
        }


