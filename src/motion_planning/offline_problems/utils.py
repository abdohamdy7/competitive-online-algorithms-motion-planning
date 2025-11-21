
import random
import math
BIG_M = 1000

def find_start_and_goal_nodes(G, start_node_id='v_0_0'):
    start_node = next(node for node in G.nodes if node.id == start_node_id)
    goal_node = list(G.nodes)[-1]
    return start_node, goal_node

def get_random_decision_epochs(num_epochs, goal_indecies):
    print("goal indecies inside helper function: ", goal_indecies)
    target_count = max(0, num_epochs - 1)

    if target_count == 0:
        return []

    last_layer_index = goal_indecies[-1]
    if last_layer_index < 0:
        raise ValueError("Invalid goal indices provided.")

    max_attempts = 5000
    for _ in range(max_attempts):
        epochs = []
        current_layer = 0

        while len(epochs) < target_count:
            step = random.randint(3, 7)
            candidate_layer = current_layer + step
            if candidate_layer >= last_layer_index:
                break
            epochs.append(candidate_layer)
            current_layer = candidate_layer

        if len(epochs) == target_count:
            tail_gap = last_layer_index - epochs[-1]
            if tail_gap <= 7:
                return epochs

    raise ValueError(
        f"Unable to sample {target_count} decision epochs with 3-7 layer spacing before reaching the goal layer."
    )

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

def get_closeset_node(G, layer_nodes, carla_route):
    # this function takes layer nodes, and carla waypoints and find the layer node that is closest to any of the carla waypoints.
    # carla route is a list of (x,y, yaw) tuples. for the reference path.

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
