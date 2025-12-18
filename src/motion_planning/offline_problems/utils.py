
import random
import math
BIG_M = 100

def find_start_and_goal_nodes(G, start_node_id='v_0_0'):
    start_node = next(node for node in G.nodes if node.id == start_node_id)
    goal_node = list(G.nodes)[-1]
    return start_node, goal_node

def get_random_decision_epochs(num_epochs, goal_indecies):
    print("goal indecies inside helper function: ", goal_indecies)
    if not goal_indecies:
        raise ValueError("Invalid goal indices provided.")

    last_layer_index = goal_indecies[-1]
    if last_layer_index < 0:
        raise ValueError("Invalid goal indices provided.")

    # Minimum number of epochs needed to ensure we can land within 7 layers of the goal
    min_epochs_needed = max(3, math.ceil(max(0, last_layer_index - 7) / 7) + 1)
    target_epochs = max(num_epochs, min_epochs_needed)
    target_count = target_epochs - 1  # exclude the start node
    max_decision_epochs = (last_layer_index - 1) // 3  # honor min spacing of 3 layers

    if target_count > max_decision_epochs:
        raise ValueError(
            f"Unable to fit {target_count} decision epochs with 3-7 layer spacing before reaching the goal layer."
        )

    def _sample(count):
        epochs = []
        current_layer = 0

        for _ in range(count):
            remaining_after = count - len(epochs) - 1
            min_step = max(3, last_layer_index - 7 - current_layer - 7 * remaining_after)
            max_step = min(7, last_layer_index - 1 - current_layer - 3 * remaining_after)

            if max_step < min_step:
                return None

            step = random.randint(min_step, max_step)
            current_layer += step
            epochs.append(current_layer)

        if epochs and last_layer_index - epochs[-1] <= 7:
            gap_to_goal = last_layer_index - epochs[-1]
            if gap_to_goal >= 3:
                return epochs
        return None

    max_attempts = 1000
    for _ in range(max_attempts):
        sampled = _sample(target_count)
        if sampled is not None:
            return sampled

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
