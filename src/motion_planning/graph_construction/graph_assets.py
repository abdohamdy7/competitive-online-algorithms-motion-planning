import math
import carla
import numpy as np
from math import *
import copy
import networkx as nx
import pickle
import sys
import os
import re
import csv


# # Add the directory containing `path_optimization` to the Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# sys.path.append(project_root)

from motion_planning.graph_construction.path_optimization import *

class Node:
    def __init__(self, id, localFrame_state=None, globalFrame_state=None, desired_velocity=None, parent=None, ego_state=None, layer=None):
        self.id = id  # Unique identifier for the node (e.g., a name or numeric ID)
        self.localFrame_state = localFrame_state  # Optional: Data associated with this node
        self.globalFrame_state = globalFrame_state  # Optional: Data associated with this node
        self.desired_velocity = desired_velocity
        self.parent = parent
        self.ego_state=ego_state
        self.adjacent = {}  # Adjacent nodes and the cost of edge to them: {node: cost}
        self.layer = layer  # Optional: Layer or category of the node

    def add_neighbor(self, neighbor, weight=0):
        """Add a connection from this node to another."""
        self.adjacent[neighbor] = weight

class Edge:
    def __init__(self,id,  edge_path=None, edge_start_node=None, edge_end_node=None, 
                 edge_cost=None,edge_distance_cost=None,edge_time_cost_list=None, edge_lateral_cost=None,
                 edge_time_lateral_cost_list=None,
                   edge_risk=0.0):
        self.id = id
        self.edge_path = edge_path
        self.edge_start_node = edge_start_node
        self.edge_end_node = edge_end_node
        self.edge_cost = edge_cost
        self.edge_distance_cost = edge_distance_cost
        self.edge_time_cost_list = edge_time_cost_list
        self.edge_lateral_cost = edge_lateral_cost
        self.edge_time_lateral_cost_list = edge_time_lateral_cost_list
        self.edge_risk = edge_risk

    
    def get_edge_cost(self):
        return self.edge_cost
    
    def get_edge_risk(self):
        return self.edge_risk
    
    def get_edge_path(self):
        return self.edge_path


def d_goal(node, goal):
    """Euclidean distance in Frenet (s, l) to goal."""
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)


def d_goal_alpha(node, goal, alpha=10):
    return np.sqrt((node[0] - goal[0])**2 + alpha* ((node[1] - goal[1])**2))


def progress_utility_frenet(node_i, node_j, goal):
    """U_p(v_i, v_{i+1}) = d_goal(v_i) - d_goal(v_{i+1})"""
    return d_goal(node_i, goal) - d_goal(node_j, goal)


def progress_utility_frenet_alpha(node_i, node_j, goal):
    """U_p(v_i, v_{i+1}) = d_goal(v_i) - d_goal(v_{i+1})"""
    distance_prg = d_goal_alpha(node_i, goal) - d_goal_alpha(node_j, goal)
    prog_lateral_penalty = distance_prg-node_j[1]
    return prog_lateral_penalty


def generate_goal_indices(wps, n_steps):
    # Calculate goal indices at every n_steps interval
    goal_indices = [i for i in range(n_steps, len(wps), n_steps)]
    
    # If the list length is not a multiple of n_steps, add the last index
    if len(wps) % n_steps > 0:
        goal_indices[-1]= len(wps) - 1
            # goal_indices.append(len(wps) - 1)
        
    return goal_indices

# Find euclidean distance between two points
def euclideanDist(p1_x, p1_y, p2_x, p2_y):
    dist = math.sqrt((p1_x - p2_x)**2+(p1_y - p2_y)**2)
    return dist

def global2local(waypoint_global, ego_state):
    waypoint_local_x = (waypoint_global[0] - ego_state[0])*math.cos(ego_state[2]) + (waypoint_global[1] - ego_state[1])*math.sin(ego_state[2])
    waypoint_local_y = -(waypoint_global[0] - ego_state[0])*math.sin(ego_state[2]) + (waypoint_global[1] - ego_state[1])*math.cos(ego_state[2])
    return waypoint_local_x, waypoint_local_y


def get_nodes_state_set( goal_index, goal_state, waypoints, ego_state, num_paths, path_offset):
    

    """Gets the goal states given a goal position.
    
    Gets the goal states given a goal position. The states 

    args:
        goal_index: Goal index for the vehicle to reach
            i.e. waypoints[goal_index] gives the goal waypoint
        goal_state: Goal state for the vehicle to reach (global frame)
            format: [x_goal, y_goal, v_goal], in units [m, m, m/s]
        waypoints: current waypoints to track. length and speed in m and m/s.
            (includes speed to track at each x,y location.) (global frame)
            format: [[x0, y0, v0],
                        [x1, y1, v1],
                        ...
                        [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle, in the global frame.
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
    returns:
        goal_state_set: Set of goal states (offsetted laterally from one
            another) to be used by the local planner to plan multiple
            proposal paths. This goal state set is in the vehicle frame.
            format: [[x0, y0, t0, v0],
                        [x1, y1, t1, v1],
                        ...
                        [xm, ym, tm, vm]]
            , where m is the total number of goal states
                [x, y, t] are the position and yaw values at each goal
                v is the goal speed at the goal point.
                all units are in m, m/s and radians
    """
    # Compute the final heading based on the next index.
    # If the goal index is the last in the set of waypoints, use
    # the previous index instead.
    # To do this, compute the delta_x and delta_y values between
    # consecutive waypoints, then use the np.arctan2() function.
    # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
    # ------------------------------------------------------------------
    if goal_index == len(waypoints) - 1:
        delta_x = waypoints[goal_index][0] - waypoints[goal_index-1][0]
        delta_y = waypoints[goal_index][1] - waypoints[goal_index-1][1]
    else:
        delta_x = waypoints[goal_index+1][0] - waypoints[goal_index][0]
        delta_y = waypoints[goal_index+1][1] - waypoints[goal_index][1]
    
    heading = np.arctan2(delta_y, delta_x)
    # ------------------------------------------------------------------

    # Compute the center goal state in the local frame using 
    # the ego state. The following code will transform the input
    # goal state to the ego vehicle's local frame.
    # The goal state will be of the form (x, y, t, v).
    goal_state_local = copy.copy(goal_state)

    # Translate so the ego state is at the origin in the new frame.
    # This is done by subtracting the ego_state from the goal_state_local.
    # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
    # ------------------------------------------------------------------
    goal_state_local[0] -= ego_state[0]
    goal_state_local[1] -= ego_state[1]
    # ------------------------------------------------------------------

    # Rotate such that the ego state has zero heading in the new frame.
    # Recall that the general rotation matrix is [cos(theta) -sin(theta)
    #                                             sin(theta)  cos(theta)]
    # and that we are rotating by -ego_state[2] to ensure the ego vehicle's
    # current yaw corresponds to theta = 0 in the new local frame.
    # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
    # ------------------------------------------------------------------
    goal_x = goal_state_local[0]*cos(-ego_state[2]) - goal_state_local[1]*sin(-ego_state[2])
    goal_y = goal_state_local[0]*sin(-ego_state[2]) + goal_state_local[1]*cos(-ego_state[2])
    # ------------------------------------------------------------------

    # Compute the goal yaw in the local frame by subtracting off the 
    # current ego yaw from the heading variable.
    # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
    # ------------------------------------------------------------------
    goal_t = heading - ego_state[2]
    # ------------------------------------------------------------------

    # Velocity is preserved after the transformation.
    goal_v = goal_state[2]

    # Keep the goal heading within [-pi, pi] so the optimizer behaves well.
    if goal_t > pi:
        goal_t -= 2*pi
    elif goal_t < -pi:
        goal_t += 2*pi

    # Compute and apply the offset for each path such that
    # all of the paths have the same heading of the goal state, 
    # but are laterally offset with respect to the goal heading.
    goal_state_set = []
    for i in range(num_paths):
        # Compute offsets that span the number of paths set for the local
        # planner. Each offset goal will be used to generate a potential
        # path to be considered by the local planner.
        offset = (i - num_paths // 2) * path_offset

        # Compute the projection of the lateral offset along the x
        # and y axis. To do this, multiply the offset by cos(goal_theta + pi/2)
        # and sin(goal_theta + pi/2), respectively.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        x_offset = offset*cos(goal_t + pi/2)
        y_offset = offset*sin(goal_t + pi/2)
        # ------------------------------------------------------------------

        goal_state_set.append([goal_x + x_offset, 
                                goal_y + y_offset, 
                                goal_t, 
                                goal_v])
        
    return goal_state_set  

def get_straight_node( goal_state_set, dl=5.0):
    new_goals_set=[]
    for i, goal in enumerate( goal_state_set):
        new_goal=list.copy(goal)
        new_goal[0] = goal[0]+dl
        new_goals_set.append(new_goal)
        # print(i, goal, new_goal)
    return new_goals_set


# Plans the path set using polynomial spiral optimization to
# each of the goal states.    
def generate_paths( goal_state_set):
    po_obj = PathOptimizer()

    """Plans the path set using the polynomial spiral optimization.

    Plans the path set using polynomial spiral optimization to each of the
    goal states.

    args:
        goal_state_set: Set of goal states (offsetted laterally from one
            another) to be used by the local planner to plan multiple
            proposal paths. These goals are with respect to the vehicle
            frame.
            format: [[x0, y0, t0, v0],
                        [x1, y1, t1, v1],
                        ...
                        [xm, ym, tm, vm]]
            , where m is the total number of goal states
                [x, y, t] are the position and yaw values at each goal
                v is the goal speed at the goal point.
                all units are in m, m/s and radians
    returns:
        paths: A list of optimized spiral paths which satisfies the set of 
            goal states. A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    x_points: List of x values (m) along the spiral
                    y_points: List of y values (m) along the spiral
                    t_points: List of yaw values (rad) along the spiral
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
            Note that this path is in the vehicle frame, since the
            optimize_spiral function assumes this to be the case.
        path_validity: List of booleans classifying whether a path is valid
            (true) or not (false) for the local planner to traverse. Each ith
            path_validity corresponds to the ith path in the path list.
    """
    paths         = []
    path_validity = []
    for goal_state in goal_state_set:
        path = po_obj.optimize_spiral(goal_state[0], goal_state[1], goal_state[2])
        norm = np.linalg.norm([path[0][-1] - goal_state[0], 
                            path[1][-1] - goal_state[1], 
                            path[2][-1] - goal_state[2]])
        if  norm> 0.1:
            
            # print("norm: ",norm)
            path_validity.append(False)
        else:
            paths.append(path)
            path_validity.append(True)

    return paths, path_validity

import numpy as np

def route_filtering(waypoints, distance_threshold=0.5):
    """
    Filters out waypoints that are too close to the previous one.
    
    Parameters:
        waypoints (np.ndarray): Array of shape (N, 2) with [x, y] waypoints.
        distance_threshold (float): Minimum allowed distance between consecutive waypoints.
        
    Returns:
        np.ndarray: Filtered list of waypoints.
    """
    if len(waypoints) == 0:
        return waypoints

    # Start with the first waypoint
    filtered = [waypoints[0]]

    for wp in waypoints[1:]:
        if np.linalg.norm(wp - filtered[-1]) > distance_threshold:
            filtered.append(wp)

    return np.array(filtered)

def transform_points_to_global_frame(nodes_ego_frame, ego_state_global_frame):
    tranformed_nodes=list.copy(nodes_ego_frame)

    for i, node in enumerate(nodes_ego_frame):

        x_transformed = ego_state_global_frame[0] + node[0]*cos(ego_state_global_frame[2]) - \
                                                node[1]*sin(ego_state_global_frame[2])
        y_transformed = ego_state_global_frame[1] + node[0]*sin(ego_state_global_frame[2]) + \
                                                node[1]*cos(ego_state_global_frame[2])

        t_transformed = node[2] + ego_state_global_frame[2]
        tranformed_nodes[i][0] = x_transformed
        tranformed_nodes[i][1] = y_transformed
        tranformed_nodes[i][2] = t_transformed

    return tranformed_nodes

# def transform_edges(self):
def transform_paths(paths, ego_state):
    """ Converts the to the global coordinate frame.

    Converts the paths from the local (vehicle) coordinate frame to the
    global coordinate frame.

    args:
        paths: A list of paths in the local (vehicle) frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    , x_points: List of x values (m)
                    , y_points: List of y values (m)
                    , t_points: List of yaw values (rad)
                Example of accessing the ith path, jth point's t value:
                    paths[i][2][j]
        ego_state: ego state vector for the vehicle, in the global frame.
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)
    returns:
        transformed_paths: A list of transformed paths in the global frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]:
                    , x_points: List of x values (m)
                    , y_points: List of y values (m)
                    , t_points: List of yaw values (rad)
                Example of accessing the ith transformed path, jth point's 
                y value:
                    paths[i][1][j]
    """
    transformed_paths = []
    for path in paths:
        x_transformed = []
        y_transformed = []
        t_transformed = []

        for i in range(len(path[0])):
            x_transformed.append(ego_state[0] + path[0][i]*cos(ego_state[2]) - \
                                                path[1][i]*sin(ego_state[2]))
            y_transformed.append(ego_state[1] + path[0][i]*sin(ego_state[2]) + \
                                                path[1][i]*cos(ego_state[2]))
            t_transformed.append(path[2][i] + ego_state[2])

        transformed_paths.append([x_transformed, y_transformed, t_transformed])

    return transformed_paths


def get_bbox_corners(x, y, yaw, length, width):
    """
    Returns the (x, y) coordinates of the 4 corners of the vehicle bounding box.
    yaw in radians, length/width in meters
    """
    # Half dimensions
    l = length / 2
    w = width / 2

    # Four corners relative to center (x, y)
    # Front left, front right, rear right, rear left
    corners = [
        (+l, +w),
        (+l, -w),
        (-l, -w),
        (-l, +w)
    ]

    # Rotate and translate
    result = []
    for dx, dy in corners:
        cx = x + (dx * math.cos(yaw) - dy * math.sin(yaw))
        cy = y + (dx * math.sin(yaw) + dy * math.cos(yaw))
        result.append((cx, cy))
    return result

def prone_nodes_hdmap_robust(node_list, world, vehicle_length=1.0, vehicle_width=1.0):
    proned_list = []
    indexes = []
    if world is None:
        raise ValueError("World not set. Please set the world before calling this function.")
    for i, node in enumerate(node_list):
        # If node contains heading, unpack; otherwise, assume heading=0
        if len(node) == 3:
            x, y, yaw = node
        else:
            x, y = node[:2]
            yaw = 0.0  # You may want to improve this!

        # Get corners of bounding box
        corners = get_bbox_corners(x, y, yaw, vehicle_length, vehicle_width)

        all_corners_drivable = True
        for cx, cy in corners:
            location = carla.Location(x=cx, y=cy, z=0.0)
            waypoint = world.get_map().get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving)
            if not waypoint or (waypoint.lane_type != carla.LaneType.Driving):
                all_corners_drivable = False
                break
        if all_corners_drivable:
            proned_list.append(node)
            indexes.append(i)
    return proned_list, indexes

def prone_nodes_hdmap(node_list, world):
    proned_list=[]
    indexes=[]
    if world is None:
        raise ValueError("World not set. Please set the world before calling this function.")
    for i, node in enumerate(node_list):
        location = carla.Location(x=node[0], y=node[1], z=0.0)
        # print(location)
        # waypoint = world.get_map().get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving)
        
        waypoint = world.get_map().get_waypoint(location, project_to_road=False)

        # print(waypoint)
        if waypoint:
            print("waypoint.lane_type",waypoint.lane_type)
            if waypoint.lane_type == None or waypoint.lane_type == carla.LaneType.Driving:
                proned_list.append(node)
                indexes.append(i)

    return proned_list,indexes

def is_valid_carla_wp(waypoint, world):
    """
    Check if a waypoint is valid in the CARLA world.
    A waypoint is considered valid if it is not None and its lane type is Driving.
    """
    location = carla.Location(x=waypoint[0], y=waypoint[1], z=0.0)
        # print(location)
        # waypoint = world.get_map().get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving)    
    waypoint = world.get_map().get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving)
    if waypoint is None:
        return False           
    
    #  Check if the waypoint's lane type is Driving
    return waypoint.lane_type == carla.LaneType.Driving

def compute_edge_cost(edge_path):
    # compute edge distance cost : transformed path cost
    x_vec, y_vec, t_vec = edge_path
    e_cost = 0
    for wp_idx, wp in enumerate(zip(x_vec, y_vec)):
        # print("wp_idx", wp_idx)
        # print("x_t", wp[0])
        # print("y_t", wp[1])
        x_t=wp[0]
        y_t=wp[1]

        if wp_idx ==0:
            x_tt=x_t
            y_tt=y_t
            continue

        e_cost+= euclideanDist(x_tt,y_tt,x_t,y_t)
        
        x_tt=x_t
        y_tt=y_t

    return e_cost


def compute_edge_distance_cost(edge_path):
    # compute edge distance cost : transformed path cost
    x_vec, y_vec, t_vec = edge_path
    e_cost = 0
    for wp_idx, wp in enumerate(zip(x_vec, y_vec)):
        # print("wp_idx", wp_idx)
        # print("x_t", wp[0])
        # print("y_t", wp[1])
        x_t=wp[0]
        y_t=wp[1]

        if wp_idx ==0:
            x_tt=x_t
            y_tt=y_t
            continue

        e_cost+= euclideanDist(x_tt,y_tt,x_t,y_t)
        
        x_tt=x_t
        y_tt=y_t

    return e_cost

def compute_edge_time_cost_list(edge_distance_cost, speeds_set):
    # compute edge time cost list for different speeds
    # edge_distance_cost: in meters
    # speeds_set: in km/hr
    # edge_time_cost_list: in seconds

    e_time_cost_list = []
    for speed in speeds_set:
        if speed <=0:
            e_time_cost_list.append(float('inf'))
        else:
            e_time = edge_distance_cost / (speed/3.6)  # speed converted to m/s
            e_time_cost_list.append(e_time)    # in seconds
    return e_time_cost_list



def compute_edge_lateral_cost(i, j):
    i_nums =  re.findall(r'\d+', i.id)
    j_nums =  re.findall(r'\d+', j.id)
    extracted_numbers_i = list(map(int, i_nums))
    extracted_numbers_j = list(map(int, j_nums))

    dev_val = abs(extracted_numbers_i[1]-extracted_numbers_j[1])

    return dev_val


def compute_edge_time_lateral_cost_list(edge_lateral_cost, edge_time_cost_list, lateral_penalty_weight=10):

    # compute edge time lateral cost list for different speeds
    # edge_lateral_cost: lateral deviation cost
    # edge_time_cost_list: in seconds
    # edge_time_lateral_cost_list: in cost units
    e_time_lateral_cost_list = []
    for e_time in edge_time_cost_list:
        e_time_lateral_cost = e_time  + lateral_penalty_weight * edge_lateral_cost
        e_time_lateral_cost_list.append(e_time_lateral_cost) # in cost units
    return e_time_lateral_cost_list

    
class GRAPH_FILE_HANDLER:
    def __init__(self):
        pass
    def save_graph(self, graph, save_dir, scenario):
        fname = f"{save_dir}/graph_{scenario}.gpickle"
        with open(fname, "wb") as f:
            pickle.dump(graph, f)
        return fname
    
    
    def load_graph(self, load_dir, scenario):
        fname = f"{load_dir}/graph_{scenario}.gpickle"
        with open(fname, "rb") as f:
            graph = pickle.load(f)
        return graph


    def save_reference_path_csv(self, path, save_dir, scenario):
        """Saves a path (e.g., a list of [x, y] points) to a CSV file."""
        fname = f"{save_dir}/reference_path_{scenario}.csv"
        
        with open(fname, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Optional: Write a header if you want
            # writer.writerow(['x_coord', 'y_coord']) 
            writer.writerows(path) # path should be an iterable of rows
        
        return fname

    def load_reference_path_csv(self, load_dir, scenario):
        """Loads a path from a CSV file."""
        fname = f"{load_dir}/reference_path_{scenario}.csv"
        path = []
        
        with open(fname, mode='r', newline='') as f:
            reader = csv.reader(f)
            # Optional: Skip header if you added one
            # next(reader, None) 
            for row in reader:
                # Convert string data back to the appropriate type (e.g., float)
                try:
                    path.append([float(row[0]), float(row[1])])
                except ValueError as e:
                    print(f"Error converting row data: {row} - {e}")
                    continue
                    
        return path