
import time
import matplotlib.pyplot as plt

from motion_planning.graph_construction.graph_assets import *

from motion_planning.graph_construction.route_loading import *
import numpy as np
from motion_planning.utils.helper_functions import *
import copy


class GraphConstruction:

    def __init__(self, map_id, road_scenario, n_seperated_wps,num_paths, path_offset,scenario=None):
        self._n_seperated_wps = n_seperated_wps
        self._num_paths = num_paths
        self._path_offset = path_offset
        self._scenario = scenario
        self._sampling_resolution= 1
        self._speeds_set = [5, 10, 15, 20, 25, 30]  # km/hr
     
        if self._scenario == 'Roundabout':
            print("Scenario: ", self._scenario)
            self._num_paths = 10
            self._n_seperated_wps=5
            self._path_offset = 1.0
            self._sampling_resolution = 2
        elif self._scenario == 'Multiple Intersections':
            print("Scenario: ", self._scenario)
            self._num_paths = 4
            self._n_seperated_wps=10
            self._path_offset = 1.75
            self._sampling_resolution = 1

        elif self._scenario in  ['Highway1','Highway2','Highway3']:    
            print("Scenario: ", self._scenario)
            self._num_paths = 3
            self._n_seperated_wps=15
            self._path_offset = 3.0
            self._sampling_resolution = 1

        if (map_id, road_scenario) in routes_config:
            start, goal = routes_config[(map_id, road_scenario)]
            self._route, self._world = load_and_plan_route(map_id, road_scenario, start, goal, self._sampling_resolution)
        else:
            raise ValueError(f"Route for map '{map_id}' and scenario '{road_scenario}' not found in configuration.")

    def construct_graph_from_carla_route(self):
        self.V = []  # list of Nodes
        self.E = []  # list of Edges

        waypoints = [[wp.transform.location.x,wp.transform.location.y] for (wp,_) in self._route]
        
        waypoints = np.array(waypoints)
        # print("waypoints before: ", waypoints)
        waypoints = route_filtering(waypoints)
        # print("waypoints after: ", waypoints)
        self._ref_path = waypoints
        n_layers = len(waypoints) / self._n_seperated_wps
            
        ego_state_list=[]
        
        ego_theta = get_angle(waypoints[0], waypoints[1])

        start_state = [waypoints[0,0], waypoints[0,1], ego_theta, None]
        # print(start_state)
        # add start node
        layer=[]
        layer.append(Node(id="v_0_0", globalFrame_state=start_state, ego_state=start_state, layer=0))
        self.V.append(layer)
        
        final_nodes_global=[]
        final_nodes_local=[]
        goal_indecies_list = generate_goal_indices(waypoints, n_steps=self._n_seperated_wps)
        self._goal_indecies_list = goal_indecies_list
        # print("goal indecies: ", goal_indecies_list)
        prev_goal_index = 0
        
        for goal_index in goal_indecies_list:
            if goal_index == goal_indecies_list[-1]: # to avoid adding last node
                continue

            goal_state = [waypoints[goal_index,0], waypoints[goal_index,1],None]

            ego_idx = prev_goal_index

            ego_theta = get_angle(waypoints[ego_idx], waypoints[ego_idx+1])
            print("waypoints: ",waypoints[ego_idx], waypoints[ego_idx+1])
            print("ego theta: ", ego_theta)

            ego_state = [waypoints[ego_idx,0], waypoints[ego_idx,1], ego_theta] # global frame
            # print("ego state: ", ego_state)
            ego_state_list.append(ego_state)
            nodes_state_set = get_nodes_state_set(goal_index, goal_state, waypoints, \
                                                ego_state, self._num_paths, self._path_offset)
            
            
            final_nodes_local.append(copy.deepcopy(nodes_state_set))

            _nodes_global = transform_points_to_global_frame((nodes_state_set), ego_state)
            
            final_nodes_global.append((_nodes_global))
            prev_goal_index = goal_index
        
        
        # proning the nodes based on carla hd map
        proned_final_nodes_global = []
        proned_final_nodes_local = []
        for e, final_node_global_layer in enumerate(final_nodes_global):

            proned_list,indexes = prone_nodes_hdmap_robust(final_node_global_layer,self._world)
            # proned_list,indexes = prone_nodes_hdmap(final_node_global_layer,self._world)
            proned_final_nodes_global.append(proned_list)
            proned_final_nodes_local.append([final_nodes_local[e][i]for i in indexes])
            # print("without prone nodes: ", final_node_global_layer)
            # print("proned nodes: ", proned_final_nodes_global)
            

        # construct layers of nodes in the graph
        layer_counter = 1
        for i, nodes in enumerate(proned_final_nodes_global):
            layer=[]
            if nodes == []:
                print("Warning: No valid nodes found in layer ", i)
                continue
            for j, node in enumerate(nodes):
                layer.append(Node(id="v_"+str(layer_counter)+str("_")+str(j+1),\
                                globalFrame_state=proned_final_nodes_global[i][j], localFrame_state=proned_final_nodes_local[i][j], desired_velocity=None, layer=layer_counter))
            layer_counter += 1
            if layer:
                self.V.append(layer)

        # add last point, goal destination
        layer=[]
        
        ego_theta = get_angle(waypoints[ego_idx], waypoints[ego_idx+1])
        print("ego theta: ", ego_theta)

        goal_state = [waypoints[-1,0], waypoints[-1,1],ego_theta,None]
        
        print ("V[-1]=",self.V[-1])
        layer.append(Node(id="v_"+str(layer_counter)+str("_")+str(1), 
                    globalFrame_state=goal_state, localFrame_state= self.V[-1][2].localFrame_state ,desired_velocity=None, parent=None, layer=layer_counter))

        print("last layer: " , layer[0].globalFrame_state)
        self.V.append(layer)

        # last filter: to make it consitent

        # print the numbder of nodes in each layer
        for i, layer in enumerate(self.V):
            print("layer ", i, " has ", len(layer), " nodes")
        # print("last point", V[-1][0].localFrame_state)
        # print("last 2nd point", V[-3][3].localFrame_state)
        # for n in V:
        #     for m in n:
        #     print(m.id, m.globalFrame_state)


        # ... code to construct nodes V ...
        
        # ... code to construct edges E ...

        
        for i in range(len(self.V)):

            # if i == 0: # first layer
            #     continue
            if i==len(self.V)-1:
                break

            # nodes in the current layer i
            # nodes in the next layer i+1
            for node_current_layer in self.V[i]:
                # print("plan from: " + str(node_current_layer.id) + " to: " + str([x.id for x in V[i+1] ]))
                # plan paths from V[i] to V[j+1]
                            # print([goal.localFrame_state for goal in V[i+1]])
                goals_local_frame=[]
                for goal_node in self.V[i+1]:
                    loc_wp_x, loc_wp_y = global2local(goal_node.globalFrame_state, node_current_layer.globalFrame_state)
                    # cur_wp = [node_current_layer.globalFrame_state[0], node_current_layer.globalFrame_state[1]]
                    # next_wp = [goal_node.globalFrame_state[0],goal_node.globalFrame_state[1]]
                    # ego_theta = get_angle(cur_wp, next_wp )

                    goals_local_frame.append([loc_wp_x, loc_wp_y,goal_node.localFrame_state[2]])

                # print("planning goals: ", goals_local_frame)
                # print("------")
                # print([x.localFrame_state for x in V[i+1]])
                    
                planned_paths, paths_validity  = generate_paths(goals_local_frame)
                # planned_paths, paths_validity  = generate_paths([x.localFrame_state for x in V[i+1]])
                print("planning done!: ", i)
                print(paths_validity)
                valid_paths = []
                # print("node cur layer->ego state: ", node_current_layer.globalFrame_state)
                for pp, planned_path in enumerate(planned_paths):
                    if paths_validity[pp] == False:
                        print("path not valid")
                    else:
                        valid_paths.append(planned_path)

                if valid_paths == []:
                    print("no valid paths")
                    continue
                transformed_paths = transform_paths(valid_paths, node_current_layer.globalFrame_state)
                # print("len of transformed paths: ", len(transformed_paths))
                # ## check the paths validity again
                # for p_id, global_path in enumerate(transformed_paths):
                #     for wp_idx in range(len(global_path[0])):
                #         wp =[global_path[0][wp_idx], global_path[1][wp_idx]]
                #         if not is_valid_carla_wp(wp, self._world):
                #             print("invalid waypoint: ", wp)
                #             # _ = transformed_paths.pop(p_id)
                #             del transformed_paths[p_id]
                #             break

                
                # print("len of transformed paths: ", len(transformed_paths))

                #prone the paths based on carla hd map
                # create edges for the graph data structure!
                edges_layer=[]
                # print("tr_paths len",len(transformed_paths[0]))
                # print("next layer len",len(self.V[i+1]))
                for k, next_node in enumerate(self.V[i+1]):
                    # print("k = ", k)
                    # print("len(transformed_paths)=",len(transformed_paths))

                    if k<=len(transformed_paths)-1:
                        e_cost = compute_edge_cost(transformed_paths[k])
                        e_d_cost = compute_edge_distance_cost(transformed_paths[k])
                        
                        e_t_cost_list = compute_edge_time_cost_list(e_d_cost, self._speeds_set)
                        e_lateral_cost = compute_edge_lateral_cost(node_current_layer, next_node)
                        e_time_lateral_cost_list = compute_edge_time_lateral_cost_list(e_lateral_cost, e_t_cost_list) # current penalty weight for lateral cost is 10

                        e = Edge(id="e("+str(node_current_layer.id)+","+str(next_node.id)+")", \
                        edge_start_node=node_current_layer, edge_end_node=next_node, edge_path=transformed_paths[k],
                        edge_cost=e_cost, edge_distance_cost=e_d_cost, edge_time_cost_list=e_t_cost_list, 
                        edge_lateral_cost=e_lateral_cost, edge_time_lateral_cost_list=e_time_lateral_cost_list)
                    
                        edges_layer.append(e)

                self.E.append(edges_layer)


         
        return self.V, self.E
    
    def get_current_world(self):
        return self._world
    
    def visualize_final_graph_both(self):
        # visualize the road
        # Plotting the polygon
        
        # road_polygon = get_road_polygon(left_boundaries, right_boundaries)
        # margin_road_polygon = get_margin_road_polygon(road_polygon, margin=-0.6)

        fig, ax = plt.subplots()
        # x, y = road_polygon.exterior.xy  # Extracting x and y coordinates from the polygon
        # ax.fill(x, y, alpha=0.5, fc='lightblue', ec='none', label='Road Area')


        # # Plotting the other polygon
        # x, y = margin_road_polygon.exterior.xy  # Extracting x and y coordinates from the polygon
        # ax.fill(x, y, alpha=0.3, fc='grey', ec='none', label='Margin Road Area')
        
        
        #  plot edges
        if self._scenario in ['Highway1', 'Highway2','Highway3']:
            z= 10.0
        else:
            z= 0.0


        for edge_layer in self.E:
            for edge in edge_layer:
                x,y,taw = edge.edge_path
                ax.plot(x,y, color='grey')
                # print("printing the edge path on the server side!!")
                for i in range(len(x) - 1):
                    start = carla.Location(x=x[i], y=y[i], z=z)
                    end = carla.Location(x=x[i+1], y=y[i+1], z=z)

                    self._world.debug.draw_line(start, end,
                                    thickness=0.1,
                                    color=carla.Color(0, 0, 0),  # grey
                                    life_time=0,  # 0 = persistent
                                    persistent_lines=True)
        # plot nodes
        done_flag= 0
        for i, layer_of_nodes in enumerate(self.V):    
            for j, node in enumerate(layer_of_nodes): # all other nodes
                if i==0: # start node
                    ax.scatter(node.globalFrame_state[0],node.globalFrame_state[1],\
                                color='red', label='Start vertex', s=16)
                    break
                
                
                if i==len(self.V)-1: # goal node
                    ax.scatter(node.globalFrame_state[0],node.globalFrame_state[1],\
                                color='green', label='Goal vertex', s=16)
                    break
                
                if i==1:
                    ax.scatter(node.globalFrame_state[0],node.globalFrame_state[1],\
                            color='blue', s=20, label = 'A vertex')
                
                if done_flag==0:
                    ax.scatter(node.globalFrame_state[0],node.globalFrame_state[1],\
                            color='blue', label='Vertex in the graph', s=16)
                    done_flag=1
                
                ax.plot(node.globalFrame_state[0],node.globalFrame_state[1],\
                            color='blue', marker = 'o', markersize=4)
                color = carla.Color(r=0, g=0, b=255)
                life_time = 120.0
                # Modify the point (e.g., x+2, y+2)
                waypoint_to_draw = carla.Location(x=node.globalFrame_state[0],
                                        y=node.globalFrame_state[1],
                                        z=z)

                # Optionally: check if the modified point is on a drivable lane
                # new_wp = world.get_map().get_waypoint(waypoint_to_draw, project_to_road=False, lane_type=carla.LaneType.Driving)
                # print("new wp: ", world)
                self._world.debug.draw_string(waypoint_to_draw, 'O',
                                        draw_shadow=False,
                                        color=color,
                                        life_time=life_time,
                                        persistent_lines=True)
        
        
                    
        
        
        
        # Setting the plot labels and title
        ax.set_title('Final Constructed Graph')
        ax.set_xlabel("Lateral distance (m)")
        ax.set_ylabel("Longitudinal distance (m)")
        ax.legend()
        # Save the figure as a PDF file
        # plt.savefig('constructed graph', format='pdf', bbox_inches='tight')

        plt.show()
        # time.sleep(3)  # Adjust as necessary for your system
    
    def visualize_final_graph(self):
        # visualize the road
        # Plotting the polygon
        
        # road_polygon = get_road_polygon(left_boundaries, right_boundaries)
        # margin_road_polygon = get_margin_road_polygon(road_polygon, margin=-0.6)

        fig, ax = plt.subplots()
        # x, y = road_polygon.exterior.xy  # Extracting x and y coordinates from the polygon
        # ax.fill(x, y, alpha=0.5, fc='lightblue', ec='none', label='Road Area')


        # # Plotting the other polygon
        # x, y = margin_road_polygon.exterior.xy  # Extracting x and y coordinates from the polygon
        # ax.fill(x, y, alpha=0.3, fc='grey', ec='none', label='Margin Road Area')
        
        
        #  plot edges
        if self._scenario in ['Highway1', 'Highway2','Highway3']:
            z= 10.0
        else:
            z= 0.0


        for edge_layer in self.E:
            for edge in edge_layer:
                x,y,taw = edge.edge_path
                # ax.plot(x,y, color='grey')
                # print("printing the edge path on the server side!!")
                for i in range(len(x) - 1):
                    start = carla.Location(x=x[i], y=y[i], z=z)
                    end = carla.Location(x=x[i+1], y=y[i+1], z=z)

                    self._world.debug.draw_line(start, end,
                                    thickness=0.1,
                                    color=carla.Color(0, 0, 0),  # grey
                                    life_time=0,  # 0 = persistent
                                    persistent_lines=True)
        # plot nodes
        done_flag= 0
        for i, layer_of_nodes in enumerate(self.V):    
            for j, node in enumerate(layer_of_nodes): # all other nodes
                if i==0: # start node
                    # ax.scatter(node.globalFrame_state[0],node.globalFrame_state[1],\
                    #             color='red', label='Start vertex', s=16)
                    break
                
                
                if i==len(self.V)-1: # goal node
                    # ax.scatter(node.globalFrame_state[0],node.globalFrame_state[1],\
                    #             color='green', label='Goal vertex', s=16)
                    break
                
                # if i==1:
                #     ax.scatter(node.globalFrame_state[0],node.globalFrame_state[1],\
                #                color='blue', s=20, label = 'A vertex')
                
                if done_flag==0:
                    # ax.scatter(node.globalFrame_state[0],node.globalFrame_state[1],\
                    #         color='blue', label='Vertex in the graph', s=16)
                    done_flag=1
                
                # ax.plot(node.globalFrame_state[0],node.globalFrame_state[1],\
                #             color='blue', marker = 'o', markersize=4)
                color = carla.Color(r=0, g=0, b=255)
                life_time = 120.0
                # Modify the point (e.g., x+2, y+2)
                waypoint_to_draw = carla.Location(x=node.globalFrame_state[0],
                                        y=node.globalFrame_state[1],
                                        z=z)

                # Optionally: check if the modified point is on a drivable lane
                # new_wp = world.get_map().get_waypoint(waypoint_to_draw, project_to_road=False, lane_type=carla.LaneType.Driving)
                # print("new wp: ", world)
                self._world.debug.draw_string(waypoint_to_draw, 'O',
                                        draw_shadow=False,
                                        color=color,
                                        life_time=life_time,
                                        persistent_lines=True)
        
        
                    
        
        
        
        # Setting the plot labels and title
        # ax.set_title('Final Constructed Graph')
        # ax.set_xlabel("Lateral distance (m)")
        # ax.set_ylabel("Longitudinal distance (m)")
        # ax.legend()
        # Save the figure as a PDF file
        # plt.savefig('constructed graph', format='pdf', bbox_inches='tight')

        # plt.show()
        time.sleep(3)  # Adjust as necessary for your system
    
    def visualize_graph_GUI(self,nodes_callback=None,path_callback=None, vehicle_callback=None):
        for i, layer_of_nodes in enumerate(self.V):    
            for node in layer_of_nodes: # all other nodes
                

                if i==len(self.V)-1: # goal node
                    nodes_callback(node.globalFrame_state[0],node.globalFrame_state[1],-1)
                
                else:
                    nodes_callback(node.globalFrame_state[0],node.globalFrame_state[1],i)

        
        # plot edges
        x_list = []
        y_list = []
        
        for edge_layer in self.E:    
            for edge in edge_layer:
                x,y,taw = edge.edge_path
                x_list.append(x)
                y_list.append(y)
        
        path_callback(x_list,y_list)

    def create_frenet_graph(self):
        # function to compute Frenet coordinates for each node and edge
        # This assumes that each node has a globalFrame_state attribute with (x, y, taw)
        # and each edge has an edge_path attribute with (x, y, taw)
        self._ref_path = np.array(self._ref_path)  # Ensure reference path is a NumPy array
        # print("ref path: ", self._ref_path)
        # step 1: compute Frenet for reference path itself

        frenet_path = self.compute_frenet(self._ref_path[:, 0], self._ref_path[:, 1])

        self._ref_path_frenet = [frenet_path[0], frenet_path[1]]  # Store Frenet coordinates for reference path


        # step 2: compute Frenet for each node and edge
        
        for i, layer_of_nodes in enumerate(self.V):
            for node in layer_of_nodes:
                # Convert node globalFrame_state to Frenet coordinates
                x = node.globalFrame_state[0]
                y = node.globalFrame_state[1]
                frenet_s, frenet_l = self.compute_frenet([x], [y])
                node.globalFrame_state_frenet = [frenet_s[0], frenet_l[0]]
                if i == len(self.V) - 1:
                    self._goal_node_frenet = (frenet_s[0], frenet_l[0])
                # print(f"Node {node.id} Frenet: {node.globalFrame_state_frenet}")

        # step 3: compute Frenet for each edge

        # This assumes each edge has an edge_path attribute with (x, y, taw)
        # and we will store Frenet coordinates in edge.edge_path_frenet
        # print("computing Frenet for edges")
      
        for edge_layer in self.E:
            for edge in edge_layer:
                # Convert edge path to Frenet coordinates
                x, y, taw = edge.edge_path
                frenet_s, frenet_l = self.compute_frenet(x, y)
                edge.edge_path_frenet = (frenet_s, frenet_l, taw)
                edge.frenet_progress = self.compute_edge_progress_frenet(edge)  # Compute progress in Frenet coordinates
                edge.time_progress = self.compute_edge_progress_time(edge)  # Compute progress in time
                # edge.time_progress = self.compute_edge_progress_time_and_lateral(edge)  # Compute progress in time
                # print(f"Edge {edge.id} Frenet: {edge.edge_path_frenet}")

        # Now, self.V and self.E contain nodes and edges with Frenet coordinates
        # You can access Frenet coordinates as follows:
        # node.globalFrame_state_frenet or edge.edge_path_frenet    

    def compute_frenet(self, x, y):
        """Converts (x, y) to Frenet (s, l) w.r.t. the reference path."""
        # Stack ref path as Nx2 array
        ref_x = self._ref_path[:, 0]
        ref_y = self._ref_path[:, 1]
        ref_points = np.vstack((ref_x, ref_y)).T
        s_list = [0]  # cumulative distance along the path
        for i in range(1, len(ref_points)):
            s_list.append(s_list[-1] + np.linalg.norm(ref_points[i] - ref_points[i-1]))
        s_list = np.array(s_list)
        # For each point, find closest ref path segment
        frenet_s = []
        frenet_l = []
        for px, py in zip(x, y):
            dists = np.hypot(ref_x - px, ref_y - py)
            idx = np.argmin(dists)
            # Find projection to segment
            if idx == len(ref_x)-1:
                idx -= 1  # clamp to valid segment
            x0, y0 = ref_x[idx], ref_y[idx]
            x1, y1 = ref_x[idx+1], ref_y[idx+1]
            dx, dy = x1 - x0, y1 - y0
            seg_len = np.hypot(dx, dy)
            if seg_len == 0:
                t = 0
            else:
                t = ((px - x0) * dx + (py - y0) * dy) / seg_len**2
                t = np.clip(t, 0, 1)
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            # s = s_list[idx] + t*seg_len
            s = s_list[idx] + t * seg_len
            # l = signed distance from projection to point
            vec = np.array([px - proj_x, py - proj_y])
            # Normal direction (rotate segment 90 degrees)
            norm = np.array([-dy, dx])
            norm = norm / (np.linalg.norm(norm) + 1e-8)
            l = np.dot(vec, norm)
            frenet_s.append(s)
            frenet_l.append(l)
        return np.array(frenet_s), np.array(frenet_l)

    def create_graph_data_structure(self):
        G = nx.DiGraph()  # Use DiGraph for directed graph, Graph for undirected


        # Add nodes to the graph (if needed, otherwise just adding edges is enough)
        for layer_idx, layer_list in enumerate( self.V):
            for node in layer_list:
                G.add_node(node,id=node.id, layer=layer_idx)
        
        # Add edges to the graph
        for edge_layer in self.E:
            for edge in edge_layer:
                # Assuming 'path' is a tuple of (from_node_id, to_node_id)
                G.add_edge(edge.edge_start_node, edge.edge_end_node,
                        edge_id= edge.id, edge_cost = edge.edge_cost, 
                        edge_distance_cost=edge.edge_distance_cost,
                        edge_time_cost_list=edge.edge_time_cost_list,
                        edge_lateral_cost=edge.edge_lateral_cost,
                        edge_time_lateral_cost_list=edge.edge_time_lateral_cost_list,
                        edge_risk=edge.edge_risk,  edge_path=edge.edge_path,
                        edge_path_frenet=edge.edge_path_frenet, frenet_progress=edge.frenet_progress, time_progress=edge.time_progress)
        
        return G, self._goal_indecies_list, self._ref_path

    def compute_edge_progress_frenet(self, edge):
        """Compute the progress of an edge based on its path."""
        if not edge.edge_path:
            return 0.0
        # Extract the goal node
        goal_node=self._goal_node_frenet
        # print("-----------------goal node: ", goal_node)
        


        vi = edge.edge_start_node.globalFrame_state
        vj = edge.edge_end_node.globalFrame_state
        x_cart = [vi[0], vj[0]]
        y_cart = [vi[1], vj[1]]

        # print (x_cart)
        # print (y_cart)
        
        edge_frenet_s,edge_frenet_l =self.compute_frenet(x_cart, y_cart)
        vi_frenet = [edge_frenet_s[0],edge_frenet_l[0]]
        vj_frenet = [edge_frenet_s[1],edge_frenet_l[1]]
        # print(vi_frenet)
        # print(vj_frenet)
        return progress_utility_frenet_alpha(vi_frenet, vj_frenet, goal_node)
    
    def compute_edge_progress_time(self, edge, method='big_M'):
        """Compute the progress of an edge based on its path."""
        # constantas

        if method == 'reciprocal':
            a=1
            b=1

            if not edge.edge_path:
                return 0.0
            # get the travel time costs
            C_ij_v = []
            # P_ij_v = {}
            P_ij_v = []
            for s in self._speeds_set:
                travel_time_cost = (edge.edge_cost / s) * 3.6  # to convert to m/s
                C_ij_v.append(travel_time_cost)
                # P_ij_v[s]=(a / (a + b * travel_time_cost))
                P_ij_v.append(a / (a + b * travel_time_cost))
            
            return min(P_ij_v)
        elif method == 'big_M':
            M = 20  # A large constant
            if not edge.edge_path:
                return 0.0
            # get the travel time costs
            C_ij_v = []
            P_ij_v = []
            for s in self._speeds_set:
                travel_time_cost = (edge.edge_cost / s) * 3.6  # to convert to m/s
                C_ij_v.append(travel_time_cost)
                prg_val = M - travel_time_cost
                P_ij_v.append(prg_val)
            
            return min(P_ij_v)
        else:
            raise ValueError("Unsupported method specified. Use 'reciprocal' or 'big_M'.")
    
    def compute_edge_progress_time_and_lateral(self, edge):
        """Compute the progress of an edge based on its path."""
        # constantas

        # get the frenet coordinates

        if not edge.edge_path:
            return 0.0
        # Extract the goal node
        goal_node=self._goal_node_frenet
        # print("-----------------goal node: ", goal_node)
        


        vi = edge.edge_start_node.globalFrame_state
        vj = edge.edge_end_node.globalFrame_state
        x_cart = [vi[0], vj[0]]
        y_cart = [vi[1], vj[1]]

        # print (x_cart)
        # print (y_cart)
        
        edge_frenet_s,edge_frenet_l =self.compute_frenet(x_cart, y_cart)
        vi_frenet = [edge_frenet_s[0],edge_frenet_l[0]]
        vj_frenet = [edge_frenet_s[1],edge_frenet_l[1]]
        
        lateral_offset = edge_frenet_l[1]
        
        a=1
        b=1

        if not edge.edge_path:
            return 0.0
        # get the travel time costs
        C_ij_v = []
        P_ij_v = []
        for s in self._speeds_set:
            travel_time_cost = (edge.edge_cost / s) * 3.6  # to convert to m/s
            C_ij_v.append(travel_time_cost)
            prg_val = a / (a + b * travel_time_cost+lateral_offset)
            P_ij_v.append(prg_val)
        
        return P_ij_v
         
         