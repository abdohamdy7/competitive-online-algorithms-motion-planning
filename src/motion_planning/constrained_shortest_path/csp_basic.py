import numpy as np
from typing import List, Tuple, Optional

"""
Constrained Shortest Path (CSP) Solution Implementation

This module provides functionality for finding the shortest path between two points
while satisfying additional constraints (e.g., risk).

The CSP problem is a variant of the classic shortest path problem where paths must 
satisfy one or more constraints while minimizing the primary cost metric.

Key components:
- CSP solver implementations
- Helper functions for constraint handling
- Path validation and optimization routines
"""

# Import required libraries 


import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import numpy as np
import math
import re
from motion_planning.constrained_shortest_path.gurobi_license import options


class CSP_BASIC:
    def __init__(self, Graph, start_node, goal_node, risk_max, risk_matrix):
        """
        Initializes the CSP problem.
        given: graph, s, g

        Parameters:
        - Graph:
        - start_node: 
        - goal_node: 
        """
        
        self._Graph = Graph
        self._edges = Graph.edges(data=True)        
        # self._PHI_ij = PHI_ij
        self._start_node = start_node
        self._goal_node = goal_node
        self._risk_max = risk_max
        self._risk_matrix = risk_matrix

        self.model = None
        # self.max_speed= 30 # km/hr
        # self.min_speed= 5  # km/hr
    
    # def get_dev_value ( self, i,j):
    #     i_nums =  re.findall(r'\d+', i.id)
    #     j_nums =  re.findall(r'\d+', j.id)
    #     extracted_numbers_i = list(map(int, i_nums))
    #     extracted_numbers_j = list(map(int, j_nums))

    #     dev_val = abs(extracted_numbers_i[1]-extracted_numbers_j[1])

    #     return dev_val

    
    def setup_model(self):

        """Sets up the ILP model for the CSP."""

        with gp.Env(params=options) as env:

            self.model = gp.Model(env=env, name="re_CSP")

            # model variables:

            # Decision Variables
            self.x = {} # edge selector
            # self._PHI_ij = {}
            # creating the variables
            
            
            # Add penalty for direction changes to the objective
            # self.dev_edge_penalty_weight = 10 # Adjust this weight to control the importance of penalizing direction changes
            # self.dev_speed_penalty_weight = 1

            # over the whole edges: x & PHI_ij
            for i, j in self._Graph.edges:
                    # edge variable : x
                    self.x[(i, j)] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{i.id}_{j.id}")
                    # self._PHI_ij[(i, j)] = self.model.addVar(vtype=GRB.CONTINUOUS,lb=0.0, name=f"PHI_{i.id}_{j.id}")
              
            # Objective: Minimize the total travel time of the path
            # Travel time is distance / speed
            # Convert edge distances to time in hours for each speed option
            # The distance on the edges is given in km, so no conversion is necessary for km/hr speeds
            self.travel_time_cost = {
                (i, j): (edge_info['edge_cost'] / 30)*3.6 # to convert to m/s
                for i, j, edge_info in self._edges
            }

            # self.deviation_cost = {
            #     (i,j): self.get_dev_value(i,j)
            #     for i,j, _ in self._edges
            #     # for v in self.speed_options
            # }

            # self.model.setObjective(gp.quicksum(self.x[i, j, v] * self.travel_time_cost[(i, j, v)] for i, j, _ in self._edges for v in self.speed_options),GRB.MINIMIZE)
            self.model.setObjective(
                gp.quicksum(self.x[i, j] * (self.travel_time_cost[(i, j)]) 
                            for i, j, _ in self._edges),
                GRB.MINIMIZE
            )

            

            #########    Model Constraints    #########

            # 1- FLOW CONSTRAINTs
            # Flow conservation constraints for intermediate vertices


            # Constraints for all nodes except start and end
            for node in self._Graph.nodes:  
                if node not in [self._start_node, self._goal_node]:
                    self.model.addConstr(
                        quicksum(self.x[i, j] for i, j, _ in self._edges if j == node ) ==
                        quicksum(self.x[i, j] for i, j, _ in self._edges if i == node),
                        f"flow_conservation_{node.id}"
                    )

            # 2- Additional constraints: Ensure the start and end conditions are met
            
            self.model.addConstr(sum(self.x[(self._start_node, j)] for i, j, _ in self._edges if i == self._start_node ) == 1, "start_condition")
            self.model.addConstr(sum(self.x[(i, self._goal_node)] for i, j, _ in self._edges if j == self._goal_node ) == 1, "goal_condition")
            
            
            


            #3- risk  constraints

            
            # for k in self._risk_matrix.keys():
            #     i,j = k
            #     self.model.addConstr(self._PHI_ij[i,j] == self._risk_matrix[i,j]) 
            
            
            # Additional risk constraint: Sum of risks along the path must not exceed max_risk
            self.model.addConstr(gp.quicksum(self.x[i,j] * self._risk_matrix[i,j] for i, j in self._Graph.edges ) <= self._risk_max, f"Risk_Constraint")


    def solve(self):
        """Solves the CSP model."""
        if not self.model:
            self.setup_model()
            
        self.model.optimize()
        # After running optimize and encountering infeasibility
        if self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Finding IIS...")
            self.model.computeIIS()
            self.model.write("model.ilp")
            
            # Print IIS
            if self.model.IISMinimal:
                print("IIS is minimal")
            else:
                print("IIS is not minimal")
            print("\nThe following constraint(s) are infeasible:")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"Infeasible constraint: {c.constrName}")

        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
            solution_edges = [(i, j) for i, j, edge_info in self._edges  if self.x[i,j].x > 0.5]
            # print(f"Edges in the optimal path: {solution_edges}")
            # Retrieve the optimal speeds for each edge on the path

            return solution_edges
        else:
            print("No optimal solution could be found.")
            return None



class CSP_max:
    def __init__(self, Graph, start_node, goal_node, risk_max, risk_matrix):
        """
        Initializes the CSP problem.
        given: graph, s, g

        Parameters:
        - Graph:
        - start_node: 
        - goal_node: 
        """
        
        self._Graph = Graph
        self._edges = Graph.edges(data=True)        
        # self._PHI_ij = PHI_ij
        self._start_node = start_node
        self._goal_node = goal_node
        self._risk_max = risk_max
        self._risk_matrix = risk_matrix

        self.model = None
        # self.max_speed= 30 # km/hr
        # self.min_speed= 5  # km/hr
    
    # def get_dev_value ( self, i,j):
    #     i_nums =  re.findall(r'\d+', i.id)
    #     j_nums =  re.findall(r'\d+', j.id)
    #     extracted_numbers_i = list(map(int, i_nums))
    #     extracted_numbers_j = list(map(int, j_nums))

    #     dev_val = abs(extracted_numbers_i[1]-extracted_numbers_j[1])

    #     return dev_val

    
    def setup_model(self):

        """Sets up the ILP model for the CSP."""

        with gp.Env(params=options) as env:

            self.model = gp.Model(env=env, name="re_CSP")

            # model variables:

            # Decision Variables
            self.x = {} # edge selector
            # self._PHI_ij = {}
            # creating the variables
            
            
            # Add penalty for direction changes to the objective
            # self.dev_edge_penalty_weight = 10 # Adjust this weight to control the importance of penalizing direction changes
            # self.dev_speed_penalty_weight = 1

            # over the whole edges: x & PHI_ij
            for i, j in self._Graph.edges:
                    # edge variable : x
                    self.x[(i, j)] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{i.id}_{j.id}")
                    # self._PHI_ij[(i, j)] = self.model.addVar(vtype=GRB.CONTINUOUS,lb=0.0, name=f"PHI_{i.id}_{j.id}")
              
            # Objective: Minimize the total travel time of the path
            # Travel time is distance / speed
            # Convert edge distances to time in hours for each speed option
            # The distance on the edges is given in km, so no conversion is necessary for km/hr speeds
            self.travel_time_cost = {
                (i, j): (edge_info['edge_cost'] / 30)*3.6 # to convert to m/s
                for i, j, edge_info in self._edges
            }

            # self.deviation_cost = {
            #     (i,j): self.get_dev_value(i,j)
            #     for i,j, _ in self._edges
            #     # for v in self.speed_options
            # }

            # self.model.setObjective(gp.quicksum(self.x[i, j, v] * self.travel_time_cost[(i, j, v)] for i, j, _ in self._edges for v in self.speed_options),GRB.MINIMIZE)
            self.model.setObjective(
                gp.quicksum(self.x[i, j] * (self.travel_time_cost[(i, j)]) 
                            for i, j, _ in self._edges),
                GRB.MINIMIZE
            )

            

            #########    Model Constraints    #########

            # 1- FLOW CONSTRAINTs
            # Flow conservation constraints for intermediate vertices


            # Constraints for all nodes except start and end
            for node in self._Graph.nodes:  
                if node not in [self._start_node, self._goal_node]:
                    self.model.addConstr(
                        quicksum(self.x[i, j] for i, j, _ in self._edges if j == node ) ==
                        quicksum(self.x[i, j] for i, j, _ in self._edges if i == node),
                        f"flow_conservation_{node.id}"
                    )

            # 2- Additional constraints: Ensure the start and end conditions are met
            
            self.model.addConstr(sum(self.x[(self._start_node, j)] for i, j, _ in self._edges if i == self._start_node ) == 1, "start_condition")
            self.model.addConstr(sum(self.x[(i, self._goal_node)] for i, j, _ in self._edges if j == self._goal_node ) == 1, "goal_condition")
            
            
            


            #3- risk  constraints

            
            # for k in self._risk_matrix.keys():
            #     i,j = k
            #     self.model.addConstr(self._PHI_ij[i,j] == self._risk_matrix[i,j]) 
            
            
            # Additional risk constraint: Sum of risks along the path must not exceed max_risk
            self.model.addConstr(gp.quicksum(self.x[i,j] * self._risk_matrix[i,j] for i, j in self._Graph.edges ) <= self._risk_max, f"Risk_Constraint")


    def solve(self):
        """Solves the CSP model."""
        if not self.model:
            self.setup_model()
            
        self.model.optimize()
        # After running optimize and encountering infeasibility
        if self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Finding IIS...")
            self.model.computeIIS()
            self.model.write("model.ilp")
            
            # Print IIS
            if self.model.IISMinimal:
                print("IIS is minimal")
            else:
                print("IIS is not minimal")
            print("\nThe following constraint(s) are infeasible:")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"Infeasible constraint: {c.constrName}")

        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
            solution_edges = [(i, j) for i, j, edge_info in self._edges  if self.x[i,j].x > 0.5]
            # print(f"Edges in the optimal path: {solution_edges}")
            # Retrieve the optimal speeds for each edge on the path

            return solution_edges
        else:
            print("No optimal solution could be found.")
            return None
