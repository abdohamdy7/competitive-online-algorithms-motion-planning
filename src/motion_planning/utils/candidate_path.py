class CANDIDATE_PATH:
    def __init__(self, risk, cost,utility, start_node, subgoal_node, solution_edge_speeds, cost_matrix, risk_matrix, utility_matrix, frenet_progress=None, time_progress=None, ratio=None) -> None:
        self.risk = risk
        self.cost = cost
        self.utility = utility
        self.start_node = start_node
        self.subgoal_node = subgoal_node
        self.solution_edge_speeds = solution_edge_speeds
        self.cost_matrix = cost_matrix
        self.risk_matrix = risk_matrix
        self.utility_matrix = utility_matrix    
        self.frenet_progress = frenet_progress
        self.time_progress = time_progress        
        self.ratio = ratio
        self.init_lists()


    def get_path_length(self):
        return len(self.solution_edge_speeds)

    def init_lists(self):
        self.risk_list = []
        self.cost_list = []
        self.utility_list = []
        for u, v, speed in self.solution_edge_speeds:
            self.risk_list.append(self.risk_matrix[u,v,speed])
            self.cost_list.append(self.cost_matrix[u,v,speed])
            self.utility_list.append(self.utility_matrix[u,v,speed])

            
