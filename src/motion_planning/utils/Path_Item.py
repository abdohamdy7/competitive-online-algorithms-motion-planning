class Path_Item:
    def __init__(self, risk, utility, time_cost, start_node, edges,frenet_progress=0.0, time_progress=0.0,time_cost_matrix_dict=None, ratio=None) -> None:
        self.risk = risk
        self.utility = utility
        self.frenet_progress = frenet_progress
        self.time_progress = time_progress
        self.time_cost = time_cost
        self.start_node = start_node
        self.edges = edges
        self.ratio = ratio
        self.time_cost_matrix_dict = time_cost_matrix_dict
        self.risk_trace = [0.0]
