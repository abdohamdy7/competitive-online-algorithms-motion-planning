'''
generate a signle offline motion planning problem instance, the problem instance is 

'''
from motion_planning.offline_problems.utils import *
from motion_planning.offline_problems.generate_candidates import generate_risk_bounded_candidate_paths
from motion_planning.risk_assessment.generate_risks_randomly import generate_random_risks
from motion_planning.constrained_shortest_path.costs_definitions import EDGE_COST
from motion_planning.constrained_shortest_path.offline_csp import get_optimal_offline_CSP
from motion_planning.constrained_shortest_path.gurobi_license import GUROBI_OPTIONS

def generate_offline_problem(graph, risk_level, risk_budget, num_epochs, goal_indecies, carla_route):

# 1- generate random risks with speeds on the edges based on the risk level.
# 2- generate random decision epochs.
# 3- for each start node and subgoal node pair, generate candidate paths using the two methods within the risk budget
    
    
    decision_epochs = get_random_decision_epochs(num_epochs= num_epochs, goal_indecies=goal_indecies)
    decision_nodes = get_decision_nodes(graph, decision_epochs, carla_route)
    start_goal_pairs = get_start_goal_node_pairs(graph, decision_nodes)
    offline_problem_generation_flag = False
    
    final_problem_risk_bounded_candidates = {}
    final_problem_all_candidates = {}
    final_problem_offline_csp = {}
        
    
    while not offline_problem_generation_flag:

        print(offline_problem_generation_flag)
        print("generating risks!!!")
            
        PHI_ij = generate_random_risks(graph, risk_level=risk_level, with_speed=True)

        problem_risk_bounded_candidates= {}
        problem_all_candidates = {}
    
        for epoch, pair in enumerate(start_goal_pairs):
    
            s, g = pair

            print(f'now generating from {s.id} to {g.id}!!!')

            risk_bounded_candidates = generate_risk_bounded_candidate_paths(
                graph=graph,start_node= s, sub_goal_node= g,
                Delta=risk_budget, PHI_ij=PHI_ij, edge_cost_key=EDGE_COST.EDGE_TIME_LATERAL_COST_LIST)

            # check if risk_bounded_candidates is None, then no need to continue in this experiment.
            if risk_bounded_candidates is None:
                print(f"No candidates found for {s.id} to {g.id} within risk budget {risk_budget}. Regenerating risks.")
                offline_problem_generation_flag = False
                break
            

            print("generated rb candidates; skipping exhaustive all-candidate enumeration (too costly).")
            print(f'number of rb candidates are: {len(risk_bounded_candidates)}')

            #     all_candidates = generate_all_candidate_paths_under_budget (
            #     graph=graph,
            #     start_node=s,
            #     sub_goal_node=g,
            #     delta_risk=risk_budget,
            #     PHI_ij=PHI_ij,
            # )

            all_candidates = None
            problem_risk_bounded_candidates[(epoch, s.id, g.id)]= risk_bounded_candidates
            problem_all_candidates[(epoch, s.id, g.id)] = all_candidates
        else:
            offline_start_node = decision_nodes[0]
            offline_goal_node = start_goal_pairs[-1][1]
            offline_sub_goals = decision_nodes
            print("All RB candidates ready; running offline CSP to stitch full path...")
            offline_solution, offline_solver = get_optimal_offline_CSP(
                graph,
                offline_start_node,
                offline_goal_node,
                sub_goals=offline_sub_goals,
                risk_matrix=PHI_ij,
                risk_budget=risk_budget,
                edge_cost_list_key=EDGE_COST.EDGE_TIME_LATERAL_COST_LIST,gurobi_env_params=GUROBI_OPTIONS
            )
            if offline_solution is None:
                print("Offline CSP infeasible with the current risks. Regenerating...")
                offline_problem_generation_flag = False
                continue
            offline_problem_generation_flag = True
            final_problem_risk_bounded_candidates[graph, risk_level, risk_budget, num_epochs] = problem_risk_bounded_candidates
            final_problem_all_candidates[graph, risk_level, risk_budget, num_epochs] = problem_all_candidates
            final_problem_offline_csp[graph, risk_level, risk_budget, num_epochs] = {
                "solution_edge_speeds": offline_solution,
                "path_nodes": offline_solver.solution_path_nodes(),
            }
        

    
    
    
    return final_problem_all_candidates, final_problem_risk_bounded_candidates, final_problem_offline_csp
