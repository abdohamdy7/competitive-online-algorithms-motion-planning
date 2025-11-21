"""
Module: risk_generator.py
-------------------------
This module provides utilities for assigning random risk values to 
a subset of edges in a graph.

Each selected edge receives a random risk value between 0.1 and `risk_max`, 
while all unselected edges receive a small baseline risk value (e.g., 0.001).
"""

import random
from typing import Any, Dict, Iterable, Optional, Tuple

import networkx as nx
import numpy as np

from motion_planning.graph_construction.main_graph_construction import SPEED_SET


BASELINE_RISK = 0.001


class RiskLevel:
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 0.9

def generate_random_risks(
    graph: nx.Graph,
    risk_level: float,
    risk_max: float = 5,
    *,
    with_speed: bool = False,
    speed_options: Optional[Iterable[float]] = None,
) -> Dict[Tuple[Any, ...], float]:
    """
    Generate random risk values for a subset of edges in a graph.

    Parameters
    ----------
    graph : nx.Graph
        The input graph whose edges will be assigned risk values.
    risk_level : float
        Fraction (0.0-1.0) of edges to assign random (non-zero) risk values to.
        Convenience constants are provided in ``RiskLevel``.
    risk_max : float
        Maximum risk value for the randomly selected edges.

    with_speed : bool, optional
        When True, risk values are generated per (edge, speed) combination.
        The returned dictionary keys are triples (u, v, speed). When False,
        keys are edge tuples (u, v). Defaults to False.
    speed_options : Iterable[float], optional
        Iterable of speed options to use when ``with_speed`` is True. If not
        provided the global ``SPEED_SET`` is used.

    Returns
    -------
    Dict[Tuple[Any, ...], float]
        Dictionary mapping edges (u, v) or (u, v, speed) to risk values,
        depending on the value of ``with_speed``.
    """
    # Select a random subset of edges
    num_edges = int(risk_level * graph.number_of_edges())
    random_edges = set(select_random_edges(graph, num_edges))

    # Initialize the dictionary to hold edge risk values
    if with_speed:
        speeds = list(speed_options if speed_options is not None else SPEED_SET)
        if not speeds:
            raise ValueError("speed_options must contain at least one speed value.")

        PHI_ijv: Dict[Tuple[Any, Any, float], float] = {}
        for (u, v) in graph.edges:
            for speed in speeds:
                if (u, v) in random_edges:
                    PHI_ijv[(u, v, speed)] = np.random.uniform(0.1, risk_max)
                else:
                    PHI_ijv[(u, v, speed)] = BASELINE_RISK
        return PHI_ijv

    PHI_ij: Dict[Tuple[Any, Any], float] = {}

    # Assign risks: random for selected edges, small baseline for others
    for (u, v) in graph.edges:
        if (u, v) in random_edges:
            PHI_ij[(u, v)] = np.random.uniform(0.1, risk_max)
        else:
            PHI_ij[(u, v)] = BASELINE_RISK

    return PHI_ij


def select_random_edges(graph: nx.Graph, num_edges: int) -> list[Tuple]:
    """
    Select a specified number of random edges from the graph.

    Parameters
    ----------
    graph : nx.Graph
        The graph from which edges will be randomly selected.
    num_edges : int
        Number of edges to randomly select.

    Returns
    -------
    random_edges : list of tuple
        List containing randomly chosen edges as (u, v) pairs.

    Raises
    ------
    ValueError
        If `num_edges` exceeds the total number of edges in the graph.
    """
    edges = list(graph.edges())

    if num_edges > len(edges):
        raise ValueError(
            f"Number of edges to select ({num_edges}) exceeds "
            f"total number of edges in the graph ({len(edges)})."
        )

    # Randomly sample edges without replacement
    random_edges = random.sample(edges, num_edges)
    return random_edges



# Example usage (for testing/debugging)
if __name__ == "__main__":
    # Create a sample graph with 10 nodes and random connections
    G = nx.gnm_random_graph(10, 20, seed=42)
    
    # Generate random risks for 5 edges
    risks = generate_random_risks(G, risk_level=RiskLevel.MEDIUM, risk_max=0.5)
    
    # Display results
    for edge, risk in risks.items():
        print(f"Edge {edge}: Risk = {risk:.4f}")
