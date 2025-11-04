"""Utilities for visualising motion-planning graphs in Carla and Matplotlib."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path

from motion_planning.utils.paths import FIGURES_DIR
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from motion_planning.risk_assessment.generate_risks_randomly import*

import networkx as nx
try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    carla = None  # type: ignore

HIGHWAY_SCENARIOS = {"Highway1", "Highway2", "Highway3"}

Node = Any
Edge = Any
NodeLayers = Sequence[Sequence[Node]]
EdgeLayers = Sequence[Sequence[Edge]]


def _z_offset_for_scenario(scenario: str) -> float:
    """Return the Carla z-offset for a given scenario."""
    return 10.0 if scenario in HIGHWAY_SCENARIOS else 0.0


def _iter_edges(edge_layers: EdgeLayers) -> Iterable[Edge]:
    for edge_layer in edge_layers:
        for edge in edge_layer:
            yield edge


def _iter_nodes(node_layers: NodeLayers) -> Iterable[Tuple[int, Node]]:
    for layer_idx, layer in enumerate(node_layers):
        for node in layer:
            yield layer_idx, node


def visualize_graph_in_matplotlib(
    nodes: NodeLayers,
    edges: EdgeLayers,
    scenario: str,
    *,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    output_path: Optional[str] = None,
) -> plt.Axes:
    """Render the constructed graph using Matplotlib."""
    if ax is None:
        _, ax = plt.subplots()

    for edge in _iter_edges(edges):
        x, y, _ = edge.edge_path
        ax.plot(x, y, color="grey", linewidth=0.6, alpha=0.7)

    start_drawn = False
    goal_drawn = False
    intermediate_drawn = False
    layer_count = len(nodes)

    for layer_idx, node in _iter_nodes(nodes):
        x, y = node.globalFrame_state[:2]
        if layer_idx == 0 and not start_drawn:
            ax.scatter(x, y, color="red", s=16, label="Start vertex")
            start_drawn = True
        elif layer_idx == layer_count - 1 and not goal_drawn:
            ax.scatter(x, y, color="green", s=16, label="Goal vertex")
            goal_drawn = True
        elif not intermediate_drawn:
            ax.scatter(x, y, color="blue", s=6, label="Vertex in the graph")
            intermediate_drawn = True
        else:
            ax.scatter(x, y, color="blue", s=6)

    ax.set_title(f"Final Constructed Graph ({scenario})")
    ax.set_xlabel("Lateral distance (m)")
    ax.set_ylabel("Longitudinal distance (m)")
    if start_drawn or goal_drawn or intermediate_drawn:
        ax.legend()

    if output_path is not None:
        ax.figure.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return ax


def visualize_graph_in_carla(
    nodes: NodeLayers,
    edges: EdgeLayers,
    world: Any,
    scenario: str,
    *,
    line_life_time: float = 0.0,
    node_life_time: float = 120.0,
    persistent_lines: bool = True,
) -> None:
    """Render the constructed graph inside Carla using debug primitives."""
    if carla is None:
        raise ImportError("Carla Python API is not available in this environment.")

    z_offset = _z_offset_for_scenario(scenario)

    for edge in _iter_edges(edges):
        x, y, _ = edge.edge_path
        for idx in range(len(x) - 1):
            start = carla.Location(x=x[idx], y=y[idx], z=z_offset)
            end = carla.Location(x=x[idx + 1], y=y[idx + 1], z=z_offset)
            world.debug.draw_line(
                start,
                end,
                thickness=0.1,
                color=carla.Color(r=0, g=0, b=0),
                life_time=line_life_time,
                persistent_lines=persistent_lines,
            )

    for _, node in _iter_nodes(nodes):
        location = carla.Location(
            x=node.globalFrame_state[0],
            y=node.globalFrame_state[1],
            z=z_offset,
        )
        world.debug.draw_string(
            location,
            "O",
            draw_shadow=False,
            color=carla.Color(r=0, g=0, b=255),
            life_time=node_life_time,
            persistent_lines=persistent_lines,
        )


def visualize_graph_in_carla_and_matplotlib(
    nodes: NodeLayers,
    edges: EdgeLayers,
    world: Any,
    scenario: str,
    *,
    line_life_time: float = 0.0,
    node_life_time: float = 120.0,
    persistent_lines: bool = True,
    show_matplotlib: bool = True,
    matplotlib_output_path: Optional[str] = None,
) -> plt.Axes:
    """Render the graph both in Carla and Matplotlib."""
    visualize_graph_in_carla(
        nodes,
        edges,
        world,
        scenario,
        line_life_time=line_life_time,
        node_life_time=node_life_time,
        persistent_lines=persistent_lines,
    )
    return visualize_graph_in_matplotlib(
        nodes,
        edges,
        scenario,
        show=show_matplotlib,
        output_path=matplotlib_output_path,
    )


def visualize_graph_with_risks(
    risk_matrix: dict[Tuple[Any, Any], float],
    graph: nx.Graph,
    show: bool = True,
    scenario_name: Optional[str] = None,
    risk_level: Optional[Any] = None,
    output_path: Optional[str] = None,
    *,
    safe_threshold: float = 0.001,
    ax: Optional[plt.Axes] = None,
    colormap: str = "berlin",
    # colormap: str = "jet",
) -> plt.Axes:
    """Visualize the graph with edge colors representing risk levels."""
    if ax is None:
        _, ax = plt.subplots()

    if not risk_matrix:
        raise ValueError("risk_matrix must contain at least one edge entry.")

    risk_values = list(risk_matrix.values())
    significant_risks = [value for value in risk_values if value > safe_threshold]
    vmin = min(significant_risks, default=safe_threshold)
    vmax = max(significant_risks, default=vmin + 1.0)
    if vmin == vmax:
        vmax = vmin + 1.0

    cmap = cm.get_cmap(colormap)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    safe_edge_color = (0.75, 0.75, 0.75, 0.4)

    for (u, v), risk_value in risk_matrix.items():
        if not graph.has_edge(u, v):
            continue
        edge_path = graph.edges[u, v].get("edge_path")
        if not edge_path or len(edge_path) < 2:
            continue
        x, y, *_ = edge_path
        if risk_value <= safe_threshold:
            ax.plot(x, y, color=safe_edge_color, linewidth=0.5, alpha=0.6)
        else:
            ax.plot(x, y, color=cmap(norm(risk_value)), linewidth=0.6, alpha=0.9)

    ax.set_title(f"Graph Visualization with Edge Risks (Density: {int (risk_level*100)}%)")
    ax.set_xlabel("Lateral distance (m)")
    ax.set_ylabel("Longitudinal distance (m)")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Risk Level")

    output_path_to_use: Path
    if output_path is None:
        if scenario_name is None or risk_level is None:
            raise ValueError(
                "scenario_name and risk_level must be provided when output_path is not specified.",
            )
        output_path_to_use = (
            Path(FIGURES_DIR)
            / "risk figures"
            / f"graph_with_risks_{scenario_name}_{risk_level}.pdf"
        )
    else:
        output_path_to_use = Path(output_path)

    output_path_to_use.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(
        output_path_to_use,
        format="pdf",
        bbox_inches="tight",
    )

    if show:
        plt.show()

    return ax


def visualize_graph_with_risks_with_speeds(
    risk_matrix: dict[Tuple[Any, Any, Any], float],
    graph: nx.Graph,
    *,
    speed: Optional[Any] = None,
    scenario_name: Optional[str] = None,
    risk_level: Optional[Any] = None,
    show: bool = True,
    output_path: Optional[str] = None,
    safe_threshold: float = 0.001,
    colormap: str = "viridis",
) -> plt.Axes:
    """
    Visualize edge risks for a single speed layer selected from a speed-aware risk matrix.

    Parameters
    ----------
    risk_matrix : dict[(u, v, speed), float]
        Risk values keyed by edge endpoints and speed option.
    graph : nx.Graph
        Graph containing the edges to plot.
    speed : optional
        Specific speed option to visualise. Defaults to the first available speed.
    scenario_name, risk_level, show, output_path, safe_threshold, colormap
        Forwarded to :func:`visualize_graph_with_risks`.

    Returns
    -------
    plt.Axes
        The matplotlib axes containing the plot.
    """
    if not risk_matrix:
        raise ValueError("risk_matrix must contain at least one (edge, speed) entry.")

    available_speeds = sorted({edge_speed for (_, _, edge_speed) in risk_matrix.keys()})
    if not available_speeds:
        raise ValueError("No speed options found in risk_matrix keys.")

    speed_to_plot = speed if speed is not None else available_speeds[0]
    if speed_to_plot not in available_speeds:
        raise ValueError(
            f"Speed {speed_to_plot!r} is not present in the risk_matrix. "
            f"Available speeds: {available_speeds}",
        )

    filtered_risks = {
        (u, v): risk_value
        for (u, v, edge_speed), risk_value in risk_matrix.items()
        if edge_speed == speed_to_plot
    }
    if not filtered_risks:
        raise ValueError(
            f"No risk values found for speed {speed_to_plot!r}.",
        )

    custom_output_path: Optional[str] = output_path
    if output_path is None and scenario_name is not None and risk_level is not None:
        custom_output_path = (
            Path(FIGURES_DIR)
            / "risk figures"
            / f"graph_with_risks_{scenario_name}_{risk_level}_speed_{speed_to_plot}.pdf"
        ).as_posix()

    ax = visualize_graph_with_risks(
        filtered_risks,
        graph,
        show=show,
        scenario_name=scenario_name,
        risk_level=risk_level,
        output_path=custom_output_path,
        safe_threshold=safe_threshold,
        colormap=colormap,
    )

    ax.set_title(f"Graph With Edge Risks (speed={speed_to_plot})")
    return ax

def emit_graph_to_callbacks(
    nodes: NodeLayers,
    edges: EdgeLayers,
    *,
    node_callback: Callable[[float, float, int], None],
    path_callback: Callable[[List[Sequence[float]], List[Sequence[float]]], None],
) -> None:
    """Push graph data to GUI callbacks."""
    layer_count = len(nodes)
    for layer_idx, node in _iter_nodes(nodes):
        x, y = node.globalFrame_state[:2]
        node_callback(x, y, -1 if layer_idx == layer_count - 1 else layer_idx)

    x_list: List[Sequence[float]] = []
    y_list: List[Sequence[float]] = []

    for edge in _iter_edges(edges):
        x, y, _ = edge.edge_path
        x_list.append(x)
        y_list.append(y)

    path_callback(x_list, y_list)


class GRAPH_VISUALIZATION:
    """Object-oriented wrapper kept for backward compatibility."""

    def __init__(self, V: NodeLayers, E: EdgeLayers, world: Any, scenario: str) -> None:
        self.V = V
        self.E = E
        self._world = world
        self._scenario = scenario

    def visualize_final_graph_matplotlib(self, *, show: bool = True) -> plt.Axes:
        return visualize_graph_in_matplotlib(
            self.V,
            self.E,
            self._scenario,
            show=show,
            output_path=f"{FIGURES_DIR}/graphs figures/constructed_graph_{self._scenario}.pdf",
        )

    def visualize_final_graph_carla(
        self,
        *,
        line_life_time: float = 0.0,
        node_life_time: float = 120.0,
        persistent_lines: bool = True,
    ) -> None:
        visualize_graph_in_carla(
            self.V,
            self.E,
            self._world,
            self._scenario,
            line_life_time=line_life_time,
            node_life_time=node_life_time,
            persistent_lines=persistent_lines,
        )

    def visualize_final_graph_both(
        self,
        *,
        line_life_time: float = 0.0,
        node_life_time: float = 120.0,
        persistent_lines: bool = True,
        show_matplotlib: bool = True,
        matplotlib_output_path: Optional[str] = "constructed_graph.pdf",
    ) -> plt.Axes:
        return visualize_graph_in_carla_and_matplotlib(
            self.V,
            self.E,
            self._world,
            self._scenario,
            line_life_time=line_life_time,
            node_life_time=node_life_time,
            persistent_lines=persistent_lines,
            show_matplotlib=show_matplotlib,
            matplotlib_output_path=matplotlib_output_path,
        )

    def visualize_graph_GUI(
        self,
        *,
        nodes_callback: Callable[[float, float, int], None],
        path_callback: Callable[[List[Sequence[float]], List[Sequence[float]]], None],
    ) -> None:
        emit_graph_to_callbacks(
            self.V,
            self.E,
            node_callback=nodes_callback,
            path_callback=path_callback,
        )
    
    


__all__ = [
    "visualize_graph_in_carla",
    "visualize_graph_in_matplotlib",
    "visualize_graph_in_carla_and_matplotlib",
    "visualize_graph_with_risks",
    "visualize_graph_with_risks_with_speeds",
    "emit_graph_to_callbacks",
    "GRAPH_VISUALIZATION",
]
