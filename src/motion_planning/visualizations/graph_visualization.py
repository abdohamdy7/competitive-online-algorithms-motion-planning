"""Utilities for visualising motion-planning graphs in Carla and Matplotlib."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
from motion_planning.utils.paths import FIGURES_DIR
import matplotlib.pyplot as plt

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
    "emit_graph_to_callbacks",
    "GRAPH_VISUALIZATION",
]
