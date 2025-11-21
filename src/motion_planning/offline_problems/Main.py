"""
Command-line automation for generating batches of offline motion-planning problems.

The script sweeps every stored planning graph across multiple risk budgets and
risk levels, invokes the single-instance generator, and persists the resulting
candidate sets through :mod:`offline_problem_recorder`.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import networkx as nx
import numpy as np

from motion_planning.graph_construction.graph_assets import GRAPH_FILE_HANDLER
from motion_planning.offline_problems.generate_offline_problem import generate_offline_problem
from motion_planning.offline_problems.offline_problem_recorder import OfflineProblemRecorder
from motion_planning.risk_assessment.generate_risks_randomly import RiskLevel
from motion_planning.utils.paths import GRAPH_DIR, ROUTES_DIR

LOGGER = logging.getLogger("offline-problem-generator")

DEFAULT_RISK_LEVELS = ("low", "mid", "high")
DEFAULT_RISK_BUDGETS = (5.0, 10.0, 15.0, 20.0)
DEFAULT_LOG_FILE = Path("logs/offline_problem_generator.log")

RiskLevelSpec = Tuple[str, float]
CandidateMapping = Mapping[Tuple[int, str, str], Sequence[object]]
EdgeMetric = Dict[Tuple[object, object, object], float]

_GRAPH_HANDLER = GRAPH_FILE_HANDLER()


@dataclass(frozen=True)
class ScenarioResources:
    """Container with the filesystem resources required for a scenario."""

    name: str
    graph_path: Path
    route_path: Path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=GRAPH_DIR,
        help="Directory containing pickled planning graphs (default: %(default)s)",
    )
    parser.add_argument(
        "--routes-dir",
        type=Path,
        default=ROUTES_DIR,
        help="Directory with reference path CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional override for recorder output directory (defaults to OfflineProblemRecorder)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        help="Subset of scenario names to process (defaults to all discovered graphs)",
    )
    parser.add_argument(
        "--risk-budgets",
        type=float,
        nargs="+",
        default=list(DEFAULT_RISK_BUDGETS),
        help="Risk budgets to sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--risk-levels",
        nargs="+",
        default=list(DEFAULT_RISK_LEVELS),
        help="Risk level labels to sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Requested number of decision epochs per offline problem (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow recorder to overwrite existing CSV artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Verbosity for console logging (default: %(default)s)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help="File path to store generator logs (default: %(default)s)",
    )
    parser.add_argument(
        "--timestamp-label",
        default=None,
        help="Custom suffix appended to every CSV artifact (default: auto-run timestamp)",
    )
    parser.add_argument(
        "--disable-timestamp",
        action="store_true",
        help="Disable automatic timestamp suffixing of output artifacts",
    )
    return parser.parse_args(argv)


def configure_logging(level: str, log_file: Optional[Path]) -> None:
    handlers: List[logging.Handler] = []
    log_level = getattr(logging, level.upper(), logging.INFO)

    if log_file:
        log_path = log_file.expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))

    handlers.append(logging.StreamHandler())

    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s", handlers=handlers)


def discover_scenarios(graphs_dir: Path, routes_dir: Path, filters: Optional[Sequence[str]]) -> List[ScenarioResources]:
    graphs_dir = graphs_dir.expanduser()
    routes_dir = routes_dir.expanduser()
    scenarios: List[ScenarioResources] = []

    desired = set(filters) if filters else None

    for graph_file in sorted(graphs_dir.glob("graph_*.gpickle")):
        scenario_name = graph_file.stem.replace("graph_", "", 1)
        if desired and scenario_name not in desired:
            continue
        route_file = routes_dir / f"reference_path_{scenario_name}.csv"
        if not route_file.exists():
            alt = routes_dir / f"reference_path_{scenario_name.replace(' ', '_')}.csv"
            route_file = alt if alt.exists() else route_file
        if not route_file.exists():
            LOGGER.warning("Skipping %s (missing reference path CSV)", scenario_name)
            continue
        scenarios.append(ScenarioResources(scenario_name, graph_file, route_file))

    if desired:
        missing = desired - {s.name for s in scenarios}
        for scen in sorted(missing):
            LOGGER.error("Requested scenario '%s' not found in %s", scen, graphs_dir)
    return scenarios


def load_graph(resource: ScenarioResources) -> nx.DiGraph:
    graph = _GRAPH_HANDLER.load_graph(resource.graph_path.parent, resource.name)
    if not isinstance(graph, nx.DiGraph):
        raise TypeError(f"Graph '{resource.name}' is not a directed NetworkX graph.")
    return graph


def load_reference_path(path: Path) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as csv_file:
        for line in csv_file:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            if len(parts) < 2:
                continue
            try:
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                LOGGER.debug("Skipping malformed waypoint row: %s", stripped)
    if not points:
        raise ValueError(f"Reference path {path} does not contain any coordinates.")
    return points


def build_goal_indices(graph: nx.DiGraph) -> List[int]:
    layers = sorted({data.get("layer", 0) for _, data in graph.nodes(data=True)})
    if not layers:
        raise ValueError("Graph does not expose layer metadata required for decision epochs.")
    return list(range(len(layers)))


def resolve_risk_levels(labels: Iterable[str]) -> List[RiskLevelSpec]:
    mapping: Dict[str, float] = {
        "low": RiskLevel.LOW,
        "mid": RiskLevel.MEDIUM,
        "medium": RiskLevel.MEDIUM,
        "high": RiskLevel.HIGH,
        "very_high": RiskLevel.VERY_HIGH,
        "veryhigh": RiskLevel.VERY_HIGH,
    }
    resolved: List[RiskLevelSpec] = []
    for label in labels:
        key = label.lower()
        if key not in mapping:
            raise ValueError(f"Unsupported risk level '{label}'. Supported: {sorted(mapping)}")
        resolved.append((label, mapping[key]))
    return resolved


def adjusted_epoch_count(requested: int, goal_indices: Sequence[int]) -> int:
    possible_epochs = max(0, len(goal_indices) - 4)
    max_supported = possible_epochs + 1
    if max_supported < 2:
        raise ValueError("Graph does not have enough layers to host decision epochs.")
    return max(2, min(requested, max_supported))


def aggregate_edge_metrics(candidates: CandidateMapping) -> Tuple[Optional[EdgeMetric], Optional[EdgeMetric], Optional[EdgeMetric]]:
    risk: EdgeMetric = {}
    cost: EdgeMetric = {}
    utility: EdgeMetric = {}
    for candidate_list in candidates.values():
        for candidate in candidate_list:
            risk_matrix = getattr(candidate, "risk_matrix", None)
            cost_matrix = getattr(candidate, "cost_matrix", None)
            utility_matrix = getattr(candidate, "utility_matrix", None)
            if isinstance(risk_matrix, Mapping):
                for key, value in risk_matrix.items():
                    risk.setdefault(key, value)
            if isinstance(cost_matrix, Mapping):
                for key, value in cost_matrix.items():
                    cost.setdefault(key, value)
            if isinstance(utility_matrix, Mapping):
                for key, value in utility_matrix.items():
                    utility.setdefault(key, value)
    return (risk or None, cost or None, utility or None)


def rebuild_decision_timeline(candidate_keys: Iterable[Tuple[int, str, str]], graph: nx.DiGraph) -> Tuple[List[int], List[object]]:
    ordered = sorted(candidate_keys, key=lambda item: item[0])
    if not ordered:
        return [], []
    node_lookup: Dict[str, object] = {getattr(node, "id", str(node)): node for node in graph.nodes}
    node_sequence: List[object] = []
    epoch_layers: List[int] = []
    start_id = ordered[0][1]
    start_node = node_lookup.get(start_id, start_id)
    node_sequence.append(start_node)
    epoch_layers.append(getattr(start_node, "layer", None))
    for _, _, goal_id in ordered:
        goal_node = node_lookup.get(goal_id, goal_id)
        node_sequence.append(goal_node)
        epoch_layers.append(getattr(goal_node, "layer", None))
    return epoch_layers, node_sequence


def run_generation(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    scenarios = discover_scenarios(args.graphs_dir, args.routes_dir, args.scenarios)
    if not scenarios:
        LOGGER.error("No scenarios found. Nothing to do.")
        return

    risk_levels = resolve_risk_levels(args.risk_levels)
    recorder = OfflineProblemRecorder(output_root=args.output_root) if args.output_root else OfflineProblemRecorder()
    timestamp_suffix = None
    if not args.disable_timestamp:
        timestamp_suffix = args.timestamp_label or datetime.now().strftime("%Y%m%d_%H%M%S")

    for scenario in scenarios:
        LOGGER.info("Loading graph for %s", scenario.name)
        graph = load_graph(scenario)
        carla_route = load_reference_path(scenario.route_path)
        goal_indices = build_goal_indices(graph)
        try:
            epochs = adjusted_epoch_count(args.num_epochs, goal_indices)
        except ValueError as exc:
            LOGGER.warning("Skipping scenario %s: %s", scenario.name, exc)
            continue
        LOGGER.debug("Scenario %s -> %d goal indices / %d epochs", scenario.name, len(goal_indices), epochs)

        for risk_budget in args.risk_budgets:
            for risk_label, risk_value in risk_levels:
                LOGGER.info(
                    "Generating offline problem (scenario=%s, budget=%.2f, risk=%s)",
                    scenario.name,
                    risk_budget,
                    risk_label,
                )

                print("Now, starting generation an offline problem: to get all candidates and rb candidates.")
                try:
                    _all_candidates_dict, risk_candidates_dict, optimal_offline_csp = generate_offline_problem(
                        graph=graph,
                        risk_level=risk_value,
                        risk_budget=risk_budget,
                        num_epochs=epochs,
                        goal_indecies=goal_indices,
                        carla_route=carla_route,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOGGER.exception("Failed to generate problem for %s (budget %.2f, level %s): %s", scenario.name, risk_budget, risk_label, exc)
                    continue

                if not risk_candidates_dict:
                    LOGGER.warning("No feasible candidates for %s (budget %.2f, level %s)", scenario.name, risk_budget, risk_label)
                    continue

                print("finished generation, now recording!!!")

                risk_candidates = next(iter(risk_candidates_dict.values()))
                decision_epochs, decision_nodes = rebuild_decision_timeline(risk_candidates.keys(), graph)
                risk_matrix, cost_matrix, utility_matrix = aggregate_edge_metrics(risk_candidates)

                try:
                    recorder.record_problem(
                        scenario.name,
                        risk_matrix=risk_matrix,
                        cost_matrix=cost_matrix,
                        utility_matrix=utility_matrix,
                        decision_epochs=decision_epochs,
                        decision_nodes=decision_nodes,
                        candidates=risk_candidates,
                        overwrite=args.overwrite,
                        timestamp=timestamp_suffix,
                    )
                except FileExistsError as exc:
                    LOGGER.warning("Recorder skipped existing artifacts for %s: %s", scenario.name, exc)
                    continue

                if optimal_offline_csp:
                    print("managed to generate the optimal offline solution!")
                    csp_payload = next(iter(optimal_offline_csp.values()))
                    solution_edges = csp_payload.get("solution_edge_speeds")
                    solution_nodes = csp_payload.get("path_nodes")
                    if solution_edges or solution_nodes:
                        try:
                            recorder.record_solution(
                                scenario.name,
                                solution_edges=solution_edges,
                                path_nodes=solution_nodes,
                                risk_matrix=risk_matrix,
                                cost_matrix=cost_matrix,
                                utility_matrix=utility_matrix,
                                overwrite=args.overwrite,
                                timestamp=timestamp_suffix,
                            )
                        except FileExistsError as exc:
                            LOGGER.warning("Solution artifacts already exist for %s: %s", scenario.name, exc)

                LOGGER.info(
                    "Recorded offline problem for %s (budget=%.2f, risk=%s) with %d candidate groups",
                    scenario.name,
                    risk_budget,
                    risk_label,
                    len(risk_candidates),
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, args.log_file)
    try:
        run_generation(args)
    except Exception as exc:  # pragma: no cover - safety net for CLI usage
        LOGGER.error("Fatal error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
