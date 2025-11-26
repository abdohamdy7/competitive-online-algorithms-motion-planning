"""
Utilities to persist the ingredients of an offline motion-planning problem.

The module writes three CSV artifacts per problem:
1. Edge metrics (risk / cost / utility) for every (u, v, speed) tuple.
2. Decision timeline (epochs + associated nodes).
3. Candidate paths for every decision epoch.

The intent matches the inline specification left by the user inside this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableSequence, Optional, Sequence, Tuple, Union

import pandas as pd

try:  # Import is optional when the module is reused outside the project.
    from motion_planning.utils.candidate_path import CANDIDATE_PATH
except Exception:  # pragma: no cover - fallback for docs / typing.
    CANDIDATE_PATH = None  # type: ignore

from motion_planning.utils.paths import OFFLINE_RESULTS_DIR

EdgeKey = Tuple[Any, Any, Any]
CandidateDict = Mapping[Any, Union[Sequence[Any], Any]]

DEFAULT_OUTPUT_ROOT = OFFLINE_RESULTS_DIR


@dataclass
class OfflineProblemRecord:
    """Container returned after writing the CSV artifacts."""

    graph_id: str
    graph_edges_path: Optional[Path]
    decision_timeline_path: Optional[Path]
    candidates_path: Optional[Path]
    graph_edges: pd.DataFrame
    decision_timeline: pd.DataFrame
    candidates: pd.DataFrame


@dataclass
class OfflineSolutionRecord:
    """Container for the recorded offline CSP solution details."""

    graph_id: str
    solution_edges_path: Optional[Path]
    solution_nodes_path: Optional[Path]
    solution_edges: pd.DataFrame
    solution_nodes: pd.DataFrame


class OfflineProblemRecorder:
    """Persist graph metrics, decision data, and candidates for offline problems."""

    def __init__(self, output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT) -> None:
        self.output_root = Path(output_root)
        self.graph_metrics_dir = self.output_root / "graphs with risks"
        self.problem_details_dir = self.output_root / "problem details"
        self.solution_details_dir = self.output_root / "solution details"
        for folder in (self.graph_metrics_dir, self.problem_details_dir, self.solution_details_dir):
            folder.mkdir(parents=True, exist_ok=True)

    def record_problem(
        self,
        graph_id: str,
        *,
        risk_matrix: Optional[Mapping[EdgeKey, float]] = None,
        cost_matrix: Optional[Mapping[EdgeKey, float]] = None,
        utility_matrix: Optional[Mapping[EdgeKey, float]] = None,
        decision_epochs: Optional[Sequence[Any]] = None,
        decision_nodes: Optional[Sequence[Any]] = None,
        candidates: Optional[CandidateDict] = None,
        overwrite: bool = True,
        timestamp: Optional[str] = None,
        filename_prefix: Optional[str] = None,
    ) -> OfflineProblemRecord:
        """
        Record the full offline problem definition as CSV files.

        Args:
            graph_id: Unique identifier for the graph (e.g., "Town05_Multi").
            risk_matrix: Mapping {(u, v, s): risk_value}.
            cost_matrix: Mapping {(u, v, s): cost_value}.
            utility_matrix: Mapping {(u, v, s): utility_value}.
            decision_epochs: Ordered epochs when decisions occur.
            decision_nodes: Node IDs aligned with the decision epochs.
            candidates: Mapping keyed by (epoch, start, goal) -> list of candidates.
            overwrite: When False, raise if a target CSV already exists.
            timestamp: Optional suffix appended to filenames for uniqueness.
        """

        stem = self._build_file_stem(graph_id, timestamp=timestamp, filename_prefix=filename_prefix)

        edge_df = self._build_edge_dataframe(graph_id, risk_matrix, cost_matrix, utility_matrix)
        timeline_df = self._build_decision_dataframe(graph_id, decision_epochs, decision_nodes)
        candidates_df = self._build_candidates_dataframe(graph_id, candidates)

        edges_path = self._write_csv(
            self.graph_metrics_dir / f"{stem}_edge_values.csv",
            edge_df,
            overwrite=overwrite,
        )
        timeline_path = self._write_csv(
            self.problem_details_dir / f"{stem}_decision_timeline.csv",
            timeline_df,
            overwrite=overwrite,
        )
        candidates_path = self._write_csv(
            self.problem_details_dir / f"{stem}_candidates.csv",
            candidates_df,
            overwrite=overwrite,
        )

        return OfflineProblemRecord(
            graph_id=graph_id,
            graph_edges_path=edges_path,
            decision_timeline_path=timeline_path,
            candidates_path=candidates_path,
            graph_edges=edge_df,
            decision_timeline=timeline_df,
            candidates=candidates_df,
        )

    def record_solution(
        self,
        graph_id: str,
        *,
        solution_edges: Optional[Sequence[EdgeKey]] = None,
        path_nodes: Optional[Sequence[Any]] = None,
        risk_matrix: Optional[Mapping[EdgeKey, float]] = None,
        cost_matrix: Optional[Mapping[EdgeKey, float]] = None,
        utility_matrix: Optional[Mapping[EdgeKey, float]] = None,
        graph: Optional[Any] = None,
        overwrite: bool = True,
        timestamp: Optional[str] = None,
        filename_prefix: Optional[str] = None,
    ) -> OfflineSolutionRecord:
        """
        Persist the offline CSP solution edges and node sequence.
        """

        stem = self._build_file_stem(graph_id, timestamp=timestamp, filename_prefix=filename_prefix)

        edges_df = self._build_solution_edge_dataframe(
            graph_id,
            solution_edges=solution_edges,
            risk_matrix=risk_matrix,
            cost_matrix=cost_matrix,
            utility_matrix=utility_matrix,
            graph=graph,
            delta=None,  # graph-based offline solution does not track delta per epoch
        )
        nodes_df = self._build_solution_nodes_dataframe(graph_id, path_nodes)

        edges_path = self._write_csv(
            self.solution_details_dir / f"{stem}_offline_graph_solution_edges.csv",
            edges_df,
            overwrite=overwrite,
        )
        nodes_path = self._write_csv(
            self.solution_details_dir / f"{stem}_solution_nodes.csv",
            nodes_df,
            overwrite=overwrite,
        )

        return OfflineSolutionRecord(
            graph_id=graph_id,
            solution_edges_path=edges_path,
            solution_nodes_path=nodes_path,
            solution_edges=edges_df,
            solution_nodes=nodes_df,
        )

    # --------------------------------------------------------------------- #
    # DataFrame builders
    # --------------------------------------------------------------------- #
    def _build_edge_dataframe(
        self,
        graph_id: str,
        risk_matrix: Optional[Mapping[EdgeKey, float]],
        cost_matrix: Optional[Mapping[EdgeKey, float]],
        utility_matrix: Optional[Mapping[EdgeKey, float]],
    ) -> pd.DataFrame:
        keys = set()
        for matrix in (risk_matrix, cost_matrix, utility_matrix):
            if matrix:
                keys.update(matrix.keys())

        rows = []
        for key in sorted(keys, key=lambda x: tuple(str(part) for part in x)):
            start, end, speed = self._normalize_edge_key(key)
            rows.append(
                {
                    "graph_id": graph_id,
                    "start_node": start,
                    "end_node": end,
                    "speed": speed,
                    "risk": None if risk_matrix is None else risk_matrix.get(key),
                    "cost": None if cost_matrix is None else cost_matrix.get(key),
                    "utility": None if utility_matrix is None else utility_matrix.get(key),
                }
            )

        return pd.DataFrame(rows, columns=["graph_id", "start_node", "end_node", "speed", "risk", "cost", "utility"])

    def _build_decision_dataframe(
        self,
        graph_id: str,
        decision_epochs: Optional[Sequence[Any]],
        decision_nodes: Optional[Sequence[Any]],
    ) -> pd.DataFrame:
        epochs_list = self._sequence_to_list(decision_epochs)
        node_list = [self._node_to_id(node) for node in self._sequence_to_list(decision_nodes)]

        max_len = max(len(epochs_list), len(node_list))
        rows = []
        for idx in range(max_len):
            rows.append(
                {
                    "graph_id": graph_id,
                    "index": idx,
                    "decision_epoch": epochs_list[idx] if idx < len(epochs_list) else None,
                    "decision_node": node_list[idx] if idx < len(node_list) else None,
                }
            )

        return pd.DataFrame(rows, columns=["graph_id", "index", "decision_epoch", "decision_node"])

    def _build_candidates_dataframe(
        self,
        graph_id: str,
        candidates: Optional[CandidateDict],
    ) -> pd.DataFrame:
        if not candidates:
            return pd.DataFrame(
                columns=[
                    "graph_id",
                    "epoch",
                    "start_node",
                    "goal_node",
                    "candidate_index",
                    "risk",
                    "cost",
                    "utility",
                    "frenet_progress",
                    "time_progress",
                    "ratio",
                    "num_edges",
                    "path_nodes",
                    "path_speeds",
                ]
            )

        rows = []
        for key, value in candidates.items():
            epoch, start_node, goal_node = self._parse_candidate_key(key)
            candidate_list = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
            for idx, candidate in enumerate(candidate_list):
                rows.append(
                    self._candidate_to_row(
                        graph_id=graph_id,
                        epoch=epoch,
                        start_node=start_node,
                        goal_node=goal_node,
                        candidate_index=idx,
                        candidate=candidate,
                    )
                )

        columns = [
            "graph_id",
            "epoch",
            "start_node",
            "goal_node",
            "candidate_index",
            "risk",
            "cost",
            "utility",
            "frenet_progress",
            "time_progress",
            "ratio",
            "num_edges",
            "path_nodes",
            "path_speeds",
        ]
        return pd.DataFrame(rows, columns=columns)

    def _build_solution_edge_dataframe(
        self,
        graph_id: str,
        *,
        solution_edges: Optional[Sequence[EdgeKey]],
        risk_matrix: Optional[Mapping[EdgeKey, float]],
        cost_matrix: Optional[Mapping[EdgeKey, float]],
        utility_matrix: Optional[Mapping[EdgeKey, float]],
        graph: Optional[Any] = None,
        delta: Optional[float] = None,
    ) -> pd.DataFrame:
        node_lookup: dict[str, Any] = {}
        if graph is not None:
            for node in graph.nodes:
                node_id = self._node_to_id(node)
                if node_id is not None:
                    node_lookup[node_id] = node
        if not solution_edges:
            return pd.DataFrame(
                columns=[
                    "graph_id",
                    "segment_index",
                    "start_node",
                    "end_node",
                    "speed",
                    "risk",
                    "cost",
                    "utility",
                    "frenet_progress",
                    "time_progress",
                    "delta",
                ]
            )
        rows = []
        for idx, edge in enumerate(solution_edges):
            start, end, speed = self._normalize_edge_key(edge)
            frenet = None
            time_prog = None
            if graph is not None and node_lookup:
                u = node_lookup.get(start)
                v = node_lookup.get(end)
                if u is not None and v is not None and graph.has_edge(u, v):
                    attr = graph.edges[u, v]
                    frenet = float(attr.get("frenet_progress", 0.0))
                    time_prog = float(attr.get("time_progress", 0.0))
            rows.append(
                {
                    "graph_id": graph_id,
                    "segment_index": idx,
                    "start_node": start,
                    "end_node": end,
                    "speed": speed,
                    "risk": None if risk_matrix is None else risk_matrix.get(edge),
                    "cost": None if cost_matrix is None else cost_matrix.get(edge),
                    "utility": None if utility_matrix is None else utility_matrix.get(edge),
                    "frenet_progress": frenet,
                    "time_progress": time_prog,
                    "delta": delta,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "graph_id",
                "segment_index",
                "start_node",
                "end_node",
                "speed",
                "risk",
                "cost",
                "utility",
                "frenet_progress",
                "time_progress",
                "delta",
            ],
        )

    def _build_solution_nodes_dataframe(
        self,
        graph_id: str,
        path_nodes: Optional[Sequence[Any]],
    ) -> pd.DataFrame:
        nodes = [self._node_to_id(node) for node in self._sequence_to_list(path_nodes)]
        rows = []
        for idx, node_id in enumerate(nodes):
            rows.append({"graph_id": graph_id, "order": idx, "node_id": node_id})
        return pd.DataFrame(rows, columns=["graph_id", "order", "node_id"])

    # --------------------------------------------------------------------- #
    # Candidate helpers
    # --------------------------------------------------------------------- #
    def _candidate_to_row(
        self,
        *,
        graph_id: str,
        epoch: Optional[Any],
        start_node: Optional[str],
        goal_node: Optional[str],
        candidate_index: int,
        candidate: Any,
    ) -> dict:
        row = {
            "graph_id": graph_id,
            "epoch": epoch,
            "start_node": start_node,
            "goal_node": goal_node,
            "candidate_index": candidate_index,
            "risk": None,
            "cost": None,
            "utility": None,
            "frenet_progress": None,
            "time_progress": None,
            "ratio": None,
            "num_edges": None,
            "path_nodes": None,
            "path_speeds": None,
        }

        if CANDIDATE_PATH is not None and isinstance(candidate, CANDIDATE_PATH):
            row["risk"] = getattr(candidate, "risk", None)
            row["cost"] = getattr(candidate, "cost", None)
            row["utility"] = getattr(candidate, "utility", None)
            row["frenet_progress"] = getattr(candidate, "frenet_progress", None)
            row["time_progress"] = getattr(candidate, "time_progress", None)
            row["ratio"] = getattr(candidate, "ratio", None)
            edges = getattr(candidate, "solution_edge_speeds", None) or []
            row["num_edges"] = len(edges)
            row["path_nodes"] = self._nodes_to_string(self._edge_list_to_nodes(edges))
            row["path_speeds"] = self._speeds_to_string(edge[2] for edge in edges)
            row["start_node"] = row["start_node"] or self._node_to_id(getattr(candidate, "start_node", None))
            row["goal_node"] = row["goal_node"] or self._node_to_id(getattr(candidate, "subgoal_node", None))
            return row

        if isinstance(candidate, Mapping):
            row["risk"] = candidate.get("risk") or candidate.get("path_risk")
            row["cost"] = candidate.get("cost") or candidate.get("path_cost")
            row["utility"] = candidate.get("utility") or candidate.get("path_utility")
            row["frenet_progress"] = candidate.get("frenet_progress")
            row["time_progress"] = candidate.get("time_progress")
            row["ratio"] = candidate.get("ratio")
            row["num_edges"] = candidate.get("num_edges")
            if candidate.get("path"):
                row["path_nodes"] = self._nodes_to_string(candidate["path"])
            elif candidate.get("path_nodes"):
                row["path_nodes"] = self._nodes_to_string(candidate["path_nodes"])
            elif candidate.get("nodes"):
                row["path_nodes"] = self._nodes_to_string(candidate["nodes"])
            if candidate.get("solution_edge_speeds"):
                row["path_speeds"] = self._speeds_to_string(edge[2] for edge in candidate["solution_edge_speeds"])
            elif candidate.get("path_speeds"):
                row["path_speeds"] = self._speeds_to_string(candidate["path_speeds"])
            elif candidate.get("speeds"):
                row["path_speeds"] = self._speeds_to_string(candidate["speeds"])
            row["start_node"] = row["start_node"] or self._node_to_id(candidate.get("start_node"))
            row["goal_node"] = row["goal_node"] or self._node_to_id(candidate.get("goal_node"))
            return row

        # Fallback: best-effort serialization.
        row["path_nodes"] = str(candidate)
        return row

    # --------------------------------------------------------------------- #
    # Generic helpers
    # --------------------------------------------------------------------- #
    def _write_csv(self, path: Path, df: pd.DataFrame, *, overwrite: bool) -> Optional[Path]:
        if df.empty:
            return None
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Pass overwrite=True to replace it.")
        df.to_csv(path, index=False)
        return path

    @staticmethod
    def _sanitize_identifier(graph_id: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in graph_id.strip())
        return cleaned or "graph"

    def _build_file_stem(
        self,
        graph_id: str,
        *,
        timestamp: Optional[str],
        filename_prefix: Optional[str],
    ) -> str:
        if filename_prefix:
            return self._sanitize_identifier(filename_prefix)
        safe_id = self._sanitize_identifier(graph_id)
        suffix = f"_{self._sanitize_identifier(timestamp)}" if timestamp else ""
        return f"{safe_id}{suffix}"

    @staticmethod
    def _normalize_edge_key(key: Any) -> Tuple[str, str, Any]:
        if not isinstance(key, tuple):
            raise ValueError(f"Edge key must be a tuple, received {type(key)}: {key}")
        if len(key) == 2:
            u, v = key
            s = None
        elif len(key) == 3:
            u, v, s = key
        else:
            raise ValueError(f"Edge key must have length 2 or 3, received {len(key)}: {key}")
        return OfflineProblemRecorder._node_to_id(u), OfflineProblemRecorder._node_to_id(v), s

    @staticmethod
    def _node_to_id(node: Any) -> Optional[str]:
        if node is None:
            return None
        if hasattr(node, "id"):
            return str(getattr(node, "id"))
        return str(node)

    @staticmethod
    def _sequence_to_list(sequence: Optional[Sequence[Any]]) -> MutableSequence[Any]:
        if sequence is None:
            return []
        if isinstance(sequence, list):
            return list(sequence)
        if isinstance(sequence, tuple):
            return list(sequence)
        if hasattr(sequence, "tolist"):
            return list(sequence.tolist())
        return list(sequence)

    @staticmethod
    def _parse_candidate_key(key: Any) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        if isinstance(key, tuple):
            epoch = key[0] if len(key) > 0 else None
            start = OfflineProblemRecorder._node_to_id(key[1]) if len(key) > 1 else None
            goal = OfflineProblemRecorder._node_to_id(key[2]) if len(key) > 2 else None
            return epoch, start, goal
        return key, None, None

    @staticmethod
    def _nodes_to_string(nodes: Optional[Iterable[Any]]) -> Optional[str]:
        if nodes is None:
            return None
        node_ids = [OfflineProblemRecorder._node_to_id(node) for node in nodes if node is not None]
        return " > ".join(node_ids) if node_ids else None

    @staticmethod
    def _speeds_to_string(speeds: Optional[Iterable[Any]]) -> Optional[str]:
        if speeds is None:
            return None
        entries = [str(speed) for speed in speeds]
        return " > ".join(entries) if entries else None

    @staticmethod
    def _edge_list_to_nodes(edges: Optional[Iterable[Tuple[Any, Any, Any]]]) -> Sequence[str]:
        if not edges:
            return []
        nodes = []
        last_end = None
        for u, v, _ in edges:
            nodes.append(OfflineProblemRecorder._node_to_id(u))
            last_end = v
        if last_end is not None:
            nodes.append(OfflineProblemRecorder._node_to_id(last_end))
        return [node for node in nodes if node is not None]


def record_offline_problem(
    graph_id: str,
    *,
    risk_matrix: Optional[Mapping[EdgeKey, float]] = None,
    cost_matrix: Optional[Mapping[EdgeKey, float]] = None,
    utility_matrix: Optional[Mapping[EdgeKey, float]] = None,
    decision_epochs: Optional[Sequence[Any]] = None,
    decision_nodes: Optional[Sequence[Any]] = None,
    candidates: Optional[CandidateDict] = None,
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = True,
    timestamp: Optional[str] = None,
) -> OfflineProblemRecord:
    """
    Convenience wrapper around `OfflineProblemRecorder.record_problem`.
    """

    recorder = OfflineProblemRecorder(output_root=output_root)
    return recorder.record_problem(
        graph_id,
        risk_matrix=risk_matrix,
        cost_matrix=cost_matrix,
        utility_matrix=utility_matrix,
        decision_epochs=decision_epochs,
        decision_nodes=decision_nodes,
        candidates=candidates,
        overwrite=overwrite,
        timestamp=timestamp,
    )


__all__ = [
    "OfflineProblemRecorder",
    "OfflineProblemRecord",
    "OfflineSolutionRecord",
    "record_offline_problem",
]
