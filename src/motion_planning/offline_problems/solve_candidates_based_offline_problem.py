from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


def _parse_risk_budget_from_filename(csv_path: Path) -> Optional[float]:
    """
    Expect filenames like `{timestamp}_{scenario}_{budget}_{risk}_{epochs}_candidates.csv`.
    Parse from the right so scenario names with underscores are supported.
    """
    stem_parts = csv_path.stem.split("_")
    if len(stem_parts) < 4:
        return None
    try:
        if stem_parts[-4] == "very":
            return float(stem_parts[-5])
        return float(stem_parts[-4])
    except (TypeError, ValueError):
        return None


def load_candidate_groups_from_csv(
    csv_path: Path | str,
    *,
    capacity_override: Optional[float] = None,
) -> Tuple[List[List[float]], List[List[float]], List[List[Tuple[int, int, int]]], List[str], float, pd.DataFrame]:
    """
    Read the candidates CSV and organize utilities/risks per decision epoch.

    Returns:
        utilities: list over epochs of utility lists.
        risks: list over epochs of risk lists.
        ids: list over epochs of (epoch, candidate_index, row_index) tuples to recover rows.
        decision_nodes: ordered list of start nodes per epoch (first is v_0_0).
        capacity: risk budget parsed from filename unless overridden.
        df: raw dataframe for downstream use.
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Candidates file is empty: {path}")

    if "epoch" not in df.columns or "candidate_index" not in df.columns:
        raise ValueError("Candidates CSV missing required columns 'epoch' and 'candidate_index'.")

    # Sort by epoch then candidate_index for deterministic ordering.
    df = df.sort_values(["epoch", "candidate_index"]).reset_index(drop=False).rename(columns={"index": "row_index"})

    utilities: List[List[float]] = []
    risks: List[List[float]] = []
    ids: List[List[Tuple[int, int, int]]] = []
    decision_nodes: List[str] = []

    epochs = sorted(df["epoch"].unique())
    for epoch in epochs:
        group = df[df["epoch"] == epoch]
        if group.empty:
            continue
        start_nodes = group["start_node"].dropna().unique()
        start_node = start_nodes[0] if len(start_nodes) > 0 else None
        if start_node is not None:
            decision_nodes.append(str(start_node))
        utilities.append(group["utility"].astype(float).tolist())
        risks.append(group["risk"].astype(float).tolist())
        ids.append([(int(epoch), int(row["candidate_index"]), int(row["row_index"])) for _, row in group.iterrows()])

    capacity = capacity_override if capacity_override is not None else _parse_risk_budget_from_filename(path)
    if capacity is None:
        raise ValueError("Risk budget could not be inferred from filename; please pass capacity_override.")

    return utilities, risks, ids, decision_nodes, capacity, df


def solve_offline_mckp_gurobi(
    utilities: Sequence[Sequence[float]],
    risks: Sequence[Sequence[float]],
    capacity: float,
    *,
    ids: Optional[Sequence[Sequence[Any]]] = None,
    mip_gap: Optional[float] = None,
    time_limit: Optional[float] = None,
    threads: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    # NEW: licensing / environment hooks
    env_options: Optional[Dict[str, Any]] = None,   # e.g., your WLS dict
    external_env: Optional[gp.Env] = None,          # reuse an env if you already opened one
    model_name: str = "offline_MCKP",
) -> Dict[str, Any]:
    """
    Offline MCKP with optional Gurobi WLS licensing and environment reuse.

    If `external_env` is given, the model is built on it (and we DO NOT close that env).
    Otherwise, we create a private environment; if `env_options` is provided, it is passed
    to `gp.Env(params=env_options)` (works with WLSACCESSID/WLSSECRET/LICENSEID).
    """

    # --- basic validation ---
    T = len(utilities)
    if T == 0:
        raise ValueError("`utilities` must contain at least one group.")
    if len(risks) != T:
        raise ValueError("`utilities` and `risks` must have the same number of groups.")
    if ids is not None and len(ids) != T:
        raise ValueError("`ids` must have the same shape as `utilities` if provided.")

    N_t: List[int] = []
    for t in range(T):
        if len(risks[t]) != len(utilities[t]):
            raise ValueError(f"Group {t}: utilities and risks lengths differ.")
        if ids is not None and len(ids[t]) != len(utilities[t]):
            raise ValueError(f"Group {t}: ids length does not match utilities.")
        if len(utilities[t]) == 0:
            raise ValueError(f"Group {t} is empty; each group must contain ≥1 item.")
        N_t.append(len(utilities[t]))

    # Fast infeasibility check
    min_risk_sum = sum(min(risks[t][n] for n in range(N_t[t])) for t in range(T))
    if min_risk_sum > capacity + 1e-12:
        return {
            "status": GRB.INFEASIBLE,
            "status_str": "INFEASIBLE (pre-check)",
            "obj_value": None,
            "risk_used": None,
            "selected_indices": None,
            "selected_ids": None,
            "y_values": {},
            "gap": None,
        }

    def _build_and_solve(_model: gp.Model) -> Dict[str, Any]:
        # params
        _model.Params.OutputFlag = 1 if verbose else 0
        if mip_gap is not None:
            _model.Params.MIPGap = float(mip_gap)
        if time_limit is not None:
            _model.Params.TimeLimit = float(time_limit)
        if threads is not None:
            _model.Params.Threads = int(threads)
        if seed is not None:
            _model.Params.Seed = int(seed)

        # variables
        y = {(t, n): _model.addVar(vtype=GRB.BINARY, name=f"y[{t},{n}]")
             for t in range(T) for n in range(N_t[t])}

        # objective: maximize total utility
        _model.setObjective(
            gp.quicksum(utilities[t][n] * y[(t, n)]
                        for t in range(T) for n in range(N_t[t])),
            GRB.MAXIMIZE,
        )

        # capacity
        _model.addConstr(
            gp.quicksum(risks[t][n] * y[(t, n)]
                        for t in range(T) for n in range(N_t[t]))
            <= capacity, name="risk_budget"
        )

        # exactly one per group
        for t in range(T):
            _model.addConstr(gp.quicksum(y[(t, n)] for n in range(N_t[t])) == 1,
                             name=f"one_of_group_{t}")

        # optimize
        _model.optimize()

        status = _model.Status
        status_str = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
        }.get(status, f"STATUS_{status}")

        has_solution = (status == GRB.OPTIMAL) or (_model.SolCount and _model.SolCount > 0)
        y_vals: Dict[tuple, float] = {}
        selected_indices: Optional[List[Optional[int]]] = None
        selected_ids: Optional[List[Any]] = None
        obj_value = None
        risk_used = None
        gap = None

        if has_solution:
            for key, var in y.items():
                y_vals[key] = var.X

            selected_indices = []
            selected_ids = [] if ids is not None else None
            for t in range(T):
                picked = None
                for n in range(N_t[t]):
                    if y_vals[(t, n)] > 0.5:
                        picked = n
                        break
                selected_indices.append(picked)
                if ids is not None:
                    selected_ids.append(ids[t][picked] if picked is not None else None)

            obj_value = float(_model.objVal)
            risk_used = (sum(risks[t][selected_indices[t]] for t in range(T))
                         if selected_indices and None not in selected_indices else None)
            try:
                gap = float(_model.MIPGap)
            except gp.GurobiError:
                gap = None

        return {
            "status": status,
            "status_str": status_str,
            "obj_value": obj_value,
            "risk_used": risk_used,
            "selected_indices": selected_indices,
            "selected_ids": selected_ids,
            "y_values": y_vals,
            "gap": gap,
        }

    # --- Environment handling ---
    if external_env is not None:
        # Use the caller's environment; do not close it here.
        model = gp.Model(env=external_env, name=model_name)
        return _build_and_solve(model)

    # Create a private env (WLS or otherwise) and close it automatically.
    # If you have WLS fields (WLSACCESSID, WLSSECRET, LICENSEID), pass them via `env_options`.
    try:
        if env_options is None:
            with gp.Env() as env:
                model = gp.Model(env=env, name=model_name)
                return _build_and_solve(model)
        else:
            with gp.Env(params=env_options) as env:
                model = gp.Model(env=env, name=model_name)
                return _build_and_solve(model)
    except gp.GurobiError as e:
        return {
            "status": -1,
            "status_str": f"Gurobi error {e.errno}: {e.message}",
            "obj_value": None,
            "risk_used": None,
            "selected_indices": None,
            "selected_ids": None,
            "y_values": {},
            "gap": None,
        }


def solve_candidates_csv(
    csv_path: Path | str,
    *,
    capacity_override: Optional[float] = None,
    env_options: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Convenience helper: read candidates CSV, solve the offline MCKP, and return the raw result.
    """
    utilities, risks, ids, decision_nodes, capacity, df = load_candidate_groups_from_csv(
        csv_path, capacity_override=capacity_override
    )

    result = solve_offline_mckp_gurobi(
        utilities=utilities,
        risks=risks,
        capacity=capacity,
        ids=ids,
        env_options=env_options,
        verbose=verbose,
    )
    result["decision_nodes"] = decision_nodes
    result["epochs"] = sorted(df["epoch"].unique().tolist())
    result["df"] = df
    return result


def write_opt_solution_csv(
    csv_path: Path | str,
    *,
    capacity_override: Optional[float] = None,
    env_options: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    output_path: Optional[Path | str] = None,
) -> Path:
    """
    Solve the offline MCKP for the given candidates file and write a CSV with the
    optimal choice per decision epoch. The new file name mirrors the input but
    ends with `_opt_solution.csv`.
    """
    result = solve_candidates_csv(
        csv_path,
        capacity_override=capacity_override,
        env_options=env_options,
        verbose=verbose,
    )
    df: pd.DataFrame = result["df"]
    selected_ids = result.get("selected_ids")
    if not selected_ids:
        raise RuntimeError(f"No feasible solution for {csv_path}: {result.get('status_str')}")

    # selected_ids is aligned with epochs; each entry is (epoch, candidate_index, row_index)
    chosen_rows: List[pd.Series] = []
    for entry in selected_ids:
        if entry is None:
            continue
        if len(entry) == 3:
            _, _, row_idx = entry
            chosen_rows.append(df.loc[df["row_index"] == row_idx].iloc[0])
        else:
            # Fallback: match by epoch & candidate_index
            epoch, cand_idx = entry[0], entry[1]
            chosen_rows.append(df[(df["epoch"] == epoch) & (df["candidate_index"] == cand_idx)].iloc[0])

    solution_df = pd.DataFrame(chosen_rows, columns=df.columns.drop("row_index").tolist() + ["row_index"])
    # Keep same column order as original (without helper column)
    solution_df = solution_df[df.columns]
    solution_df = solution_df.drop(columns=["row_index"])

    in_path = Path(csv_path)
    if output_path is None:
        stem = in_path.stem.replace("_candidates", "_opt_solution")
        output_path = in_path.with_name(f"{stem}.csv")
    out_path = Path(output_path)
    solution_df.to_csv(out_path, index=False)
    return out_path
