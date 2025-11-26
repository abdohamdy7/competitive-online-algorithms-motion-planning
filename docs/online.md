# Online Algorithms Overview

The online layer consumes offline artifacts and produces online decisions under a risk budget.

- **Candidates-based**: CZL-ORB, BAT-ORB operate on `*_candidates.csv` and write `*_online_<ALG>.csv` to `results/data/online solutions/candidates`.
- **Graph-based**: ITM-ORB operates on `*_edge_values.csv` + `*_decision_timeline.csv` and produces per-epoch selections (extendable to full paths).

Shared helpers live in `src/motion_planning/online_algorithms/`:
- `load_problem_to_solve.py` for loaders/writers.
- `thresholds.py` for global threshold parameters.

## Correct usage (no fallbacks)

### CZL-ORB (candidates)
- Inputs: a `*_candidates.csv`, total budget `Δ0`, and global `rho_min/rho_max` computed over the full set of candidates (use `czl_thresholds([...])`).
- Call: `run_czl_orb(candidates_csv, capacity_override=Δ0, rho_min=..., rho_max=..., output_root=results/data/online solutions/candidates)`.
- Avoid relying on auto rho bounds from a single file; precompute them globally.

### BAT-ORB (candidates)
- Inputs: a `*_candidates.csv`, total budget `Δ0`, and global `δ_min` (min positive risk) computed over the full set of candidates (use `bat_threshold([...])`).
- Call: `run_bat_orb(candidates_csv, capacity_override=Δ0, delta_min=..., output_root=results/data/online solutions/candidates)`.
- Avoid running without `delta_min`; compute it once from all candidates.

### ITM-ORB (graph)
- Inputs: `*_edge_values.csv`, `*_decision_timeline.csv`, total budget `Δ0`, candidate files in the sibling `problem details` directory (to derive CZL thresholds), optional graph pickle, and Gurobi params (e.g., `GUROBI_OPTIONS`).
- Call: `run_itm_online(edge_values_csv, decision_timeline_csv, capacity=Δ0, psi_t=None, gurobi_params=GUROBI_OPTIONS, graph_pickle_path=..., output_root=results/data/online solutions/graph-based)`.
- The code computes per-epoch `psi` via CZL (`czl_psi`) using `rho_min/rho_max` from the candidates; if candidates are missing, ITM will raise.
- Avoid omitting `capacity` (to skip filename parsing) and ensure candidates exist alongside the graph artifacts.
