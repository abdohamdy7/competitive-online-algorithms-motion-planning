# Offline Problems

## Artifacts
- Edge metrics: `*_edge_values.csv`
- Decision timelines: `*_decision_timeline.csv`
- Candidates: `*_candidates.csv`
- Offline optimal: `*_solution_edges.csv` (graph-based), `*_opt_solution.csv` (candidates-based)

## Generation
Offline problems are generated via `src/motion_planning/offline_problems/Main.py`, which builds risks, candidates, and persists CSV artifacts under `results/data/offline problems/`.
