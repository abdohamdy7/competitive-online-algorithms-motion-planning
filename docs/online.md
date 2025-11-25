# Online Algorithms Overview

The online layer consumes offline artifacts and produces online decisions under a risk budget.

- **Candidates-based**: CZL-ORB, BAT-ORB operate on `*_candidates.csv` and write `*_online_<ALG>.csv` to `results/data/online solutions/candidates`.
- **Graph-based**: ITM-ORB operates on `*_edge_values.csv` + `*_decision_timeline.csv` and produces per-epoch selections (extendable to full paths).

Shared helpers live in `src/motion_planning/online_algorithms/`:
- `load_problem_to_solve.py` for loaders/writers.
- `thresholds.py` for global threshold parameters.
