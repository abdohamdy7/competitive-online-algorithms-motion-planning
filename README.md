# Competitive Online Algorithms for Motion Planning

This repository implements a full pipeline for **competitive online motion planning** under a **risk budget**. The workflow builds planning graphs from CARLA routes, generates offline problem artifacts (risks, candidates, optimal baselines), and then runs online algorithms (CZL-ORB, BAT-ORB, ITM-ORB) to make per-epoch decisions. The results can be evaluated using competitive ratio and other metrics.

![Framework structure](structure.svg)

**Framework At A Glance**
1. **Graph construction**: build a layered planning graph from a CARLA reference route.
2. **Offline problems**: generate risk/candidate artifacts and compute offline optimal solutions.
3. **Online algorithms**: consume offline artifacts and make decisions under a remaining risk budget.
4. **Evaluation**: compute competitive ratio, regret, travel time, and related metrics.

**Project Structure**
- `src/motion_planning/graph_construction/`: CARLA route loading, lattice graph construction, and graph assets.
- `src/motion_planning/offline_problems/`: offline problem generation, candidate creation, and CSV artifact recording.
- `src/motion_planning/online_algorithms/`: CZL-ORB, BAT-ORB, ITM-ORB, and shared loaders/thresholds.
- `src/motion_planning/constrained_shortest_path/`: MILP solvers (Gurobi) for constrained shortest-path variants.
- `src/motion_planning/risk_assessment/`: risk generation utilities.
- `src/motion_planning/evaluation/`: competitive ratio and evaluation utilities/notebooks.
- `src/motion_planning/ros2/orb_ws/src/`: ROS2 nodes for graph construction, risk assessment, and online planning.
- `results/`: generated outputs (offline artifacts, online solutions, evaluation files).
- `docs/`: MkDocs documentation source (mirrored in `site/` after build).

**How It Works**
1. **Graph construction** uses CARLA maps/routes to create a layered graph with candidate edges and speeds.
2. **Offline generation** builds artifacts like edge metrics, decision timelines, candidates, and offline optimal solutions.
3. **Online solvers** select actions per epoch based on thresholds (CZL/BAT) or a MILP (ITM), updating remaining budget.
4. **Evaluation** scripts compute metrics such as competitive ratio and regret.

**Quickstart**
1. Activate the expected environment:

```bash
conda activate carla915
```

2. Build a graph from a CARLA route:

```bash
python3 -m motion_planning.graph_construction.main_graph_construction
```

3. Generate offline problems (batch automation with CLI options):

```bash
python3 -m motion_planning.offline_problems.Main --help
```

4. Run online algorithms using the generated artifacts (see `docs/online.md` for correct usage).

**Notes And Dependencies**
- Graph construction depends on a running **CARLA** server and installed `carla` Python package.
- Some offline/online solvers use **Gurobi** (see `src/motion_planning/constrained_shortest_path/`).
- Output artifacts are stored under `results/data/` (offline problems, solutions, and online outputs).

**Documentation**
- Render docs locally with:

```bash
mkdocs serve
```

- Build the static site with:

```bash
mkdocs build
```
