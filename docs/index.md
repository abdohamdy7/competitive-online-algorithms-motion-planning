# Competitive Online Algorithms

Welcome to the documentation. This site covers the offline problem generation pipeline and the online algorithms (CZL-ORB, BAT-ORB, ITM-ORB) for motion planning.

Use the navigation to browse topics.

![Framework structure](assets/structure.svg)

## Code Organization
- `src/motion_planning/offline_problems/`: Generate risks, candidates, offline solutions; record CSV artifacts.
- `src/motion_planning/online_algorithms/`: CZL-ORB, BAT-ORB, ITM-ORB and helpers.
- `src/motion_planning/evaluation/`: Competitive ratio and evaluation notebooks.
- `results/data/`: All offline/online artifacts (graphs, problem details, solutions).
- `docs/`: MkDocs documentation.

## Status Checklist
- [x] Generate CARLA scenario from map/town.
- [x] Generate route (reference path) for each scenario.
- [x] Construct lattice graphs for each reference path.
- [x] Generate offline risks/candidates (graph-based and candidates-based).
- [x] Compute offline optimal solutions (graph edges and candidates opt).
- [x] Run online CZL-ORB (candidates).
- [x] Run online BAT-ORB (candidates).
- [x] Run online ITM-ORB (graph).
- [x] Compute competitive ratio (utility/cost objectives).
- [x] Run bulk experiments.
- [x] Visualize results.
- [ ] try different datasets (dataset with low step)

| step  |       cap for candidates                      |
--------------------------------------------------------
|0.5    | epoch_size / graph_size * Delta               |
|1.0    | max ( (epoch_size / graph_size) * Delta, 2.0)    |
|1.0    | max ( (epoch_size / graph_size) * Delta, Delta/num_of_epoch)    |
