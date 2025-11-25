# ITM-ORB

ITM-ORB is a graph-based online solver using a MILP.

## Inputs
- Edge metrics: `*_edge_values.csv`
- Decision timeline: `*_decision_timeline.csv`
- Optional graph pickle, total budget $\Delta_0$, and threshold $\psi$.

## MILP (per epoch)
Variables: binary $x_{ij}^\nu$, continuous $\delta_t \in [0, \Delta_t]$.
Constraints:
- Flow conservation from $v_s$ to $v_g$.
- Risk: $\sum x\,r \le \delta_t$.
- Utility: $\sum x\,u \ge \psi\,\delta_t$ (if $\psi>0$).
- One speed per edge: $\sum_\nu x_{ij}^\nu \le 1$.
Objective: maximize $\sum x\,u$.

## Outputs
Per-epoch selections (edges with speeds, $\delta_t$, objective, remaining budget). Extend to full path constraints as needed.
