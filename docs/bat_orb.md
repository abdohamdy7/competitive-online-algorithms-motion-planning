# BAT-ORB

BAT-ORB is a threshold online algorithm for candidates-based problems.

## Threshold
- Compute global $\delta_{\min}$ (minimum positive risk) via `bat_threshold`.
- Per epoch, $\Psi_{bat}(t) = (\Delta_0 / \Delta_t) \ln(1 + \Delta_0 / \delta_{\min})$.

## Policy
- Feasible set: candidates with $R(\tau) \le \Delta_t$ and $\rho(\tau) \ge \Psi_{bat}(t)$.
- Pick feasible candidate with highest $\rho$; if none, return `None`.
- Deduct chosen risk from remaining budget.

## Outputs
Writes selected rows to `*_online_BAT-ORB.csv` under `results/data/online solutions/candidates`.
