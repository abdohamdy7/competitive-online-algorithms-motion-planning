# CZL-ORB

CZL-ORB is a threshold online algorithm for candidates-based problems.

## Thresholds
- Compute global $\rho_{\min}, \rho_{\max}$ across all candidates via `czl_thresholds`.
- Per epoch, threshold $\Psi_{czl}(t) = ((\rho_{max} e)/\rho_{min})^z (\rho_{min}/e)$ with $z = 1 - \Delta_t/\Delta_0$.

## Policy
- Feasible set: candidates with $R(\tau) \le \Delta_t$ and $\rho(\tau) \ge \Psi_{czl}(t)$.
- Pick feasible candidate with highest $\rho$; if none, return `None`.
- Deduct chosen risk from remaining budget.

## Outputs
Writes selected rows to `*_online_CZL-ORB.csv` under `results/data/online solutions/candidates`.
