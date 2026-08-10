# Phase 4 Design: Heston Calibration Framework

Tested on 2019-06-21 (and 2019-06-24 for the warm-start check) before any batch run, per the
phase plan. All 5 decisions from `docs/phase4_handoff.md` are now locked with empirical evidence,
not just chosen a priori.

## The 5 decisions, as built

**1. Objective — vega-weighted IV-space squared error.** `sum(vega_i * (model_iv_i -
market_iv_i)^2)` plus the Feller penalty (below). Converts every Heston model price to an
implied vol via `bs_implied_vol` (new shared utility in `src/pricers.py`) before comparing, so
price-space isn't dominating the fit.

**2. Optimizer — 54-point coarse grid (cheap forward pricing only) then L-BFGS-B refine the top
3.** Plain single-seed local optimization risks a bad local minimum — confirmed directly: on
2019-06-24, cold-starting from the grid converged to rho=-0.83, while warm-starting from a
known-good neighboring day's params converged to rho=-0.96 with a *lower* objective (0.0065 vs
0.0076) — i.e. the grid-seeded cold start landed in a real, worse local optimum. Multi-seed
refinement is doing real work, not just insurance.

**3. Feller condition — soft quadratic penalty, not hard constraint.** Final fit on 2019-06-21:
`2*kappa*theta - xi^2 = -0.0003`, mildly violated. Allowed by design; didn't destabilize the fit.

**4. Bounds** — kappa[0.01,20], theta[0.001,4.0], xi[0.01,5.0], rho[-0.999,0.999], v0[0.0001,4.0].
No parameter hit a bound in the final calibration (rho got close, -0.96 vs -0.999, but didn't
pin) — wide enough to avoid the Phase 2 Vasicek-MLE-style degenerate-boundary trap.

**5. Cross-date seeding — warm-start from the prior date's converged params, tested and
validated.** See the compute-cost section below; this decision has a direct, large effect on
whether Phase 4's batch run is hours or minutes.

## Fit quality on 2019-06-21 (SPY spot $294, 45 contracts, 5 expiries, balanced 20 puts/25 calls)

Calibrated: kappa=2.006, theta=0.0345 (18.6% long-run vol), xi=0.372, rho=-0.962, v0=0.0189
(13.7% initial vol, close to the 12.4% ATM-IV seed). Mean |IV error| = 1.86pp, roughly even
between puts (2.03pp) and calls (1.72pp) after fixing a sampling imbalance (see below).

**Real finding, not a bug:** the residual ~1.7-2pp average error is a genuine single-factor
Heston limitation — SPY's smile that day has a steep near-term put skew (27% IV at 15% OTM,
1-month) against a much flatter call wing (~10-12% across strikes), and a single (rho, xi) pair
can't simultaneously match a shape that asymmetric across every expiry. This is exactly the kind
of gap jump-diffusion models exist to close — a useful, expected result to carry into the later
Heston-vs-Merton-Jump comparison (Phase 8), not something to "fix" here.

**One real bug caught and fixed during this phase:** the initial `select_chain()` used
`np.linspace` over the combined sorted-by-moneyness rows, which silently pulled 27 puts vs only
18 calls on the test date (put strikes were simply denser in that moneyness window in the real
data) — this was inflating the call-side fit error asymmetry (was 2-6pp calls vs <1.5pp puts
before the fix) independent of any real model limitation. Fixed by selecting puts/calls
separately per expiry with an explicit split. After the fix, the two sides' errors converged to
comparable magnitude (2.03 vs 1.72pp), isolating the true, smaller, expected model-limitation gap
from what had been a sampling artifact.

## Compute cost — the actual reason this section matters
| | Objective | Time |
|---|---|---|
| Day 1 (2019-06-21) cold-start (grid + 3 refines) | 0.007732 | 243.9s |
| Day 2 (2019-06-24) cold-start | 0.007576 | 247.5s |
| Day 2 warm-start (seeded from Day 1, single refine) | 0.006456 (better fit) | 5.8s |

**42x speedup, and a better fit, not a worse one.** At ~4min/date cold-started, ~180 monthly
batch dates would cost ~12 hours of compute for Heston alone — a real risk to the project's
overall ETA. Warm-started, the same 180 dates cost roughly 1 cold start + 179×5.8s ≈ 21 minutes.
Warm-starting isn't just the cleaner design choice, it's load-bearing for the schedule.

**Fallback for the batch phase (not tested here, flagged for whoever runs it):** if a warm-started
date's objective comes back much worse than the prior date's, that's a regime-break signal (e.g.
a vol-shock day) — fall back to a fresh grid search for that date rather than trusting a stale
warm start blindly.

**Further speedup available but not needed yet:** most of the ~4min cold-start cost is rebuilding
QuantLib `HestonModel`/`Process`/`Engine` objects from scratch on every objective evaluation.
These could be mutated in place via `SimpleQuote.setValue()` instead of reconstructed, likely
cutting cold-start time further — worth doing only if batch-run time becomes a real constraint,
skipped here since warm-starting already gets per-date cost to ~6s.

## Scripts
`src/pricers.py` (added `bs_implied_vol`), `src/phase4_heston_calibration.py` (single-date design
+ fit), `src/phase4_heston_warmstart_test.py` (decision 5 validation). Results:
`results/phase4_heston_single_date_fit.csv`, `results/phase4_heston_params_test.csv`.

## Handoff to batch execution (Sonnet)
Design is locked. Batch phase: run `select_chain` + calibration across ~180 monthly dates
2010-2025, warm-started except the first date of each vendor era (OptionsDX 2010 and
AlphaVantage 2014 — don't warm-start across the vendor boundary, different data quirks), persist
params per date, QA for convergence failures and the regime-break fallback above.
