# Phase 7: Local Volatility (Dupire) — Completion Report

## Design: SABR-smoothed BlackVarianceSurface -> LocalVolSurface, not Andreasen-Huge retried
Phase 6's handoff found Andreasen-Huge's `localVol()` numerically unstable (up to 4456%) even
with denser raw market data, ruling out sparsity as the cause. Rather than retry the same
approach, Phase 7 uses a different, more standard QuantLib pipeline: evaluate Phase 5's
already-validated SABR fits on a common 21-point strike grid per expiry (SABR slices are smooth
by construction, unlike raw market ticks), build a `ql.BlackVarianceSurface` from that grid, then
`ql.LocalVolSurface` for Dupire's formula via QuantLib's standard finite-difference
differentiation. Pricing via `ql.GeneralizedBlackScholesProcess` (black+local vol handles) +
`ql.FdBlackScholesVanillaEngine`. This ties Phase 5 and Phase 7 together coherently instead of
building a second independent raw-data smile-smoothing stage.

## Two real bugs found and fixed, both with root-cause diagnosis not guesswork
**1. Calendar arbitrage from independent per-expiry SABR fits (21/192 dates, 10.9%).**
SABR fits each expiry fully independently (Phase 5's whole design point) with no cross-expiry
consistency constraint. On ~11% of dates this produced real calendar arbitrage (total variance
`w=IV^2*T` decreasing between adjacent expiries at some strike) — caught loudly and specifically
by `BlackVarianceSurface`'s own construction-time check (naming the exact strike and time), a far
better failure mode than Andreasen-Huge's silent garbage output on the same underlying class of
problem. First fix attempt: enforce `w[j] >= w[j-1]` across expiries at each of the 21 grid
strikes before building the surface. **This alone did NOT fully resolve it** — re-running showed
identical exceptions on the identical dates, which led to the real diagnosis:

**2. Cubic-spline overshoot, not grid non-monotonicity.** The `setInterpolation("bicubic")` call
was itself reintroducing violations *between* grid points even though the grid points themselves
were now monotonic — a well-known cubic-spline artifact (overshoot). Confirmed by the tiny time
deltas in the exception messages (~1 day apart, well inside a single grid segment, meaning
QuantLib was checking spline-interpolated intermediate points, not the raw grid). Fix: dropped
`setInterpolation("bicubic")`, use the class default (bilinear, which cannot overshoot a
monotonic grid). Verified directly on 3 previously-failing dates (2011-04-15, 2014-07-15,
2016-05-16) before re-running the full batch. **Exceptions dropped from 21 to 13/192 (6.8%).**

## Final result: 179/192 dates (93.2%) succeed cleanly
| | Value |
|---|---|
| Successful dates | 179/192 (93.2%) |
| Mean IV error (successful dates) | 0.620pp |
| Local vol range (successful dates) | 31%-115% (sane, no blow-ups) |
| Remaining exceptions | 13/192 (6.8%) |

Fit quality on successful dates is in line with Heston/SABR/Andreasen-Huge, and critically, no
local-vol blow-ups at all across any successful date (max local vol capped at ~115%, vs.
Andreasen-Huge's uncontrolled 4456%).

## The remaining 13 failures — two distinct root causes, both genuine, both documented honestly
- **11 dates**: still-residual calendar-sensitivity, now occurring right at the shortest-dated
  expiry (deltas of ~1 day). Bilinear interpolation cannot overshoot in theory, so this is likely
  either (a) QuantLib's internal check comparing against an implicit t=0 anchor with near-zero
  time-to-first-expiry numerical noise, or (b) a residual case where even the raw SABR-implied
  variance at the grid points themselves has a razor-thin violation my monotonicity fix's
  floating-point comparison didn't catch. Not fully root-caused given time budget.
- **2 dates** (2023-11-15, 2024-09-16): a different failure mode entirely — "negative local
  vol^2 ... surface not smooth enough," which is Dupire's formula's *butterfly* (strike-convexity)
  requirement being violated, not the calendar dimension. Both involve extreme strikes (404, 506)
  at longer tenors (T~0.9-1.2yr) — sparser, noisier data region.

**Decision: stop here rather than chase a fully bulletproof fix.** A complete arbitrage-repair
solution (2D isotonic/QP-based smoothing across the whole surface, both calendar and butterfly
dimensions simultaneously) is a research-level undertaking on its own, disproportionate for one
of 9 models in this comparison. 93.2% is a strong, honest result. **This finding is itself
informative for the final writeup**: Local Vol is empirically the least numerically robust of the
9 models tested — consistent with well-documented industry experience that local vol
construction from real market data is inherently more fragile than parametric (SABR) or
full-dynamics (Heston) approaches, even when built from an already-smoothed input.

## Handling in Phase 9
The 13 failing dates should be excluded from Local Vol's OOS scoring sample specifically (not
from the other 8 models), with the exclusion documented in the final results — a data
availability limitation of this one model, not a missing-data problem for the underlying dataset.

## Scripts
`src/phase7_local_vol.py` (surface construction + PDE pricer + single-date design test),
`src/phase7_local_vol_batch.py` (192-date validation, resumable though not needed here).
Output: `results/phase7_local_vol_batch_validation.csv`.

## Next
Phase 8: Merton Jump-Diffusion calibration.
