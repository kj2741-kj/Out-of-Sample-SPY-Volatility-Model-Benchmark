# Phase 9: Out-of-Sample Scoring — Design and Pre-Run Findings

## The OOS design (the crux of the whole project)
Each model is calibrated ONCE at a monthly date `t` (Phases 4-8), then evaluated at `t+h`
**without recalibrating**, against real market quotes at `t+h`. Horizons h = +5/+10/+21 trading
days, which turns the comparison into a **decay curve** — "whose calibration stays valid
longest" — rather than a single snapshot.

This framing matters: Phases 4-8 reported *in-sample* fit, where SABR (15 params/date) trivially
beats Heston (5) and constant-vol (1). That ranking is close to meaningless as a model
comparison. The OOS test is what actually discriminates.

## Scoring grid — resolves the decision open since the Phase 4 plan
The full liquid universe is **10,702,466 contracts** (measured, not estimated) — infeasible to
price with 9 models. Resolution: a fixed grid of 5 moneyness (0.90/0.95/1.00/1.05/1.10) x 3
tenors (~1m/3m/6m), taking the nearest actual listed contract to each point, OTM convention
(puts below spot, calls above). ~15 contracts x 192 dates x 3 horizons ≈ 8,600 scorings per
model. Identical grid for every model by construction, so the comparison differs only by model.

## Gap found and closed while building: BSM family had no calibration
Phase 3 built and cross-validated BSM/Black-76/GBM-MC/CRR-American but never *calibrated* them —
they had no per-date sigma and so weren't scorable. Added `src/phase9_prep_constvol.py`: fits one
constant vol per date using the **same** vega-weighted IV objective and the **same**
`select_chain()` contract set as every other model, so the comparison isn't contaminated by
differing calibration methodology. In IV space with a constant-vol model this reduces to a
closed-form vega-weighted mean IV (exact, instant). Result: sigma 11.3%-50.1% across 2010-2025
(top end = COVID), in-sample error 4.03pp vs 0.29-2pp for the sophisticated models — the
expected naive baseline.

## Pricing convention for the two surface models (deliberate, non-obvious)
Models 7 (AH surface) and 8 (LocalVol) are both priced by **local-vol PDE at the new spot**, not
by reading an implied vol off the t-surface. Reason: model 8's surface IS built from SABR fits,
so reading IV off it would make it mathematically collapse into model 6 (SABR) and the
comparison would be circular. Pricing by PDE tests genuine local-vol *dynamics*, which differ
from a sticky smile. The two models then differ only in how their surface was constructed (raw
market quotes vs SABR-smoothed) — a deliberate, informative pairing.

## Two findings from the smoke test (2 dates, before the full run)

**1. `% inside bid-ask` is near-useless at these horizons — and that is itself the finding.**
Measured SPY spreads on the scoring grid are $0.01-0.09, i.e. **0.6-3% of mid**. Landing inside a
2-cent window with parameters frozen 5-21 trading days earlier is near-impossible for any model.
Smoke test gave 0-8% across all 9 models. This metric (from the original project spec) should be
reported honestly as *non-discriminating at these horizons* rather than presented as a ranking of
near-zeros. RMSE and vega-scaled RMSE carry the signal.

**2. AH_Surface prices only 6-9 of 15 contracts — survivorship bias, must be flagged.**
Direct consequence of the local-vol instability documented in Phase 6. AH drops exactly the
contracts it cannot handle, so its RMSE is computed on an easier subset and is **not directly
comparable** to models scoring all 15. Any table reporting AH's RMSE must show its `n` alongside.
Do not present AH as "winning" on RMSE without that caveat.

**Preliminary smoke-test ordering** (2 dates only, indicative not conclusive): LocalVol 0.271 <
AH 0.323 (see caveat) < SABR 0.341 < Merton 0.345 < Heston 0.378 << BSM/Black76/GBM-MC/CRR
~0.78-0.81. The ~2.3x gap between the sophisticated models and the constant-vol family is the
headline result to confirm at full scale.

## Runtime
8 seconds for 2 calibration dates -> ~13 minutes for all 192. The feared Local-Vol PDE bottleneck
did not materialize at PDE_GRID=100 (surfaces are built once per calibration date and reused
across all three horizons, which is what keeps it cheap).

## Scripts
`src/phase9_oos_scoring.py` (grid builder + metrics), `src/phase9_prep_constvol.py` (BSM-family
calibration), `src/phase9_oos_runner.py` (per-model OOS pricing + main loop, resumable).
Merton params are loaded from its final parquet, falling back to the in-progress checkpoint CSV
so a partially-complete Merton batch never blocks the harness.

## Status
Harness built and smoke-tested. Full run is blocked only on Phase 8 (Merton) finishing its 192
dates, since Phase 9 needs Merton params for every calibration date.
