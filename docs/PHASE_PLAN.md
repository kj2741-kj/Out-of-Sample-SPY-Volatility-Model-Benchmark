# Options Model Validation — Phase Plan

9-model out-of-sample comparison against real SPY option quotes (2010-2025): **BSM, Black-76,
GBM Monte Carlo, American CRR-binomial/Barone-Adesi-Whaley, Heston, SABR, SVI, Local Volatility
(Dupire), Merton Jump-Diffusion**. Discounting from a Vasicek short-rate model fit to FRED
Treasury yields. Scored via RMSE, vega-scaled RMSE, % of predictions landing inside the bid-ask
spread.

Not a literal replication of an existing project — no matching repo exists on
github.com/Aditya8321 (confirmed via GitHub API); independently built from his portfolio site's
project description, then expanded well past it. Locked data range 2010-2025 (his site claims
2008-2025; 2008-09 doesn't exist for free anywhere, so that claim is unverifiable/likely
optimistic).

This also absorbs the old separate "Vol Surface Calibration (SVI + Heston)" idea that used to be
blocked on Bloomberg/OptionMetrics — free SPY data unblocks it, so it's not a standalone project
anymore. Heston is Phase 4 here; SVI (Phase 6) and Local Vol (Phase 7) fill out the rest of that
old idea.

| # | Phase | Model tier | Status |
|---|---|---|---|
| 0 | Vendor IV/greeks methodology replication check | Sonnet | **DONE** |
| 1 | Data pipeline: merge SPY spot, clean daily chains, IV≤0.01 filter, split vendor eras | Sonnet | **DONE** |
| 2 | Vasicek short-rate fit to FRED yields → discount curve | Opus | **DONE** |
| 3 | Baseline pricers: BSM + Black-76 + GBM-MC + American CRR/BAW, cross-validated | Sonnet | **DONE** |
| 4 | Heston stochastic-vol calibration at t0 | Fable 5 (design) → Sonnet (batch run) | **DONE** — 192/192 dates, 188 converged (97.9%) |
| 5 | SABR per-expiry smile calibration (alpha/beta/rho/nu) | Fable 5 (design) → Sonnet (batch run) | **DONE** — 960/960 fits, 18s, 99.9% converged |
| 6 | Arbitrage-free implied-vol surface (Andreasen-Huge, pivoted from raw SVI) | Fable 5 (design) → Sonnet/Opus (batch run + validation) | **DONE** — 0.288pp fit, 192/192, 0 exceptions |
| 7 | Local Volatility (Dupire) — SABR-smoothed surface, PDE pricer | Opus | **DONE** — 179/192 (93.2%) succeed, 0.62pp fit; least robust of 9 models (documented finding) |
| 8 | Merton Jump-Diffusion calibration at t0 | Opus | **DONE** — 192/192, 188 converged; lambda tracks real crises |
| 9 | OOS scoring harness across all 9 models | Opus | **DONE** — ~4,850 scorings, 3 horizons |
| 10 | Writeup: results, model ranking by era | Sonnet | **DONE** — docs/FINAL_REPORT.md |

**Why the American pricer (Phase 3):** SPY options are American-style. Phase 0 found European
BSM has a systematic IV/theta/rho bias vs vendor values and attributed it partly to this. The
CRR/BAW pricer quantifies that early-exercise premium directly instead of leaving it as an
unverified assumption.

**Why SVI + Local Vol (Phases 6-7) instead of just Heston:** gives three genuinely distinct
schools of vol modeling to compare in the writeup — full stochastic dynamics (Heston), parametric
smile fits (SABR vs SVI, two different functional forms for the same job), and a surface-derived
deterministic vol function (Dupire local vol, built on top of the SVI fit). SVI's arbitrage-free
constraint (no calendar/butterfly arbitrage across strikes and expiries) is one of the fiddlier
practical quant problems — comparable to or harder than SABR calibration.

**Rules carried from Phase 0:**
- Drop any row with `implied_volatility <= 0.01` before trusting vega/theta/rho (vendor solver
  artifact on illiquid deep-ITM OptionsDX-era quotes, ~3.5% of that era).
- Keep OptionsDX (2010-13) and AlphaVantage (2014-25) eras analyzed separately throughout —
  different vendors, different quirks (no open_interest pre-2014, different missing-day patterns).
- No recalibration of Heston/SABR/SVI/Merton during the OOS evaluation window — calibrate once at
  t0, score forward.

**Open decision for Phase 9:** scoring all 23.5M option rows across 9 models isn't realistic
compute-wise. Likely resolution: score against a fixed daily moneyness/tenor grid rather than
the full chain. Decide when Phase 9 starts.

**Scope note:** this grew from a 4-model/7-phase idea to 9 models/11 phases over two sessions.
Said plainly rather than downplayed — ETA below reflects the real size now.

ETA at 2026-08-06: ~15-23 sessions / ~44-58 hours for Phases 1-10. Phases 4-7
(Heston/SABR/SVI/Local Vol) together are the long pole. 5-8 weeks steady part-time, longer if
sparser.
