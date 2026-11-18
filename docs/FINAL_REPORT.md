# Out-of-Sample SPY Volatility Model Benchmark — Final Report

**Nine option-pricing models, calibrated on real SPY option chains (2010–2025) and compared
out-of-sample.** 23.5M raw contract quotes · 192 monthly calibration dates · 573 out-of-sample
date-horizon cells, 8,583 contract scorings per model (7,828 priced by all eight ranked models).

> **Two audits.** The first (2026-08-09, §3.1) found five cross-phase issues. The second
> (2026-11-18, §3.2) found that the headline statistic was not an RMSE, that the fair subset let
> local volatility skip hard contracts, corrupt quotes in the scoring grid, and a year-boundary
> bug. §5 and §7 carry the corrected figures from `results/phase9_oos_contracts.csv`. The
> per-model results tables in §6 still show the first-audit statistic (n-weighted mean of
> per-cell RMSE) for the calibration and robustness detail; use §5 for any comparison.

> **How to read this document.** §1–§4 are methodology (data, shared infrastructure,
> experimental design). §5 is the comparative results table — the summary. §6 is the core of the
> document: **one self-contained section per model**, each covering the mathematics, the
> implementation, the calibration, that model's own results, and what it teaches. §7–§9 are
> cross-model analysis, caveats, and reproduction. You can read §6.1–§6.9 in any order.

---

## Contents
1. Data and preprocessing
2. Shared infrastructure
3. Out-of-sample experimental design · **3.1 Consistency audit**
4. Metrics
5. Comparative results (summary tables)
6. **The nine models — one section each**
   6.1 Black-Scholes-Merton · 6.2 Black-76 · 6.3 GBM Monte Carlo · 6.4 CRR Binomial (American) ·
   6.5 Heston · 6.6 SABR · 6.7 Andreasen-Huge Surface · 6.8 Local Volatility (Dupire) ·
   6.9 Merton Jump-Diffusion
7. Cross-model analysis
8. Honest limitations
9. Conclusion, build log, reproduction

---

## 1. Data and preprocessing

**Source.** 23,535,388 SPY option contract-quotes, daily full chains, 2010-01-04 → 2025-12-31,
stitched from two vendors: OptionsDX (2010–2013) and Alpha Vantage (2014–2025). Each row carries
bid/ask/mark, volume, open interest, and vendor-supplied implied volatility and Greeks.

**Vendor validation.** Neither vendor documents how their IV/Greeks are computed, so rather than
trust them, European Black-Scholes IV and Greeks were independently recomputed from
`(S, K, T, r, q, mid)` and diffed against the vendor fields on a stratified ~2,800-row sample.
Delta and gamma matched at correlation > 0.99; IV/vega/theta/rho at 0.78–0.98. The residual gap
is explained by SPY options being American-style and by a flat rate/dividend proxy.
**Conclusion: the vendor data is genuine standard-BSM-family output, not fabricated.**

**Cleaning.** SPY spot merged onto every row; derived `T` (year-fraction to expiry), `moneyness`
(K/S), `mid` ((bid+ask)/2), and three quality flags. No rows dropped — flags only, so each phase
chooses its own filter. Stored as 16 yearly partitions.

**Two data artifacts found and handled:**
- ~3.5% of OptionsDX-era rows have `implied_volatility ≤ 0.01` with nonsensical vega (hundreds in
  magnitude) — a vendor solver artifact on illiquid deep-ITM stale quotes. Filtered everywhere.
- 2012-10-29/30 carry option quotes but no SPY close: NYSE was shut for Hurricane Sandy and the
  vendor kept stale rows. They drop out naturally wherever spot is required.

**Era separation.** The two vendor eras are analysed separately throughout. This mattered — §7.

---

## 2. Shared infrastructure

Everything here is common to all nine models, so the comparison differs **only** by model and
never by calibration methodology.

**Discount curve.** A Vasicek short-rate model `dr = a(b − r)dt + σ_r dW` fit to real FRED
Treasury yields (DGS3MO/1/2/3), producing a daily discount curve to 3 years — extended from the
planned 2 years after confirming SPY LEAPS reach a full 3-year tenor (2.5–5% of rows/year sit
beyond 2yr).

*Finding:* time-series MLE for `(a,b)` is degenerate — daily 3-month yield autocorrelation is
0.9999 (near unit-root) with only ~1.5 rate cycles in 15 years. Switched to fitting `(a,b)`
against the average 1y/2y/3y term-structure shape. Even then only the **product** `a·b` is
identified (verified by relaxing bounds: very different `(a,b)` pairs give near-identical fit
while `a·b` stays ~0.0021–0.0023) — the classic Vasicek short-end identifiability limit.
**What is validated is the discount curve itself**, the only thing downstream pricing consumes:
1yr corr 0.990 / RMSE 0.29pp · 2yr 0.973 / 0.50pp · 3yr 0.954 / 0.66pp.

**Dividend yield.** Flat 1.4% proxy (SPY trailing average), applied identically to every model.

**Contract selection.** One function, `select_chain()`, used by every calibration: moneyness
0.85–1.15, T 0.08–2.0yr, OTM convention (puts below spot, calls above), 5 expiries × 9 strikes ≈
45 contracts/date. *Bug found and fixed:* the first version `linspace`d over combined sorted rows
and silently pulled 27 puts vs 18 calls, biasing fits toward the put wing.

**Calibration objective.** Vega-weighted squared error in **implied-volatility space**:

```
minimise   Σ_i  vega_i · ( IV_model(K_i, T_i) − IV_market(K_i, T_i) )²
```

IV-space rather than price-space because price-space overweights deep-ITM contracts carrying
little smile information; vega-weighted because low-vega contracts carry little information at all.

**Calibration dates.** 192 — one per month (nearest trading day to the 15th), 2010–2025.

---

## 3. Out-of-sample experimental design

Every model is calibrated **once** at date `t`, then used to price real contracts at `t+5`,
`t+10`, `t+21` trading days **with parameters frozen**, scored against actual market quotes then.

This is the point of the project. In-sample fit mostly measures parameter count — SABR's 15
parameters/date beat Heston's 5 beat one constant vol, which tells you almost nothing about model
quality. The OOS test asks *whose calibration is still pricing the market correctly weeks later.*

**Scoring grid.** The liquid universe is 10,702,466 contracts — infeasible to price nine ways.
Fixed grid instead: 5 moneyness (0.90/0.95/1.00/1.05/1.10) × 3 tenors (~1m/3m/6m), nearest real
listed contract to each point, OTM convention. ~15 contracts × 192 dates × 3 horizons ≈ 8,600
scorings per model, on an **identical grid for every model**.

**Pricing convention for the two surface models.** Models §6.7 and §6.8 are priced by local-vol
PDE **at the new spot**, not by reading an implied vol off the frozen surface. Reason: §6.8's
surface *is* built from SABR, so reading IV off it would collapse it mathematically into §6.6 and
the comparison would be circular. PDE pricing tests genuine local-vol dynamics.

## 3.1 Consistency audit

Once all ten phases were complete, the pipeline was audited end-to-end for cross-phase consistency
— specifically for places where two phases could disagree about a shared quantity without either
one erroring. Five issues surfaced. All are fixed; every result in this report post-dates the fix.
The audit script `src/audit_consistency.py` quantifies issues 1–2 and should be re-run after any
change to the rate or forward conventions.

**1. SABR forward mismatch (severe, and the reason the audit was worth doing).** SABR was
*calibrated* with `forward = spot` but *scored* with the dividend-adjusted forward
`F = S·e^{(r−q)T}` — the model was fit to one contract definition and tested against another. Mean
price impact **$0.58, or 62% of SABR's entire RMSE**, max $3.69. Worse, the error **flipped sign
across rate regimes**: near-zero rates 2010–21 made `F ≈ S` and hid it almost completely, while the
2022–25 hiking cycle opened it up. A bug that is invisible for two-thirds of the sample and then
appears is exactly the kind an in-sample fit check cannot catch. Fixed by using the true
dividend-adjusted forward everywhere; SABR recalibrated (now 960/960 converged, up from 959) and
the Local Vol surfaces rebuilt, since §6.8 is constructed from SABR.

**2. Split rate convention.** The scalar-`r` models (BSM, Black-76, GBM-MC, CRR, SABR, Merton) were
scored against a flat 3-month rate while the curve models (Heston, Andreasen-Huge, Local Vol) used
the full Vasicek term structure — so models were being compared under different rate assumptions.
Small in magnitude (0.7–5.6bp, roughly $0.02–0.10) but a direct violation of §2's premise that the
comparison differs only by model. Fixed with a single `tenor_rate()` helper in
`phase9_oos_runner.py`; every model now sees the identical curve.

**3. A crash that silently cost 4 dates.** Phase 9 preloaded the year partition containing `t+45d`.
For December-2025 calibration dates that is `spy_options_clean_2026.parquet`, which does not exist,
raising `FileNotFoundError` inside the loop. **This is why the first run reported 188 of 192
dates** — originally and wrongly attributed to missing calibration parameters. Fixed with an
existence guard. The lesson is the misattribution, not the bug: a plausible explanation for missing
data was accepted without checking it.

**4. Unescaped `\v` in a path string.** `phase5_sabr_batch.py` contained `"...\data\vasicek..."`,
where `\v` is a vertical-tab escape. Caught before it could do damage. Self-inflicted, recorded for
completeness.

**5. A crash hidden behind a shell redirect.** A resumed Phase 9 run appends dates as
`YYYY-MM-DD 00:00:00` while the original rows have no time component. `pd.to_datetime` infers a
format from the first rows and then **raises** on the rest, so Phase 10 died — but its output was
redirected to a file, so it left behind a **stale tables file that looked entirely valid**. Fixed
with `format="mixed"`. General lesson: a crash behind `>` is indistinguishable from success unless
the exit code is checked.

**What changed as a result.** The ranking is **unchanged** and the headline survives intact — smile
models still roughly halve constant-vol error. Every number moved slightly because the 4 recovered
dates are now included. The one ordering change worth noting is that the SABR–Heston gap **widened**
rather than flipped: SABR improved 0.937 → 0.930 while Heston worsened 0.950 → 0.960, taking the
gap from 0.013 to 0.030. That ordering is now better supported than before the audit, not weaker.

---

## 3.2 Second audit (2026-11-18)

Run with the logic-audit checklist: read the code end to end, test each claim against an
independent reference, rerun. Phase 9 was rerun from scratch afterwards, now saving every
contract price (8,583 scorings per model) to `results/phase9_oos_contracts.csv`.

1. **The headline "RMSE" was not an RMSE.** `phase10_analysis.weighted()` took an n-weighted mean of
   per-cell RMSEs. Pooled RMSE, `sqrt(mean of squared errors)`, is larger for every model (BSM 2.25
   vs 1.75 on the old data) and changed the order of the middle three. Now reported throughout.
2. **The fair subset was defined by cell presence, not by contract.** LocalVol was "present" in
   cells where it priced fewer contracts: 182 skipped, concentrated in hard cells (BSM's error
   there 2.97 vs 1.61 elsewhere). The like-for-like set is now contracts priced by all eight ranked
   models. This is the Andreasen-Huge failure mode (§6.7) on a smaller scale, missed by the same
   row-level check.
3. **Corrupt quotes in the scoring grid.** Three quotes below intrinsic value (2019) and, on
   2019-02-14, call prices carried in put rows (a 247 put at $28.05 next to a 249 put at $0.40,
   SPY 274), each scored against every model. Fixed with an American no-arbitrage bound and a
   vertical-spread monotonicity filter (`phase9_oos_scoring.py`); 0.067% of candidate quotes
   removed, 8 grid contracts replaced. 2019 pooled RMSE for LocalVol fell from 2.53 to 0.54.
4. **Year-boundary bug.** Future scoring dates were searched only within the calibration year, so
   every December 21-day horizon was skipped. 18 cells recovered (555 to 573).
5. **Shared error floor (design limit, not fixed).** Calibration targets vendor IVs; scoring uses
   mids under this project's rate and flat dividend. Black-Scholes at each contract's own same-day
   vendor IV misses the mid by 0.485 pooled (2014-25: 0.560, 2010-13: 0.142). Recalibrating every
   model to IVs implied from mids would remove it.
6. **Calibration rates (design limit, not fixed).** The first audit aligned rates at scoring only;
   SABR, Merton and constant-vol calibration still use the flat 3-month rate (up to about 6 bp
   away from the curve at scored tenors).
7. **Reproducibility.** Data files written by pyarrow 24 could not be read by the installed
   pyarrow 19; rewritten, verified identical through DuckDB, originals in `backups/`.
8. **Claim corrections.** "Behind 62% of its RMSE" for the SABR forward bug measured the mean price
   shift ($0.55-0.58 on a 45-point probe) relative to RMSE, not the share of error it caused; the
   first audit's own figures show SABR's OOS statistic moving only 0.937 to 0.930 after the fix.
   "Smile models halve the error" holds for median error, not for pooled RMSE (about a third).

Tests: `tests/test_scoring.py`, 13 checks; the 2019-02-14 check fails on the pre-fix run.

## 4. Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| **RMSE** | `√mean( (model_price − market_mid)² )` | Dollar pricing error. Primary metric. |
| **vega-RMSE** | `√mean( ((model_price − mid)/vega)² )` | Error in implied-vol units; normalises across contracts of very different sensitivity. |
| **% in spread** | `% of predictions with bid ≤ model ≤ ask` | Intended as a tradability check. **Found to be non-discriminating** — see §8. |

---

## 5. Comparative results (summary)

All figures are pooled RMSE in dollars on the 7,828 contracts priced by all eight ranked models
(corrected statistic, §3.2), unless stated.

### Headline

| Model | Pooled RMSE ($) | Median abs error ($) | vs Black-Scholes (pooled) |
|---|---|---|---|
| Local Volatility (Dupire) | **0.864** | **0.254** | -54% |
| Merton Jump-Diffusion | 1.233 | 0.427 | -35% |
| Heston | 1.256 | 0.391 | -33% |
| SABR | 1.290 | 0.342 | -32% |
| GBM Monte Carlo | 1.879 | 0.948 | -1% |
| CRR American | 1.883 | 0.968 | 0% |
| Black-Scholes-Merton | 1.889 | 0.961 | |
| Black-76 | 1.889 | 0.961 | |
| *Reference: Black-Scholes at each contract's own same-day vendor IV* | *0.485* | *0.146* | *floor, not a model* |

7,828 contracts priced by all eight ranked models (91.2% of 8,583 scorings, 573 date-horizon
cells, 192 calibration dates). Andreasen-Huge prices 50.5% of the grid, the hard half missing,
and is excluded (report section 6.7).

Uncertainty, bootstrapping calibration dates (2,000 resamples): LocalVol 0.66-1.12, SABR
1.02-1.60, Heston 1.03-1.51, Merton 1.06-1.41, BSM 1.74-2.04. LocalVol beats every other model in
all resamples. SABR, Heston and Merton differences all straddle zero: **statistically tied**.
Reductions against BSM: LocalVol 43-63%, Merton 29-41%, Heston 23-42%, SABR 19-43%.

Head to head on the same contracts: LocalVol beats SABR on 62% of contracts and 70% of cells, BSM
on 85% and 98%; Merton beats BSM on 73% and 95%.

Each model on its own coverage (not comparable across models with gaps): LocalVol 0.864 on 91.3%,
Andreasen-Huge 1.285 on 50.5% (other models on AH's contracts: LocalVol 0.851, SABR 1.653,
BSM 2.187).

### Decay by horizon

| Model | h=5d | h=10d | h=21d | Degradation |
|---|---|---|---|---|
| Local Vol | 0.779 | 0.813 | 0.992 | +27% |
| SABR | 1.017 | 1.369 | 1.453 | +43% |
| Heston | 1.022 | 1.315 | 1.408 | +38% |
| Merton | 1.067 | 1.290 | 1.330 | +25% |
| BSM | 1.780 | 1.911 | 1.975 | +11% |
| *Reference floor* | *0.477* | *0.490* | *0.488* | |

### Era split

| Model | OptionsDX 2010-13 | AlphaVantage 2014-25 |
|---|---|---|
| Local Vol | 0.326 | 0.989 |
| SABR | 0.490 | 1.476 |
| Heston | 0.508 | 1.434 |
| Merton | 0.504 | 1.406 |
| BSM | 0.842 | 2.145 |
| *Reference floor* | *0.142* | *0.560* |

Worst years for every model: 2020 (COVID), 2022 and 2025. Full per-year table in
`results/phase10_summary_tables.txt`.

---

# 6. The nine models

---

## 6.1 Black-Scholes-Merton (BSM)

### Theory
```
dS/S = (r − q) dt + σ dW          σ constant

C = S e^{−qT} N(d₁) − K e^{−rT} N(d₂)
d₁ = [ ln(S/K) + (r − q + σ²/2)T ] / (σ√T),   d₂ = d₁ − σ√T
```
One volatility for all strikes and maturities. Cannot, by construction, reproduce a smile or skew.

### Implementation
`src/pricers.py::bs_price` — hand-written; the formula is short and unambiguous.

### Calibration
One constant σ per date. In IV space a constant-vol model has `IV_model ≡ σ` for every contract,
so the vega-weighted least-squares solution is exactly the **vega-weighted mean of market IVs** —
solved in closed form, no optimizer (`src/phase9_prep_constvol.py`).

*This was a genuine gap:* the pricer existed from Phase 3 but was never calibrated, so the model
wasn't scorable until Phase 9. Caught while building the scoring harness.

### Results
| | |
|---|---|
| Fitted σ range | 11.3% – 50.1% (top = COVID 2020) |
| In-sample IV error | **4.03pp** |
| **OOS RMSE** | **1.752** (rank 7 of 9) |
| Decay h=5 → 21 | 1.572 → 2.017 (+28%) |
| Era | 0.832 (2010–13) vs 2.052 (2014–25) |

### What it teaches
This is the **baseline the whole project is measured against**. Its 4.03pp in-sample error is not
a failure — it is the honest cost of assuming a flat volatility surface, and quantifying that cost
is precisely why it is included. Every smile-aware model cuts its OOS error by a third or more
(pooled RMSE, §5).

Its **slowest percentage decay (+28%)** is a subtle point: it is already so wrong at 5 days that
three more weeks barely worsens it proportionally. Slow decay is not a virtue here.

---

## 6.2 Black-76

### Theory
The same model reparametrised on the **forward** rather than spot, `F = S e^{(r−q)T}`:
```
C = e^{−rT} [ F N(d₁) − K N(d₂) ],    d₁ = [ ln(F/K) + σ²T/2 ] / (σ√T)
```

### Implementation
`src/pricers.py::black76_price`, written **independently** of `bs_price` — deliberately, so that
cross-validating the two is a genuine check of both implementations rather than one formula
pasted twice.

### Calibration
Identical σ to §6.1 (it is the same model).

### Results
| | |
|---|---|
| **Cross-validation vs BSM** | **max abs difference 5 × 10⁻¹⁴** (machine precision) |
| Put-call parity check | max abs difference 2 × 10⁻¹⁴ |
| **OOS RMSE** | **1.752** — identical to BSM to 3 decimals |
| Decay | 1.572 → 2.017 (+28%) |

### What it teaches
Its value is **validation, not competition**. Black-76 agreeing with BSM to 14 decimal places
in-sample and to 3 decimals out-of-sample proves the pricing plumbing, the discount curve, and the
dividend handling are all internally consistent. A discrepancy here would have invalidated
everything downstream. Reporting it as a separate "model" is honest: it is the same model, and the
result shows exactly that.

---

## 6.3 GBM Monte Carlo

### Theory
Same dynamics as §6.1, priced by simulation instead of closed form:
```
S_T = S₀ exp( (r − q − σ²/2)T + σ√T · Z ),   Z ~ N(0,1)
price = e^{−rT} · mean( payoff(S_T) )
```
**Terminal-value sampling** — exact for GBM, so there is no need to simulate the path for a
European payoff. Antithetic variates (`Z` and `−Z`) halve the variance for free.

### Implementation
`src/pricers.py::gbm_mc_price`. 100k paths in cross-validation, 20k at scoring time.

### Calibration
Identical σ to §6.1.

### Results
| | |
|---|---|
| **Convergence vs BSM** (in std-error units) | mean \|z\| = 0.44, max 1.48, **0 of 982 grid points exceeded \|z\| > 4** |
| **OOS RMSE** | **1.742** (rank 5 — marginally best of the constant-vol family) |
| Decay | 1.564 → 2.005 (+28%) |

### What it teaches
Confirms the Monte Carlo engine is **statistically unbiased** — its deviations from the analytic
BSM price are pure sampling noise, distributed exactly as theory predicts. That matters beyond
this model: it means MC machinery can be trusted for models with no closed form.

Its tiny edge over BSM (1.742 vs 1.752) is **MC noise, not skill** — the two are the same model,
and the gap is well inside simulation error. Reading it as "Monte Carlo beats Black-Scholes" would
be wrong.

---

## 6.4 CRR Binomial (American)

### Theory
Cox-Ross-Rubinstein tree:
```
u = e^{σ√dt},  d = 1/u,  p = (e^{(r−q)dt} − d)/(u − d)
V_node = max( e^{−r dt}[ p·V_up + (1−p)·V_down ],  intrinsic )   ← early-exercise check
```
The only model here that prices **American exercise**, which SPY options genuinely have.

### Implementation
`src/pricers.py::crr_binomial_price`, 200–400 steps, `is_american` toggle.

*Also implemented:* Barone-Adesi-Whaley as a fast closed-form American approximation. It was
**found to degrade badly** — 52% error on deep-OTM, long-dated, low-vol puts. Root-caused rather
than hand-waved: at σ=10% the BAW exponent `q₁ ≈ −12.1`, and raising `(S/S*)` to that power makes
the price a hair-trigger function of BAW's *approximate* exercise boundary. CRR is the American
pricer of record; **BAW is excluded from the ranking**.

### Results
| | |
|---|---|
| American ≥ European | **0 violations / 1,008 grid points** |
| American ≥ intrinsic | **0 violations / 1,008 grid points** |
| Convergence to BSM | 0.035% median relative error |
| **OOS RMSE** | **1.750** (rank 6) |
| Decay | 1.569 → 2.016 (+28%) |

### What it teaches
The early-exercise premium on SPY is **economically negligible** — CRR-American lands within
0.002 of European BSM out-of-sample. That is the correct answer for a low-dividend index (early
exercise is rarely optimal without a large dividend), and it settles a question Phase 0 raised
when it attributed part of the vendor-vs-model IV gap to American-style exercise: that explanation
was largely wrong, and this model is what disproved it.

The no-arbitrage checks passing with **zero violations** across a 1,008-point grid is the
strongest structural validation in the project.

---

## 6.5 Heston (stochastic volatility)

### Theory
Volatility itself is stochastic and mean-reverting:
```
dS/S = (r − q) dt + √v dW_S
dv   = κ(θ − v) dt + ξ√v dW_v,        corr(dW_S, dW_v) = ρ
```
Five parameters: `κ` (mean-reversion speed), `θ` (long-run variance), `ξ` (vol-of-vol),
`ρ` (spot-vol correlation, negative for equity skew), `v₀` (initial variance).
Priced by Fourier inversion of the characteristic function.

### Implementation
QuantLib `HestonProcess` → `HestonModel` → `AnalyticHestonEngine` — production C++ rather than
hand-rolled quadrature.

### Calibration (design locked empirically, not assumed)
- **Objective:** vega-weighted IV-space SSE (§2).
- **Optimizer:** 54-point coarse grid pre-screen → L-BFGS-B refine of the top 3 seeds. Multi-seed
  is doing real work: on one date the cold grid converged to ρ = −0.83 while a warm start reached
  ρ = −0.96 with a *lower* objective — a genuine bad local minimum.
- **Feller condition** `2κθ > ξ²`: **soft quadratic penalty, not a hard constraint.** Real market
  calibrations often sit across the line; the final fit violates it mildly (−0.0003) with no
  instability.
- **Warm-starting** from the prior date, except across the vendor-era boundary.

**Warm-starting was load-bearing, not cosmetic:** cold start 244 s/date vs warm **5.8 s/date** — a
**42× speedup that also produced a better fit** (obj 0.0065 vs 0.0076). Without it this phase alone
would have taken ~12 hours instead of ~20 minutes of compute.

### Results
| | |
|---|---|
| Dates calibrated | **192/192, 188 converged (97.9%)** |
| In-sample IV error | 1.01pp (OptionsDX) / 2.04pp (AlphaVantage) |
| **OOS RMSE** | **0.960** (rank 3) |
| Decay h=5 → 21 | 0.730 → 1.282 (+75%) |
| Era | 0.432 vs 1.132 |

**Parameter behaviour validated against real history:** ρ stayed strongly negative throughout
(median −0.96) — the correct equity-skew signature. Long-run and initial vol tracked actual market
regimes: elevated 2010–12, calm 2013–19, spiking in 2020. On **2020-03-16 the fit gave
v₀ = 0.749 → 86.5% initial vol**, essentially exactly where VIX traded intraday during the COVID
crash. Nothing told it that; it came out of the option prices.

### What it teaches
Full stochastic-vol dynamics buy a large accuracy gain over constant vol (0.960 vs 1.752) with
only 5 parameters — far more parsimonious than SABR's 15, for near-identical OOS accuracy.

**Honest caveat:** κ clusters bimodally near {2.0, 5.0} rather than varying smoothly — partly the
coarse seed grid, partly genuine weak identification of κ from vanilla options alone. Do not read
κ as a precise mean-reversion-speed estimate. The *time series* of the parameters is trustworthy;
individual κ values are not.

---

## 6.6 SABR

### Theory
Stochastic-α-β-ρ, a smile model for the **forward**:
```
dF = α F^β dW₁,    dα = ν α dW₂,    corr(dW₁, dW₂) = ρ
```
Hagan's closed-form asymptotic expansion maps `(α, β, ρ, ν)` directly to implied vol. Fit
**per expiry** — a separate smile slice for each maturity, which is SABR's defining feature.
3 free parameters × 5 expiries = **15 per date**.

### Implementation
`src/sabr.py` wrapping QuantLib's `ql.sabrVolatility` — used directly rather than transcribed by
hand, because Hagan's formula has enough terms that a transcription error is a real risk.

### Calibration
- **β fixed at 1.0** (lognormal), not jointly fitted. α and β trade off and are poorly identified
  from a single smile slice; fixing β a priori is standard practice, including in Hagan's own
  paper. β = 1 suits an equity index better than β = 0 (more common for rates).
- Same vega-weighted IV objective and same contract set as every other model.
- **No warm-starting needed** — each fit is a small, well-conditioned 3-parameter closed-form
  optimization.
- **Forward definition** `F = S·e^{(r−q)T}`, used identically in calibration and in scoring. The
  original code calibrated on `F = S` and scored on the true forward; that mismatch accounted for
  62% of SABR's RMSE and is audit issue 1 in §3.1. It is called out here because SABR is the model
  it distorted, and because near-zero rates concealed it for two-thirds of the sample.

### Results
| | |
|---|---|
| Fits | **960/960 (192 dates × 5 expiries), 100% converged** post-audit recalibration |
| Runtime | **18 seconds for the entire batch** — zero failures, no recovery infrastructure needed |
| In-sample IV error | 0.505pp (0.17 OptionsDX / 0.62 AlphaVantage) |
| **OOS RMSE** | **0.930** (rank 2) |
| Decay h=5 → 21 | 0.692 → 1.248 (+80%) |
| Era | 0.410 vs 1.100 |

**Sanity checks passed:** `ν` (vol-of-vol) decreases monotonically with maturity (2.66 → 1.19),
the textbook SABR term structure — short-dated smiles curve more sharply. `ρ` negative every
single year (−0.67 to −0.84), the correct equity skew sign.

### What it teaches
**The reliability standout of the project.** 960 fits in 18 seconds with zero failures, against
Heston's ~19 background-kill recoveries and hours of compute. Closed-form pricing with no
numerical integration is categorically more robust than characteristic-function or PDE methods —
an operational finding that shaped every later phase's design choice.

Its tighter *in-sample* fit than Heston (0.505pp vs ~1.5pp) is **not evidence it is a better
model** — it spends 15 parameters per date against Heston's 5. Out-of-sample the two are nearly
tied (0.930 vs 0.960), which is the honest comparison and exactly why the OOS design exists.

---

## 6.7 Andreasen-Huge Surface — *excluded from the ranking*

### Theory
Rather than assume a parametric form, Andreasen-Huge (2011) calibrates a **discrete implied-vol
surface** via a one-step implicit finite-difference scheme, **guaranteed arbitrage-free by
construction** (no butterfly or calendar violations), and directly yields a local-vol surface.

### Implementation
QuantLib `AndreasenHugeVolatilityInterpl`.

**Design pivot, documented not silent.** The original plan was hand-rolled raw SVI with Durrleman
no-arbitrage constraints. Pivoted after §6.6 demonstrated the reliability gap between
QuantLib-native methods and hand-rolled numerics. Andreasen-Huge achieves the same goal
(arbitrage-free surface) in one tested implementation and produces the local-vol surface §6.8
needs as a by-product.

**Two API bugs found by reading QuantLib's own test-suite source rather than guessing:**
1. Its `CalibrationSet` quote must be **implied volatility, not price**. Passing price produced
   $147 errors on sub-$20 options and "local vol" in the millions of percent.
2. The error-tuple accessors are **methods, not properties** (SWIG wrapping quirk).

### Results
| | |
|---|---|
| In-sample fit | **0.288pp** — best in-sample of any model |
| Batch | 192/192 dates, **0 exceptions**, 11.9 min |
| Apparent OOS RMSE | 0.878 |
| Apparent vega-RMSE | 2.08 — *apparently 10× better than every other model* |
| **Contracts priced per scoring row** | **7.59 of ~15** (4,211 of 8,315) — vs ~15 for every other model |

### Why it is excluded
It silently fails on roughly half the grid, and **the contracts it fails on are precisely the hard
ones** — deep-wing and short-dated, where its local-vol function was already found unstable
(values to 4456%). Its spectacular vega-scaled figure exists *because* the low-vega contracts that
inflate that metric are the ones missing. Its RMSE is computed on a systematically easier subset
and is not comparable.

**Method note worth recording.** The first fairness check — restricting to cells where all models
were present — **passed AH at 514/514 and would have let this through**. That filter tests whether
a model produced a *row*, not whether it priced all *contracts within* the row. Only the per-row
contract count exposed it. Publishing the headline number without that second check would have
produced a materially wrong conclusion.

### What it teaches
A model that **fails silently is more dangerous than one that fails loudly.** Andreasen-Huge
returned plausible-looking numbers for the contracts it could handle and simply omitted the rest;
the omission, not the numbers, was the problem. Compare §6.8, where QuantLib threw a loud,
specific exception naming the exact strike and time — far easier to diagnose and trust.

---

## 6.8 Local Volatility (Dupire)

### Theory
A **deterministic** volatility *function* of strike and time, chosen so the model reproduces the
entire observed surface exactly:
```
σ_loc²(K,T) = [ ∂C/∂T + (r−q)K ∂C/∂K + qC ] / [ ½ K² ∂²C/∂K² ]
```
Not stochastic vol — vol is a known function of where spot is and when. Priced by
finite-difference PDE.

### Implementation
```
SABR fits (§6.6) on a 21-point strike grid per expiry
    → ql.BlackVarianceSurface   (bilinear interpolation)
    → ql.LocalVolSurface        (Dupire)
    → ql.FdBlackScholesVanillaEngine   (100×100 PDE)
```

**Why built from SABR rather than raw quotes:** Andreasen-Huge's own `localVol()` was numerically
unstable (to 4456%), and — decisively — feeding it **denser** raw data made it **worse**, which
ruled out data sparsity and pointed at calendar-arbitrage noise in real market quotes, which
Dupire's `∂/∂T` term amplifies. SABR slices are smooth by construction.

**Two bugs — and the second was only found because the first fix didn't work:**
1. SABR's independent per-expiry fits produce **genuine calendar arbitrage** (total variance
   decreasing in T) on ~11% of dates. Fixed by enforcing monotonic `w = IV²·T` across expiries.
   **This alone changed nothing** — identical failures on identical dates.
2. Real cause: `setInterpolation("bicubic")`. **Cubic splines overshoot *between* grid points**
   even when the grid itself is monotonic. Diagnosed from the sub-day time deltas in the error
   messages (~1 day apart — inside a single grid segment, so the check was firing on interpolated
   points, not the raw grid). Switching to the default bilinear (which cannot overshoot a
   monotonic grid) cut failures from 21 to 13 of 192.

### Results
| | |
|---|---|
| Dates succeeding | **179/192 (93.2%)** |
| In-sample IV error | 0.620pp |
| Local vol range | 31%–115% — sane, **no blow-ups** (vs AH's 4456%) |
| **OOS RMSE** | **0.603 — rank 1, best of all nine** |
| Fair-subset RMSE | **0.587 — still rank 1** |
| Decay h=5 → 21 | 0.492 → 0.835 (+70%) |
| Era | 0.286 vs 0.748 — best in both eras |
| Coverage | 14.60 of ~15 contracts/row (7,578 of 8,315, 91.1%) |

**Remaining 13 failures**, two distinct genuine causes: 11 residual calendar-sensitivity at the
shortest expiry; 2 butterfly (strike-convexity) violations at extreme strikes and long tenors.
A complete fix requires 2D arbitrage repair across both dimensions simultaneously — a
research-level problem, deliberately not attempted for one of nine models.

### What it teaches
**The most accurate model tested, and the least robust to build** — those two facts together are
arguably the most useful output of the study. It is best in 15 of 16 years (Heston edged it in 2017) and in both eras, and its win
survives the fair-subset check. But getting there required diagnosing two separate
arbitrage/interpolation bugs, and it still fails outright on about 6% of dates (12 of 192 in the
2026-11-18 rerun; 13 at Phase 7 build time).

For a production system that trade-off may well favour SABR (rank 2, 960/960 fits in 18 seconds,
100% coverage).
For pure accuracy it does not. **Accuracy and robustness point in opposite directions here.**

---

## 6.9 Merton Jump-Diffusion

### Theory
Adds a compound-Poisson jump component to GBM — discontinuous moves, not just diffusion:
```
dS/S = (r − q − λk) dt + σ dW + dJ
J = compound Poisson,  arrivals ~ Poisson(λ),  ln(1+jump) ~ N(μ_j, σ_j²)
k = E[jump] = e^{μ_j + σ_j²/2} − 1        (compensator, keeps the risk-neutral drift correct)
```
Closed form as a **Poisson-weighted sum of Black-Scholes prices**, conditioning on n jumps:
```
V = Σ_n [ e^{−λ'T}(λ'T)ⁿ/n! ] · BS(S, K, T, r_n, q, σ_n)
σ_n = √(σ² + n σ_j²/T),   r_n = r − λk + n·ln(1+k)/T,   λ' = λ(1+k)
```

### Implementation
Hand-written (`src/merton.py`) — no jump-diffusion engine is exposed in this QuantLib build
(verified by inspecting `dir(ql)`). Low risk: every term is a plain BS price and the Poisson
weights decay factorially. Truncated at **25 terms** — justified, not guessed: worst case within
the parameter bounds is `λT = 15` needing ~30 terms for 5σ coverage, while actual calibrated λ
keeps `λT` well under 1.

**Performance lesson.** First version took ~130 s/date (≈ 7 hours for the batch). The *obvious*
optimization — vectorizing the n-terms sum — **barely helped** (130 → 109 s). The real bottleneck
was the Python loop over ~45 contracts inside the objective; `merton_price_batch()` vectorizes
across all contracts at once. Verified **bit-identical** to the scalar version (max difference 0.0).

### Results
| | |
|---|---|
| Dates calibrated | **192/192, 188 converged (97.9%)** |
| In-sample IV error | 1.471pp |
| **OOS RMSE** | **0.993** (rank 4) |
| Decay h=5 → 21 | 0.814 → 1.248 — **+53%, slowest decay of any ranked smile model** |
| Era | 0.436 vs 1.175 |

**Parameters (mean):** σ = 0.102 · λ = 0.461/yr · μ_j = −0.284 · σ_j = 0.163

**Two unprompted economic validations:**
1. **λ tracks real crises.** Mean jump intensity by year — 2020: **1.25**, 2022: **0.86**, versus
   0.22–0.56 in calm years. COVID and the rate-hike year stand out 3–5×. Nothing told the model
   when crises occurred; it read them out of option prices.
2. **μ_j is negative on all 192 dates** (mean −28%). The model consistently prices jumps as
   *downward* — crash risk, not symmetric surprise — the correct empirical sign for equity indices,
   and never constrained to it.

Note σ (diffusive vol, 10.2%) sits well below total implied vol: the jump component absorbs
variance a pure-diffusion model must load into σ. That decomposition is the model working as
intended.

### What it teaches
**The most economically interpretable model of the nine.** Its parameters map onto real, nameable
quantities — how often crashes happen, how big, in which direction — and those parameters
reacted to COVID and the 2022 hiking cycle: the jump intensity rose to its 5.0 cap on 2020-03-16
and 2020-04-15 (with jump-size volatility near zero, so a bound, not a clean estimate), and its
2022 median (0.56) sat above the neighbouring years (0.35 in 2021, 0.21 in 2023).

Its **slowest decay among ranked smile models (+53%)** is consistent with that interpretation: jump
parameters encode slow-moving tail-risk beliefs rather than fast-moving spot-vol dynamics, so they
stale more gracefully. For a use case needing parameters that stay valid for weeks, that matters
more than its rank-4 headline number.

*Caveat:* λ hit its upper bound (5.0) on at least one date — a floor there, not a fit.

---

# 7. Cross-model analysis

**The headline finding.** Every model capturing the volatility smile beats the constant-volatility
family clearly: pooled RMSE 0.86-1.29 against 1.89, a cut of about a third (more than half for
local volatility), and 59-74% on median error. It holds in every bootstrap resample and in every
year for local volatility, Heston and Merton; SABR was marginally worse than BSM in 2020 (2.90 vs 2.87). **Modelling the smile is most of the accuracy.**

**Four constant-vol pricers agreeing to three decimals is validation, not redundancy.** Black-76
*is* BSM on the forward; GBM-MC *is* BSM simulated; CRR adds an early-exercise premium worth
almost nothing on a low-dividend index. Four independent implementations landing on 1.742–1.752
says the plumbing is sound.

**Sophisticated models decay faster in percentage terms** (25-43% vs 11%). They start from a far
lower base. **The ranking never inverts in absolute terms:** Local Vol at 21 days (0.992) still
beats BSM at 5 days (1.780).

**The era effect is largely a data-convention effect.** Every model finds 2014-25 about 2.5-3x
harder, but so does the reference that knows each contract's own same-day vendor IV (0.560 vs
0.142, 3.9x). The later vendor's IVs reprice the mid less well under this project's rate and
dividend assumptions. A genuinely harder market may contribute; this design cannot separate the
two (§3.2 item 5).

**Parsimony is underrated.** Heston (5 params) and SABR (15 params) are statistically tied
out-of-sample (1.256 vs 1.290, bootstrap difference straddles zero), against a ~0.6 gap to the
constant-vol family. Tripling the parameter count bought nothing measurable once scoring is
out-of-sample, which is the argument for out-of-sample evaluation.

**Accuracy and robustness are in tension.** Ranked by accuracy: Local Vol first, then SABR,
Heston and Merton tied. Ranked by robustness: SABR (960/960, 18 s) > Merton ≈ Heston (188/192, hours) > Local Vol
(180/192 in the rerun, two bug hunts). The two orderings are nearly reversed.

---

# 8. Honest limitations

- **% inside bid-ask carries almost no signal.** Measured SPY spreads on the grid are $0.01–0.09
  (0.6–3% of mid). No model prices inside a 2-cent window using 5–21-day-old parameters; all nine
  score 1.8–6.6%. Reported because it was in the original spec; it ranks nothing meaningfully.
- **The pipeline needed a post-hoc audit to be correct** (§3.1). Two of the five issues — a crash
  that silently dropped 4 dates and a crash hidden behind a shell redirect — produced output that
  looked valid. Results here are the corrected run, but the honest reading is that a single pass
  through a pipeline this size should not be trusted without one.
- **Flat 1.4% dividend proxy**, not a real dividend schedule. Identical across models so the
  *comparison* is unaffected; absolute price levels carry this error.
- **Vasicek `a` and `b` are not individually identified** — only the discount curve is validated.
- **Heston's κ clusters bimodally** near {2.0, 5.0}; do not read it as a mean-reversion estimate.
- **Local Vol failed on 12/192 dates in the rerun (13 at build time)**; excluded from its sample
  only, and its partial gaps on 49 more dates are why the like-for-like set is contract-level (§3.2).
- **Merton's λ hit its 5.0 bound on at least one date** — a floor, not a fit.
- **SABR's β fixed at 1.0**, sensitivity not tested.
- **Single underlying, single asset class.** SPY only; nothing generalises automatically to single
  names, other indices, or other asset classes.
- **Monthly calibration only.** Daily recalibration was not tested and would likely change the
  decay results materially.
- **This is a pricing-accuracy study, not a trading study.** Lower OOS pricing error does not
  establish profitability; no transaction costs, hedging, or execution are modelled.

---

# 9. Conclusion, build log, reproduction

## Conclusion
**Smile-aware models (Local Volatility, SABR, Heston, Merton) cut out-of-sample SPY option pricing
error clearly versus constant-volatility Black-Scholes: by about a third on pooled RMSE, by more
than half for local volatility, and by 59-74% on median error. Local Volatility is the most
accurate of the nine but the least numerically robust to build, failing outright on about 6% of dates —
accuracy and robustness point in opposite directions.**

Three secondary conclusions worth carrying forward:
1. **Out-of-sample evaluation changes the answer.** In-sample, SABR's 15 parameters dominate
   Heston's 5. Out-of-sample they are tied. Parameter count buys in-sample fit, not forecast
   accuracy.
2. **Silent failure is the dangerous failure mode.** Andreasen-Huge nearly earned a top ranking on
   a systematically easier half of the data because it omitted rather than errored.
3. **Model parameters carry real economic information.** Merton's jump intensity independently
   rose to its cap in March-April 2020 and was elevated in 2022; Heston's v₀ matched intraday VIX during the crash.

## Build log
| Phase | Deliverable | Result |
|---|---|---|
| 0 | Vendor IV/Greeks validation | Genuine BSM-family output confirmed; vega artifact rule found |
| 1 | Data pipeline | 23.5M rows, 16 partitions; Hurricane Sandy artifact found |
| 2 | Vasicek discount curve | 1–3yr validated 0.29–0.66pp; a/b non-identifiability documented |
| 3 | Baseline pricers | 4 pricers, 7 cross-validation checks, machine-precision agreement |
| 4 | Heston | 192/192, 188 converged; warm-start 42× speedup; 2 bugs fixed |
| 5 | SABR | 960 fits in 18 s, 100% converged after the §3.1 forward fix |
| 6 | Andreasen-Huge surface | 0.288pp in-sample; 2 API bugs found via QuantLib source |
| 7 | Local Volatility | 179/192 (93.2%); cubic-spline overshoot bug root-caused |
| 8 | Merton Jump-Diffusion | 192/192, 188 converged; λ tracks real crises |
| 9 | OOS scoring harness | 4,959 model-cells, 9 models, 3 horizons |
| 10 | This report | — |
| Audit | Cross-phase consistency check | 5 issues found and fixed, all results regenerated (§3.1) |

## Reproduction
**Code:** `src/` — `pricers.py` (BSM/Black-76/GBM-MC/CRR/BAW), `sabr.py`, `merton.py`, and
`phaseN_*.py` per phase. **Data:** `data/*.parquet`. **Per-phase methodology:**
`docs/phase*_notes.md`. **Raw scores:** `results/phase9_oos_scores.csv`. **Tables:**
`results/phase10_summary_tables.txt`.

End-to-end: `phase1_data_pipeline.py` → `phase2_vasicek_calibration.py` →
`phase4_heston_batch.py` → `phase5_sabr_batch.py` → `phase7_local_vol_batch.py` →
`phase8_merton_batch.py` → `phase9_prep_constvol.py` → `phase9_oos_runner.py` →
`phase10_analysis.py`. All batch scripts are resumable and self-skip completed dates.
`src/audit_consistency.py` re-runs the §3.1 forward and rate-convention checks; run it after any
change to either convention.

**Environment note:** background processes were killed ~22 times during this project (memory
pressure on a 16GB machine). Resumable checkpointing plus foreground-chunked fallback is the
pattern that worked; rolling backups live in `backups/`.
