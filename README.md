# Out-of-Sample SPY Volatility Model Benchmark

Nine option-pricing models calibrated on real SPY option chains (2010–2025) and compared
**out-of-sample**: calibrate once, freeze the parameters, then price the market 5, 10 and 21
trading days later against actual quotes.

**23.5M contract-quotes · 192 monthly calibrations · 555 out-of-sample date-horizon cells per model
(~8,300 contract prices each)**

---

## Why out-of-sample

In-sample fit mostly measures parameter count — SABR fits 15 parameters per date and beats
Heston's 5 in-sample, which says little about model quality. Freezing the parameters and scoring
weeks forward asks the question that matters: *whose calibration is still pricing the market
correctly later?*

## Headline result

Smile-aware models cut out-of-sample pricing error clearly against constant-volatility
Black-Scholes: by about a third on pooled RMSE for SABR, Heston and Merton, by more than half for
local volatility, and by 59-74% on median error. Local volatility is best by a margin that holds in
every bootstrap resample; SABR, Heston and Merton are statistically tied with each other (their
order changes with the metric).

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

> Second audit, 2026-11-18 (report section 3.2). The earlier headline reported an n-weighted mean of
> per-cell RMSEs and a "fair subset" defined by cell presence, which let local volatility skip 182
> contracts; three corrupt quotes (one vendor day carrying call prices in put rows) and a
> year-boundary bug that dropped every December 21-day horizon were also fixed. Figures above are
> pooled RMSE on contract-level like-for-like data from `results/phase9_oos_contracts.csv`.

**Local Volatility is the most accurate model and the least robust to build**: it fails surface
construction outright on about 6% of dates (12 of 192 in the 2026-11-18 rerun). Accuracy and robustness point in opposite directions.

**A floor shared by every model.** Black-Scholes at each contract's own same-day vendor IV still
misses the mid by 0.485 pooled (0.560 in 2014-25, 0.142 in 2010-13), because calibration targets
vendor IVs while scoring uses mids under this project's rate and dividend assumptions. Much of the
2014-25 "era effect" is this data-convention gap rather than a harder market.

## Models

| | Model | Implementation |
|---|---|---|
| 1 | Black-Scholes-Merton | closed form (`src/pricers.py`) |
| 2 | Black-76 | closed form, forward-based — written independently of #1 as a cross-check |
| 3 | GBM Monte Carlo | terminal-value sampling + antithetic variates |
| 4 | CRR Binomial | American exercise; Barone-Adesi-Whaley also implemented as a cross-check |
| 5 | Heston | QuantLib `AnalyticHestonEngine`, warm-started calibration |
| 6 | SABR | QuantLib `sabrVolatility` (Hagan), per-expiry, β fixed at 1.0 |
| 7 | Andreasen-Huge | QuantLib arbitrage-free surface |
| 8 | Local Volatility | SABR-smoothed `BlackVarianceSurface` → `LocalVolSurface` → FD PDE |
| 9 | Merton Jump-Diffusion | hand-written Poisson-weighted BS sum, vectorised |

Discounting throughout: a Vasicek short-rate model fit to FRED Treasury yields.

## Repository layout

```
src/      pricers.py, sabr.py, merton.py, and phaseN_*.py — one module per phase
docs/     FINAL_REPORT.md (full methodology + results) and per-phase notes
results/  raw out-of-sample scores and summary tables
data/     calibrated parameters per date (bulk chain data is gitignored — see below)
```

**Start here:** [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) — §6 has one self-contained section
per model (theory → implementation → calibration → results → what it teaches).

## Data

The raw dataset (23.5M SPY contract-quotes, 2010–2025, ~1.4GB) is **not** in this repo. It was
assembled from two vendors — OptionsDX (2010–13) and Alpha Vantage (2014–25) — and validated
independently before use (see `docs/phase0_methodology_notes.md`: vendor IV/Greeks were
recomputed from scratch and diffed against the supplied values).

`data/clean/` (932MB of cleaned yearly partitions) is gitignored and regenerable via
`src/phase1_data_pipeline.py`. The small calibrated-parameter files **are** committed.

## Reproducing

```
python src/phase1_data_pipeline.py        # clean + partition raw chains
python src/phase2_vasicek_calibration.py  # discount curve
python src/phase4_heston_batch.py         # Heston,  192 dates
python src/phase5_sabr_batch.py           # SABR,    960 fits
python src/phase7_local_vol_batch.py      # local vol surfaces
python src/phase8_merton_batch.py         # Merton,  192 dates
python src/phase9_prep_constvol.py        # constant-vol baseline
python src/phase9_oos_runner.py           # out-of-sample scoring
python src/phase10_analysis.py            # summary tables
```

All batch scripts are resumable and skip already-completed dates.

Requires `QuantLib`, `numpy`, `pandas`, `scipy`, `pyarrow`, `yfinance`. Data files were rewritten
on 2026-11-18 so pyarrow 19 can read them (they had been written by pyarrow 24).

Checks: `python tests/test_scoring.py` (quote filters, pooled RMSE, like-for-like, results file).

## Scope and limitations

This is a **pricing-accuracy study, not a trading study** — no transaction costs, hedging or
execution are modelled, and lower pricing error does not imply profitability. Single underlying
(SPY). Flat 1.4% dividend proxy. Monthly calibration cadence only. Calibration targets vendor IVs, which creates a
shared error floor (above). Full list in `docs/FINAL_REPORT.md` §8.
