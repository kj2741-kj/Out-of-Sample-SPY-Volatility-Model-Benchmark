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

Every model that captures the volatility smile roughly **halves** the out-of-sample pricing error
of the constant-volatility family.

| Model | OOS RMSE | vega-RMSE | Coverage |
|---|---|---|---|
| Local Volatility (Dupire) | **0.603** | 8.73 | 91.1% |
| SABR | 0.930 | 21.46 | 100% |
| Heston | 0.960 | 21.60 | 99.9% |
| Merton Jump-Diffusion | 0.993 | 22.32 | 100% |
| GBM Monte Carlo | 1.742 | 25.43 | 100% |
| CRR American | 1.750 | 25.67 | 100% |
| Black-Scholes-Merton | 1.752 | 25.52 | 100% |
| Black-76 | 1.752 | 25.52 | 100% |
| *Andreasen-Huge* | *0.878* | *2.08* | *50.6% — excluded, see report §6.7* |

> Post-audit figures, regenerated from `results/phase9_oos_scores.csv`. A consistency audit found
> five issues (a SABR forward-definition mismatch, a split rate convention, and a crash that
> silently dropped 4 dates, among others); all were fixed and every affected result recomputed.
> See `docs/FINAL_REPORT.md` §3.1.

**Local Volatility is the most accurate model and the least robust to build** — it fails surface
construction outright on 7% of dates. Accuracy and robustness point in opposite directions.

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

Requires `QuantLib`, `numpy`, `pandas`, `scipy`, `pyarrow`, `yfinance`.

## Scope and limitations

This is a **pricing-accuracy study, not a trading study** — no transaction costs, hedging or
execution are modelled, and lower pricing error does not imply profitability. Single underlying
(SPY). Flat 1.4% dividend proxy. Monthly calibration cadence only. Full list in
`docs/FINAL_REPORT.md` §8.
