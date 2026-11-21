# Out-of-Sample SPY Volatility Model Benchmark

Nine option-pricing models (plus true Dupire local vol, added in Phase 12) calibrated on real SPY
option chains (2010–2025) and compared
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

> **Correction, 2026-11-20 (Phase 12).** The model previously labelled "Local Volatility (Dupire)" was
> never local volatility: it was priced with QuantLib's `FdBlackScholesVanillaEngine` without
> `localVol=True`, so it priced Black-Scholes at the frozen SABR-smoothed implied-vol surface, read by
> strike. It is relabelled below as a **sticky-strike surface**; its numbers are unchanged and valid
> for that model. True Dupire local vol is now built properly (`src/phase12_dupire.py`, model-identity
> tests in `tests/test_phase12_dupire.py`) and scored on the same contracts. See
> [`docs/phase12_dupire_localvol_notes.md`](docs/phase12_dupire_localvol_notes.md).

Smile-aware models cut out-of-sample pricing error clearly against constant-volatility
Black-Scholes. The best model is the simplest smile model of all: **today's implied-vol surface,
frozen and read by strike**, 54% below Black-Scholes. SABR, Heston, Merton and true Dupire local
vol sit about a third below Black-Scholes and are statistically close to each other.

| Model | Pooled RMSE ($) | Median abs error ($) | vs Black-Scholes (pooled) |
|---|---|---|---|
| Sticky-strike surface (SABR-smoothed IV surface frozen by strike) | **0.864** | **0.254** | -54% |
| Merton Jump-Diffusion | 1.233 | 0.427 | -35% |
| Heston | 1.256 | 0.391 | -33% |
| SABR | 1.290 | 0.342 | -32% |
| Local Volatility (true Dupire, Phase 12) | 1.359 | 0.346 | -28% |
| GBM Monte Carlo | 1.879 | 0.948 | -1% |
| CRR American | 1.883 | 0.968 | 0% |
| Black-Scholes-Merton | 1.889 | 0.961 | |
| Black-76 | 1.889 | 0.961 | |
| *Reference: Black-Scholes at each contract's own same-day vendor IV* | *0.485* | *0.146* | *floor, not a model* |

7,828 contracts priced by every ranked model (91.2% of 8,583 scorings, 573 date-horizon cells,
192 calibration dates). True Dupire prices all 8,583. Andreasen-Huge (also a sticky-strike surface,
same engine issue) prices 50.5% of the grid and is excluded (report section 6.7). Bootstrap 95%
ranges over calibration dates: sticky-strike surface 0.66-1.12, Merton 1.05-1.41, Heston 1.03-1.52,
SABR 1.02-1.61, Dupire 1.17-1.57, Black-Scholes 1.74-2.05.

**Why sticky strike wins.** SABR (beta 1) and Merton move the smile with the forward; frozen local
vol moves it against spot; the sticky-strike surface keeps it fixed by strike. SPY's smile stayed
closest to fixed by strike, which Phase 11's hedging test confirms from the other side (the best
hedge delta sits about a third of the way from sticky-strike Black-Scholes toward local vol).

> Second audit, 2026-11-18 (report section 3.2). The earlier headline reported an n-weighted mean of
> per-cell RMSEs and a "fair subset" defined by cell presence, which let local volatility skip 182
> contracts; three corrupt quotes (one vendor day carrying call prices in put rows) and a
> year-boundary bug that dropped every December 21-day horizon were also fixed. Figures above are
> pooled RMSE on contract-level like-for-like data from `results/phase9_oos_contracts.csv`.

**The sticky-strike surface is the most accurate model and the least robust to build**: its surface
construction fails outright on about 6% of dates (12 of 192 in the 2026-11-18 rerun). The Phase 12
Dupire build, made directly from the SABR slices, prices every date.

**A floor shared by every model.** Black-Scholes at each contract's own same-day vendor IV still
misses the mid by 0.485 pooled (0.560 in 2014-25, 0.142 in 2010-13), because calibration targets
vendor IVs while scoring uses mids under this project's rate and dividend assumptions. Much of the
2014-25 "era effect" is this data-convention gap rather than a harder market.

## Greeks, delta hedging and P&L explain (Phase 11, 2020-2021)

An extension on the 24 monthly calibration dates of 2020 (COVID stress) and 2021 (calm). Each date's
scoring-grid contracts are followed daily for up to 21 trading days with parameters frozen:
7,196 contract-days, 355 contracts, 6,818 one-day hedges. Details: [`docs/phase11_greeks_hedging_notes.md`](docs/phase11_greeks_hedging_notes.md).

- **Greeks.** SABR and Merton deltas (bump-and-reprice) sit above the Black-Scholes delta at the market's
  own implied vol, by +0.08 to +0.13 at the money.
- **Hedging.** A daily delta hedge with Black-Scholes at market implied vol has the smallest error. Frozen
  SABR and Merton deltas roughly double the squared hedging error (SABR -100%, 95% cluster-bootstrap
  interval -195% to -53%). The variance-minimising delta sits below Black-Scholes in every slice, so the
  smile models move delta the wrong way.
- **P&L explain.** Unexplained daily option P&L: delta 31%, +gamma 29%, +vega 12%, +theta 4.9%.

- **Local vol.** True Dupire deltas sit 0.12-0.14 below Black-Scholes and hedge better than SABR and
  Merton but worse than Black-Scholes (gain -33%); in the March-April 2020 stress they tie. The
  least-squares blend puts the best hedge delta 32% of the way from Black-Scholes to local vol in both
  years, 50% in the stress window (in sample).

## Models

| | Model | Implementation |
|---|---|---|
| 1 | Black-Scholes-Merton | closed form (`src/pricers.py`) |
| 2 | Black-76 | closed form, forward-based — written independently of #1 as a cross-check |
| 3 | GBM Monte Carlo | terminal-value sampling + antithetic variates |
| 4 | CRR Binomial | American exercise; Barone-Adesi-Whaley also implemented as a cross-check |
| 5 | Heston | QuantLib `AnalyticHestonEngine`, warm-started calibration |
| 6 | SABR | QuantLib `sabrVolatility` (Hagan), per-expiry, β fixed at 1.0 |
| 7 | Andreasen-Huge | QuantLib arbitrage-free surface (same engine issue: priced as a sticky-strike surface) |
| 8 | Sticky-strike surface (was labelled Local Volatility) | SABR-smoothed `BlackVarianceSurface`, FD PDE without `localVol=True` = BS at the surface vol by strike |
| 8b | Local Volatility (true Dupire, Phase 12) | Gatheral's Dupire formula on the SABR slices → `FixedLocalVolSurface` → FD PDE with `localVol=True` |
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
python src/phase11_greeks_hedging.py      # Greeks panel 2020-21
python src/phase11_analysis.py            # hedging + P&L explain tables
python src/phase11_audit.py               # audit checks
python src/phase12_localvol_oos.py        # true Dupire OOS on the Phase 9 contracts
python src/phase12_analysis.py            # Phase 12 tables
```

All batch scripts are resumable and skip already-completed dates.

Requires `QuantLib`, `numpy`, `pandas`, `scipy`, `pyarrow`, `yfinance`. Data files were rewritten
on 2026-11-18 so pyarrow 19 can read them (they had been written by pyarrow 24).

Checks: `python tests/test_scoring.py` (quote filters, pooled RMSE, like-for-like, results file) and
`python tests/test_phase12_dupire.py` (model-identity tests for local vol).

## Scope and limitations

This is a **pricing-accuracy study, not a trading study** — no transaction costs or execution are
modelled (Phase 11 adds a daily delta-hedging test on 2020-2021), and lower pricing error does not imply profitability. Single underlying
(SPY). Flat 1.4% dividend proxy. Monthly calibration cadence only. Calibration targets vendor IVs, which creates a
shared error floor (above). Full list in `docs/FINAL_REPORT.md` §8.
