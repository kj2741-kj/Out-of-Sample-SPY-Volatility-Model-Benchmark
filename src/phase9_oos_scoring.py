"""
Phase 9: out-of-sample scoring harness across all 9 models.

THE OOS DESIGN (this is the crux of the whole project):
Each model is calibrated ONCE at a monthly date t (Phases 4-8 did this), then evaluated on
FUTURE dates t+h WITHOUT recalibrating. That's the actual out-of-sample test -- it measures how
well each model's fitted parameters keep pricing the market as time passes, not how well it
fits the data it was trained on (which Phases 4-8 already reported, and where more-parameter
models like SABR trivially win).

Horizons h: +5, +10, +21 trading days. This turns the comparison into a decay curve -- "whose
calibration stays valid longest," a more informative question than a single snapshot.

Scoring grid (resolves the open decision flagged since the Phase 4 plan): the full liquid
universe is 10.7M contracts, infeasible to price with 9 models (LocalVol alone is a PDE solve
per contract). Instead: a fixed moneyness x tenor grid per scoring date -- 5 moneyness buckets
(0.90/0.95/1.00/1.05/1.10) x 3 tenor buckets (~1m/~3m/~6m), nearest actual listed contract to
each grid point. ~15 contracts x 192 dates x 3 horizons = ~8,600 contract-scorings per model.
Comparable across models by construction, and a deliberately liquid, economically meaningful
slice rather than a mass of untradeable deep-wing quotes.

Metrics (per the original project spec): RMSE (price), vega-scaled RMSE, % inside bid-ask.
"""
import numpy as np
import pandas as pd

PROJ = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation"
HORIZONS = [5, 10, 21]
GRID_MONEYNESS = [0.90, 0.95, 1.00, 1.05, 1.10]
GRID_TENORS = [1 / 12, 3 / 12, 6 / 12]
MONEYNESS_TOL = 0.02
TENOR_TOL_FRAC = 0.5
Q_DIV = 0.014


def build_scoring_grid(df, score_date):
    """Nearest actual listed contract to each (moneyness, tenor) grid point on score_date."""
    day = df[(df["date"] == score_date) & df["has_valid_quote"] & df["iv_quality_ok"]].copy()
    if len(day) == 0:
        return pd.DataFrame()
    picks = []
    for m in GRID_MONEYNESS:
        for tenor in GRID_TENORS:
            cand = day[(np.abs(day["moneyness"] - m) < MONEYNESS_TOL) &
                       (np.abs(day["T"] - tenor) < tenor * TENOR_TOL_FRAC)]
            if len(cand) == 0:
                continue
            # OTM convention (same as calibration chains): puts below spot, calls above
            want_call = m >= 1.0
            side = cand[cand["is_call"] == want_call]
            if len(side) == 0:
                side = cand
            score = (side["moneyness"] - m).abs() / MONEYNESS_TOL + (side["T"] - tenor).abs() / tenor
            pick = side.loc[score.idxmin()]
            picks.append(pick)
    if not picks:
        return pd.DataFrame()
    out = pd.DataFrame(picks).drop_duplicates(subset=["contractID"])
    out["grid_moneyness"] = out["moneyness"].round(3)
    return out


def score_metrics(model_prices, contracts):
    """RMSE, vega-scaled RMSE, % inside bid-ask -- the three metrics from the project spec."""
    mp = np.asarray(model_prices, dtype=float)
    mid = contracts["mid"].values
    bid = contracts["bid"].values
    ask = contracts["ask"].values
    vega = np.maximum(contracts["vega"].values, 1e-3)

    valid = np.isfinite(mp) & np.isfinite(mid)
    if valid.sum() == 0:
        return dict(rmse=np.nan, vega_rmse=np.nan, pct_in_spread=np.nan, n=0)
    err = mp[valid] - mid[valid]
    return dict(
        rmse=float(np.sqrt(np.mean(err ** 2))),
        vega_rmse=float(np.sqrt(np.mean((err / vega[valid]) ** 2))),
        pct_in_spread=float(np.mean((mp[valid] >= bid[valid]) & (mp[valid] <= ask[valid])) * 100),
        n=int(valid.sum()),
    )
