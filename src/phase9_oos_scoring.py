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


def arbitrage_ok(day):
    """AUDIT FIX (2026-11-18): drop quotes that violate American no-arbitrage bounds, i.e. a mid
    below intrinsic value by more than half the spread, or above the underlying (call) / strike
    (put). Three such quotes reached the 2019 scoring grid (mids $0.03-0.42 on contracts worth
    $13-28), and because RMSE squares errors they dominated that year for every model. Contracts
    just above intrinsic (valid American prices below the European bound) are kept."""
    S, K, mid = day["spy_close"], day["strike"], day["mid"]
    half_spread = (day["ask"] - day["bid"]).clip(lower=0) / 2
    intrinsic = np.where(day["is_call"], np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    upper = np.where(day["is_call"], S, K)
    return (mid >= intrinsic - half_spread) & (mid <= upper)


def monotone_ok(day, min_gap=0.01):
    """AUDIT FIX (2026-11-18): enforce the model-free vertical-spread bound within each
    (expiration, option type): put prices must not fall as strike rises, call prices must not
    rise. On 2019-02-14 the vendor carried call-like prices in PUT rows (a 247 put at $28.05
    next to a 249 put at $0.40, SPY 274); those quotes pass every single-contract bound and the
    day had no matching calls for a parity check. A pair violates if the gap exceeds both
    half-spreads plus 1 cent. The quote in the most violations is removed, repeatedly, until
    the group is consistent, which keeps the largest arbitrage-free set."""
    keep = pd.Series(True, index=day.index)
    half = ((day["ask"] - day["bid"]).clip(lower=0) / 2).values
    pos = {ix: i for i, ix in enumerate(day.index)}
    for (_, is_call), g in day.groupby(["expiration", "is_call"]):
        if len(g) < 2:
            continue
        g = g.sort_values("strike")
        K, m = g["strike"].values, g["mid"].values
        h = half[[pos[ix] for ix in g.index]]
        lower_k = K[:, None] < K[None, :]
        tol = h[:, None] + h[None, :] + min_gap
        if is_call:   # lower-strike call must cost at least as much
            v = lower_k & (m[:, None] + tol < m[None, :])
        else:         # lower-strike put must cost no more
            v = lower_k & (m[:, None] > m[None, :] + tol)
        v = v | v.T
        alive = np.ones(len(g), dtype=bool)
        while True:
            counts = (v & alive[None, :] & alive[:, None]).sum(axis=1)
            if counts.max() == 0:
                break
            alive[np.argmax(counts)] = False
        keep.loc[g.index[~alive]] = False
    return keep


def build_scoring_grid(df, score_date, r=None):
    """Nearest actual listed contract to each (moneyness, tenor) grid point on score_date.
    `r` is accepted for interface stability; the quote filters below are model-free."""
    day = df[(df["date"] == score_date) & df["has_valid_quote"] & df["iv_quality_ok"]].copy()
    if len(day) == 0:
        return pd.DataFrame()
    day = day[arbitrage_ok(day)]
    day = day[monotone_ok(day)]
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
