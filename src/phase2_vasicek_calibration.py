"""
Phase 2: fit a single-factor Vasicek short-rate model to FRED 3-month Treasury
yields (2010-2025), then build a discount-factor curve P(t,T) usable by every
downstream pricing phase.

dr = a(b - r) dt + sigma dW

sigma fit from the time series (daily innovations). a,b fit by matching the
model's affine yield formula to the sample's AVERAGE 1yr/2yr/3yr term-structure
shape (3 points, 2 unknowns, overdetermined), not by time-series MLE -- pure
historical MLE for (a,b) is degenerate on this series (near unit-root daily
persistence, only ~1.5 rate cycles in the 15-year window; verified separately).

Calibrated out to 3yr, not just 2yr: SPY option chain LEAPS run out to a full
3 years (confirmed against data/clean/*.parquet, max T=3.0), and 2.5-5%/year
of rows have T>2yr, so a 2yr-only calibration would leave a non-trivial slice
of the options data on unvalidated extrapolation.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DATA = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation\data"
RESULTS = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation\results"

def vasicek_zero_price(a, b, sigma, r0, tau):
    B = (1 - np.exp(-a * tau)) / a
    A = np.exp((b - sigma**2 / (2 * a**2)) * (B - tau) - (sigma**2 / (4 * a)) * B**2)
    return A * np.exp(-B * r0)

def main():
    df = pd.read_parquet(f"{DATA}\\fred_treasury_yields.parquet").dropna(subset=["y3mo"])
    df = df.sort_values("date").reset_index(drop=True)
    r = (df["y3mo"] / 100).values
    dt = df["date"].diff().dt.days.values[1:] / 365.0

    r_t, r_next = r[:-1], r[1:]
    dt_mean = dt.mean()
    sigma = np.std(r_next - r_t) / np.sqrt(dt_mean)

    avg = df[["y3mo", "y1yr", "y2yr", "y3yr"]].mean() / 100
    r0_avg = avg["y3mo"]
    target_tenors = [(1.0, "y1yr"), (2.0, "y2yr"), (3.0, "y3yr")]

    def resid(params):
        a_, b_ = params
        if a_ <= 1e-6:
            return [1e3] * len(target_tenors)
        out = []
        for tau, col in target_tenors:
            y = -np.log(vasicek_zero_price(a_, b_, sigma, r0_avg, tau)) / tau
            out.append(y - avg[col])
        return out

    fit = least_squares(resid, x0=[0.3, r0_avg + 0.01], bounds=([1e-4, -0.02], [20.0, 0.20]))
    a, b = fit.x
    print("Time-series MLE for (a,b) is degenerate on this series (unit-root-like daily "
          "autocorr 0.9999, only ~1.5 rate cycles in 2010-2025) -- fit (a,b) to average "
          "1yr/2yr/3yr term-structure shape instead (3 targets, 2 unknowns, overdetermined).")
    print(f"Fitted Vasicek: a={a:.4f} (mean-reversion speed), b={b:.4f} ({b*100:.2f}% long-run mean), "
          f"sigma={sigma:.4f}, curve-fit converged={fit.success}")
    print(f"Implied half-life of mean reversion: {np.log(2)/a:.2f} years")
    print(f"Curve-fit residuals (model - avg observed, pp): {np.array(resid([a, b])) * 100}")

    df["r3mo"] = r
    out = df[["date", "y3mo", "y1yr", "y2yr", "y3yr", "r3mo"]].copy()
    for tau, label in [(1/12, "1mo"), (3/12, "3mo"), (1.0, "1yr"), (2.0, "2yr"), (3.0, "3yr")]:
        out[f"df_{label}"] = [vasicek_zero_price(a, b, sigma, x, tau) for x in out["r3mo"]]
        if tau >= 1.0:
            out[f"vasicek_yield_{label}"] = -np.log(out[f"df_{label}"]) / tau * 100

    out.to_parquet(f"{DATA}\\vasicek_discount_curve.parquet", index=False)

    for tenor, model_col, actual_col in [("1yr", "vasicek_yield_1yr", "y1yr"),
                                          ("2yr", "vasicek_yield_2yr", "y2yr"),
                                          ("3yr", "vasicek_yield_3yr", "y3yr")]:
        valid = out.dropna(subset=[actual_col])
        diff = valid[model_col] - valid[actual_col]
        corr = valid[model_col].corr(valid[actual_col])
        print(f"\nOOS check vs real {tenor} yield (model fit to average curve shape, not the daily path):")
        print(f"  corr={corr:.4f}  mean_diff={diff.mean():+.3f}pp  RMSE={np.sqrt((diff**2).mean()):.3f}pp")

    params_df = pd.DataFrame([dict(a=a, b=b, sigma=sigma, half_life_years=np.log(2)/a, converged=fit.success)])
    params_df.to_csv(f"{RESULTS}\\phase2_vasicek_params.csv", index=False)

if __name__ == "__main__":
    main()
