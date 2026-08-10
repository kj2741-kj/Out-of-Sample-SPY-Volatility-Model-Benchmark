"""
Phase 0 methodology check: neither OptionsDX nor AlphaVantage publish an exact
spec for how they compute implied_volatility/delta/gamma/theta/vega/rho (both
vendors only advertise "industry-standard option pricing model, usually
Black-Scholes" in third-party docs, no disclosed dividend/rate convention or
American-vs-European treatment). So instead of trusting the vendor numbers
blind, independently recompute European Black-Scholes IV + Greeks from
(S, K, T, r, q, mid-price) and diff against the vendor's reported values.

SPY options are American-style (early exercise allowed), so a plain European
BSM replication is expected to show a small systematic bias vs whatever the
vendor actually used (likely a binomial or Barone-Adesi-Whaley American
approximation) -- that bias itself is diagnostic, not a bug in our replication.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

DATA_DIR = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Data"
PROJ_DIR = r"C:\Users\Kartavya\Desktop\NYU\Summer\Options Model Validation"
Q_DIV_YIELD = 0.014  # SPY trailing dividend yield proxy, flat for this diagnostic pass

def bs_price(S, K, T, r, q, sigma, is_call):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def bs_implied_vol(price, S, K, T, r, q, is_call):
    if T <= 0 or price <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if price < intrinsic - 1e-6:
        return np.nan
    f = lambda sig: bs_price(S, K, T, r, q, sig, is_call) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-6, maxiter=200)
    except ValueError:
        return np.nan

def bs_greeks(S, K, T, r, q, sigma, is_call):
    if T <= 0 or sigma <= 0:
        return dict(delta=np.nan, gamma=np.nan, theta=np.nan, vega=np.nan, rho=np.nan)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if is_call:
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

def main():
    opts = pd.read_parquet(f"{DATA_DIR}\\spy_options_2010_2025_combined.parquet")
    spy = pd.read_parquet(f"{PROJ_DIR}\\data\\spy_underlying_close.parquet")
    irx = pd.read_parquet(f"{PROJ_DIR}\\data\\riskfree_irx.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    irx["date"] = pd.to_datetime(irx["date"])

    # stratified sample: ~1500 rows per vendor, spread across years, liquid mid-price rows only
    opts = opts.dropna(subset=["bid", "ask", "implied_volatility", "delta"])
    opts = opts[(opts["bid"] > 0) & (opts["ask"] > 0)]
    opts["mid"] = (opts["bid"] + opts["ask"]) / 2

    samples = []
    for src in ["OptionsDX", "AlphaVantage"]:
        sub = opts[opts["source"] == src]
        n_per_year = max(1, 1500 // sub["year"].nunique())
        samples.append(
            sub.groupby("year", group_keys=False).apply(
                lambda g: g.sample(min(len(g), n_per_year), random_state=42)
            )
        )
    sample = pd.concat(samples, ignore_index=True)

    sample = sample.merge(spy, on="date", how="left")
    sample = sample.merge(irx, on="date", how="left")
    sample = sample.dropna(subset=["spy_close", "irx_yield_pct"])

    sample["T"] = (sample["expiration"] - sample["date"]).dt.days / 365.0
    sample = sample[sample["T"] > 1 / 365]  # drop expiration-day rows, BSM undefined at T=0
    sample["r"] = sample["irx_yield_pct"] / 100.0
    sample["is_call"] = sample["type"] == "call"

    results = []
    for row in sample.itertuples(index=False):
        iv = bs_implied_vol(row.mid, row.spy_close, row.strike, row.T, row.r, Q_DIV_YIELD, row.is_call)
        if np.isnan(iv):
            continue
        g = bs_greeks(row.spy_close, row.strike, row.T, row.r, Q_DIV_YIELD, iv, row.is_call)
        results.append(dict(
            contractID=row.contractID, source=row.source, year=row.year, type=row.type,
            moneyness=row.strike / row.spy_close, T=row.T,
            vendor_iv=row.implied_volatility, my_iv=iv,
            vendor_delta=row.delta, my_delta=g["delta"],
            vendor_gamma=row.gamma, my_gamma=g["gamma"],
            vendor_theta=row.theta, my_theta=g["theta"],
            vendor_vega=row.vega, my_vega=g["vega"],
            vendor_rho=row.rho, my_rho=g["rho"],
        ))

    res = pd.DataFrame(results)
    res.to_csv(f"{PROJ_DIR}\\results\\phase0_replication_sample.csv", index=False)

    print(f"Sample size (successful IV solve): {len(res)}")
    for src in ["OptionsDX", "AlphaVantage"]:
        sub = res[res["source"] == src]
        print(f"\n--- {src} (n={len(sub)}) ---")
        for col in ["iv", "delta", "gamma", "theta", "vega", "rho"]:
            diff = sub[f"my_{col}"] - sub[f"vendor_{col}"]
            corr = sub[f"my_{col}"].corr(sub[f"vendor_{col}"])
            print(f"  {col:6s} corr={corr:.4f}  mean_diff={diff.mean():+.4f}  "
                  f"median_abs_diff={diff.abs().median():.4f}  "
                  f"mean_abs_diff={diff.abs().mean():.4f}")

if __name__ == "__main__":
    main()
