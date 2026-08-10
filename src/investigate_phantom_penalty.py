import pandas as pd
import numpy as np
from phase4_heston_calibration import build_curves, select_chain, heston_model_prices, Q_DIV, PROJ
from pricers import bs_implied_vol

df_year = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_2017.parquet")
vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]
prog = pd.read_csv(f"{PROJ}\\results\\phase4_heston_batch_progress.csv")
prog["date"] = pd.to_datetime(prog["date"])

d = pd.Timestamp("2017-10-16")
row = prog[prog["date"] == d].iloc[0]
params = [row["kappa"], row["theta"], row["xi"], row["rho"], row["v0"]]
print("params:", params)

spot = df_year.loc[df_year["date"] == d, "spy_close"].iloc[0]
r3mo = vas.loc[vas["date"] == d, "r3mo"].iloc[0]
eval_date, calendar, day_count, r_ts, q_ts = build_curves(d, r3mo, vas_params)
contracts = select_chain(df_year, d)

prices = heston_model_prices(params, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts)
for p, r in zip(prices, contracts.itertuples(index=False)):
    if not np.isfinite(p) or p <= 0:
        print("FAILED PRICE:", r.strike, r.T, r.moneyness, r.is_call, "price=", p, "market_mid=", r.mid)
        continue
    iv = bs_implied_vol(p, spot, r.strike, r.T, r3mo, Q_DIV, r.is_call)
    if not np.isfinite(iv):
        intrinsic = max(spot - r.strike, 0) if r.is_call else max(r.strike - spot, 0)
        print("FAILED IV SOLVE:", r.strike, r.T, r.moneyness, r.is_call, "price=", p,
              "intrinsic=", intrinsic, "price-intrinsic=", p - intrinsic, "market_mid=", r.mid)
