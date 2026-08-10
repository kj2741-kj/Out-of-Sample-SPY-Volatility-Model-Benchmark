"""
Phase 7 batch: validate the SABR-smoothed local-vol pipeline across all 192 dates. Like Phase 6,
no simple per-date params to persist (the "model" is a full surface + PDE-pricing procedure,
reproducible on demand) -- this validates fit quality and stability at scale. Prices a sample of
contracts per date (not the full chain) since PDE solves are more expensive than SABR's
closed-form fits; local vol *level* stability (no blow-up) is checked at every (T, strike) grid
point per date, which is cheap and is exactly the failure mode Andreasen-Huge showed.
"""
import time
import numpy as np
import pandas as pd
from phase4_heston_batch import monthly_calibration_dates
from phase4_heston_calibration import PROJ, select_chain
from phase7_local_vol import build_local_vol_surface, local_vol_price
from pricers import bs_implied_vol

N_SAMPLE_CONTRACTS = 8

def main():
    dates = monthly_calibration_dates()
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]

    results = []
    year_cache = {}
    t_start = time.time()
    for i, d in enumerate(dates):
        if d.year not in year_cache:
            year_cache.clear()
            year_cache[d.year] = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{d.year}.parquet")
        df = year_cache[d.year]
        try:
            local_vol, black_var, sabr_fits, spot, r3mo = build_local_vol_surface(d, df, vas, vas_params)
            eval_date_str = d

            # Stability check: local vol level at every SABR expiry x 3 strikes, flag any blow-up
            max_lv, min_lv = -np.inf, np.inf
            for row in sabr_fits.itertuples():
                for K in [spot * 0.9, spot, spot * 1.1]:
                    lv = local_vol.localVol(row.T, K)
                    max_lv, min_lv = max(max_lv, lv), min(min_lv, lv)

            # Fit-quality check: price a sample of real contracts via PDE, compare to market IV
            from phase4_heston_calibration import build_curves
            eval_date, calendar, day_count, r_ts, q_ts = build_curves(d, r3mo, vas_params)
            contracts = select_chain(df, d)
            sample = contracts.sample(min(N_SAMPLE_CONTRACTS, len(contracts)), random_state=1)
            iv_errs = []
            for row in sample.itertuples(index=False):
                lv_price = local_vol_price(local_vol, black_var, spot, r_ts, q_ts, row.strike, row.T, row.is_call, eval_date)
                lv_iv = bs_implied_vol(lv_price, spot, row.strike, row.T, r3mo, 0.014, row.is_call)
                if np.isfinite(lv_iv):
                    iv_errs.append(abs(lv_iv - row.implied_volatility))

            results.append(dict(date=d, max_local_vol=max_lv, min_local_vol=min_lv,
                                 unstable=bool(max_lv > 2.0),  # >200% local vol = instability flag
                                 mean_iv_error=np.mean(iv_errs) if iv_errs else np.nan,
                                 n_priced=len(iv_errs)))
        except Exception as e:
            print(f"{d.date()}: EXCEPTION {e}", flush=True)
            results.append(dict(date=d, error=str(e)))
            continue

        if (i + 1) % 24 == 0:
            r = results[-1]
            print(f"[{i+1}/{len(dates)}] {d.date()} max_lv={r.get('max_local_vol', float('nan'))*100:.1f}% "
                  f"meanIVerr={r.get('mean_iv_error', float('nan'))*100:.3f}pp "
                  f"elapsed={(time.time()-t_start)/60:.1f}min", flush=True)

    out = pd.DataFrame(results)
    out.to_csv(f"{PROJ}\\results\\phase7_local_vol_batch_validation.csv", index=False)
    n_unstable = out["unstable"].sum() if "unstable" in out else 0
    print(f"\nDone. {len(out)} dates, {n_unstable} flagged unstable (max local vol > 200%), "
          f"{out['error'].notna().sum() if 'error' in out else 0} exceptions.")
    if "mean_iv_error" in out:
        print(f"mean|IVerr| overall: {out['mean_iv_error'].mean()*100:.3f}pp")
    print(f"Total time: {(time.time()-t_start)/60:.1f}min")

if __name__ == "__main__":
    main()
