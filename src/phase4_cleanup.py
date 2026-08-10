"""
Phase 4 cleanup, run after the fixed objective() (see phase4_heston_calibration.py --
clamps near-zero-noise Heston prices instead of slamming a flat +10 penalty on them).

Two categories of rows to fix in results/phase4_heston_batch_progress.csv:
1. Phantom +10 penalty dates (objective > 5): params are actually fine (verified: same
   params, fixed objective drops from ~10 to ~0.001) -- just recompute the objective value,
   no re-optimization needed.
2. Genuinely non-converged dates (converged=False): real optimizer failures, need fresh
   cold-start re-optimization with the fixed objective function.
"""
import numpy as np
import pandas as pd
from phase4_heston_calibration import calibrate_one_date, build_curves, select_chain, objective, PROJ

def main():
    prog = pd.read_csv(f"{PROJ}\\results\\phase4_heston_batch_progress.csv")
    prog["date"] = pd.to_datetime(prog["date"])
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vas_params = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]

    phantom = prog[prog["objective"] > 5].copy()
    nonconverged = prog[prog["converged"] == False].copy()
    print(f"{len(phantom)} phantom-penalty rows, {len(nonconverged)} non-converged rows "
          f"({len(set(phantom['date']) & set(nonconverged['date']))} overlap)")

    year_cache = {}
    def get_year_df(year):
        if year not in year_cache:
            year_cache.clear()
            year_cache[year] = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{year}.parquet")
        return year_cache[year]

    # Category 1: phantom-penalty rows -- recompute objective with existing params, no re-fit.
    for idx, row in phantom.iterrows():
        d = row["date"]
        df_year = get_year_df(d.year)
        spot = df_year.loc[df_year["date"] == d, "spy_close"].iloc[0]
        r3mo = vas.loc[vas["date"] == d, "r3mo"].iloc[0]
        eval_date, calendar, day_count, r_ts, q_ts = build_curves(d, r3mo, vas_params)
        contracts = select_chain(df_year, d)
        params = [row["kappa"], row["theta"], row["xi"], row["rho"], row["v0"]]
        val, n_valid = objective(params, spot, contracts, eval_date, calendar, day_count, r_ts, q_ts, r3mo)
        old_obj = prog.loc[prog["date"] == d, "objective"].iloc[0]
        prog.loc[prog["date"] == d, "objective"] = val
        print(f"{d.date()}: objective corrected {old_obj:.5f} -> {val:.5f} (params unchanged)", flush=True)
        prog.to_csv(f"{PROJ}\\results\\phase4_heston_batch_progress.csv", index=False)

    # Category 2: genuinely non-converged -- fresh cold-start re-optimization with the fix.
    # Skip dates already re-fixed by a prior (killed) run of this script -- check the
    # "cleaned" marker column so re-running after a kill doesn't redo completed work.
    if "cleanup_done" not in prog.columns:
        prog["cleanup_done"] = False

    for idx, row in nonconverged.iterrows():
        d = row["date"]
        if prog.loc[prog["date"] == d, "cleanup_done"].iloc[0]:
            print(f"{d.date()}: already cleaned up in a prior run, skipping", flush=True)
            continue
        df_year = get_year_df(d.year)
        print(f"\n{d.date()}: re-calibrating cold (was converged=False, obj={row['objective']:.5f})...", flush=True)
        res = calibrate_one_date(d, df_year, vas, vas_params, warm_start_x=None, verbose=True)
        old_obj, old_conv = row["objective"], row["converged"]
        if res.get("converged") and (not old_conv or res["objective"] < old_obj):
            for col in ["kappa", "theta", "xi", "rho", "v0", "feller", "objective",
                        "converged", "mode", "mean_iv_error"]:
                prog.loc[prog["date"] == d, col] = res.get(col)
            prog.loc[prog["date"] == d, "fallback_triggered"] = True
            print(f"  IMPROVED: converged={res['converged']} obj={res['objective']:.5f} "
                  f"meanIVerr={res['mean_iv_error']*100:.2f}pp", flush=True)
        else:
            print(f"  NO IMPROVEMENT (new obj={res.get('objective', np.nan):.5f} conv={res.get('converged')}) "
                  f"-- keeping original, flagging as genuinely hard to fit", flush=True)
        prog.loc[prog["date"] == d, "cleanup_done"] = True
        prog.to_csv(f"{PROJ}\\results\\phase4_heston_batch_progress.csv", index=False)

    prog.to_parquet(f"{PROJ}\\data\\heston_params_by_date.parquet", index=False)

    n_conv = prog["converged"].sum()
    n_phantom_left = (prog["objective"] > 5).sum()
    print(f"\nFinal: {n_conv}/{len(prog)} converged, {n_phantom_left} phantom-penalty rows remaining")

if __name__ == "__main__":
    main()
