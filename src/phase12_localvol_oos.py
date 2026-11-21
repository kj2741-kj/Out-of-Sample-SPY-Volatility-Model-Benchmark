"""
Phase 12 OOS rerun: price every Phase 9 contract-scoring (results/phase9_oos_contracts.csv) with
TRUE Dupire local vol (phase12_dupire), parameters frozen at the calibration date.

Per calibration date t: Dupire surface from the stored Phase 5 SABR slices at t, the curve at t,
spot at t. Per scoring date d = t+h: the same local-vol function of (S, calendar time), i.e. the
surface time axis shifted by (d - t); spot, curve and tenor from d. Output adds column
`LocalVol_Dupire` (and keeps the old `LocalVol` column, which is the frozen sticky-strike surface).
Checkpointed every 8 calibration dates; resumable.
"""
import os
import time
import numpy as np
import pandas as pd

from phase4_heston_calibration import build_curves, PROJ, Q_DIV
from phase9_oos_runner import tenor_rate
from phase12_dupire import DupireSurface, price_lv

IN = f"{PROJ}\\results\\phase9_oos_contracts.csv"
OUT = f"{PROJ}\\results\\phase12_localvol_oos_contracts.csv"
GRID = 200


def main():
    c = pd.read_csv(IN)
    for k in ("calib_date", "score_date"):
        c[k] = pd.to_datetime(c[k], format="mixed")
    vas = pd.read_parquet(f"{PROJ}\\data\\vasicek_discount_curve.parquet")
    vp = pd.read_csv(f"{PROJ}\\results\\phase2_vasicek_params.csv").iloc[0]
    sa = pd.read_parquet(f"{PROJ}\\data\\sabr_params_by_date_expiry.parquet")
    sa["date"] = pd.to_datetime(sa["date"])
    spot_t = {}
    for y in range(2010, 2026):
        d = pd.read_parquet(f"{PROJ}\\data\\clean\\spy_options_clean_{y}.parquet", columns=["date", "spy_close"])
        spot_t.update(d.groupby("date")["spy_close"].first().to_dict())

    done = pd.DataFrame()
    if os.path.exists(OUT):
        done = pd.read_csv(OUT)
        for k in ("calib_date", "score_date"):
            done[k] = pd.to_datetime(done[k], format="mixed")
    done_dates = set(done["calib_date"].unique()) if len(done) else set()
    parts = [done] if len(done) else []
    meta = []
    t0 = time.time()
    cal_dates = [t for t in sorted(c["calib_date"].unique()) if t not in done_dates]
    for n, t in enumerate(cal_dates):
        t = pd.Timestamp(t)
        rows = c[c["calib_date"] == t].copy()
        sl = sa[sa["date"] == t]
        vals = np.full(len(rows), np.nan)
        try:
            ev_t, _, _, rts_t, _ = build_curves(t, vas.loc[vas.date == t, "r3mo"].iloc[0], vp)
            dup = DupireSurface(float(spot_t[t]), lambda T: tenor_rate(rts_t, T), Q_DIV, sl.to_dict("records"))
            mv = dup.mesh_vol()
            gshare = dup.guarded_share
        except Exception as ex:
            print(f"{t.date()} surface failed: {ex}", flush=True)
            dup, gshare = None, np.nan
        if dup is not None:
            for sd, grp in rows.groupby("score_date"):
                sd = pd.Timestamp(sd)
                ev, _, _, rts, qts = build_curves(sd, vas.loc[vas.date == sd, "r3mo"].iloc[0], vp)
                surf = dup.ql_surface(ev, shift=(sd - t).days / 365.0)
                for ix, r in grp.iterrows():
                    try:
                        vals[rows.index.get_loc(ix)] = price_lv(surf, float(r["spot"]), rts, qts, ev, float(r["strike"]),
                                                                float(r["T"]), bool(r["is_call"]), grid=GRID, mesh_vol=mv)
                    except RuntimeError:
                        pass
        rows["LocalVol_Dupire"] = vals
        rows["dupire_guarded_share"] = gshare
        parts.append(rows)
        if (n + 1) % 8 == 0 or n == len(cal_dates) - 1:
            pd.concat(parts, ignore_index=True).to_csv(OUT, index=False)
            print(f"[{n+1}/{len(cal_dates)}] {t.date()} {(time.time()-t0)/60:.1f} min", flush=True)
    out = pd.read_csv(OUT)
    assert len(out) == len(c), f"row count mismatch: {len(out)} written vs {len(c)} input"
    print(f"done: {len(out)} rows, Dupire priced {out['LocalVol_Dupire'].notna().sum()}", flush=True)


if __name__ == "__main__":
    main()
