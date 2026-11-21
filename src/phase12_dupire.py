"""
Phase 12: true Dupire local volatility (fix for the Phase 7/9 localVol=False bug).

Phase 7/9 built QuantLib's FdBlackScholesVanillaEngine without localVol=True, so the model named
"LocalVol" priced Black-Scholes at the frozen surface's implied vol by strike (sticky strike).
And the Dupire surface itself (LocalVolSurface on a bilinear BlackVarianceSurface) was too noisy
to reprice its own calibration date. This module builds local vol properly:

1. Smile per calibrated expiry T_i: the stored Phase 5 SABR fit, written as total implied variance
   w_i(y) = sigma_SABR(K = F_i e^y)^2 * T_i in log-moneyness y against that expiry's forward.
2. Calendar consistency: at each y, w is made non-decreasing across expiries (cumulative max).
3. Between expiries w is linear in T at fixed y; before the first expiry w = w_1 * T / T_1 and after
   the last w = w_n * T / T_n (constant implied vol in the wings of time).
4. Local variance from Gatheral's (2006) form of Dupire's formula:
     sigma_loc^2 = w_T / (1 - (y/w) w_y + 0.25 (-0.25 - 1/w + y^2/w^2) w_y^2 + 0.5 w_yy)
   with w_y, w_yy from central differences of the CLOSED-FORM SABR smile (smooth, no surface
   interpolation noise) and w_T the slope between slices.
5. Guard rails: denominator floored at 1e-3 (a negative value means the input smile has
   butterfly arbitrage there), local vol clipped to [0.02, 2.0]; the share of guarded nodes is
   reported. y is evaluated inside [Y_LO, Y_HI]; beyond it the smile is held flat.
6. Pricing: FdBlackScholesVanillaEngine(..., localVol=True) on a FixedLocalVolSurface.
   Out of sample on day d > t the SAME function of (S, calendar time) is used: the surface's
   time axis is shifted by d - t, so the model is genuinely frozen, not re-anchored.
"""
import numpy as np
import QuantLib as ql
from sabr import sabr_iv, BETA

Y_LO, Y_HI = -1.0, 0.6
N_K = 141
H_Y = 1e-3
DENOM_FLOOR = 1e-3
LV_MIN, LV_MAX = 0.02, 2.0


def _w_slice(y, F, T, a, rho, nu):
    yy = np.clip(y, Y_LO, Y_HI)
    out = np.empty_like(yy, dtype=float)
    for k, v in enumerate(yy):
        try:
            iv = sabr_iv(F * np.exp(v), F, T, a, rho, nu, BETA)
        except RuntimeError:
            iv = np.nan
        out[k] = iv * iv * T if np.isfinite(iv) and iv > 0 else np.nan
    # fill any failed points by nearest valid value
    if np.isnan(out).any():
        good = np.where(np.isfinite(out))[0]
        if len(good) == 0:
            raise RuntimeError("SABR slice unusable")
        out = np.interp(np.arange(len(out)), good, out[good])
    return out


class DupireSurface:
    """Local vol on an absolute-strike x time grid, built at calibration date t."""

    def __init__(self, spot, r_of_T, q, slices, t_grid=None):
        """slices: iterable of dicts with T, forward (unused, recomputed), alpha, rho, nu.
        r_of_T: function T -> continuously compounded zero rate (from the curve at t)."""
        sl = sorted(slices, key=lambda s: s["T"])
        self.spot, self.q = spot, q
        self.Ti = np.array([s["T"] for s in sl])
        Fi = np.array([spot * np.exp((r_of_T(T) - q) * T) for T in self.Ti])
        self.strikes = spot * np.exp(np.linspace(Y_LO - 0.2, Y_HI + 0.2, N_K))
        if t_grid is None:
            t_grid = np.unique(np.concatenate([np.geomspace(2 / 365, self.Ti[-1] * 1.5, 60), self.Ti]))
        self.times = t_grid
        nT, nK = len(t_grid), len(self.strikes)
        lv = np.zeros((nK, nT))
        guarded = 0
        total = 0
        # per-slice w, w_y, w_yy as functions evaluated at arbitrary y arrays
        params = [(F, s["T"], s["alpha"], s["rho"], s["nu"]) for F, s in zip(Fi, sl)]
        for j, T in enumerate(t_grid):
            F_T = spot * np.exp((r_of_T(T) - q) * T)
            y = np.log(self.strikes / F_T)
            W = np.array([_w_slice(y, *p) for p in params])            # (n_slices, nK)
            Wp = np.array([_w_slice(y + H_Y, *p) for p in params])
            Wm = np.array([_w_slice(y - H_Y, *p) for p in params])
            # calendar fix at fixed y: total variance non-decreasing in T
            W, Wp, Wm = (np.maximum.accumulate(A, axis=0) for A in (W, Wp, Wm))
            i = np.searchsorted(self.Ti, T)
            if i == 0:
                fac = T / self.Ti[0]
                w, wp, wm = W[0] * fac, Wp[0] * fac, Wm[0] * fac
                wT = W[0] / self.Ti[0]
            elif i >= len(self.Ti):
                fac = T / self.Ti[-1]
                w, wp, wm = W[-1] * fac, Wp[-1] * fac, Wm[-1] * fac
                wT = W[-1] / self.Ti[-1]
            else:
                T0, T1 = self.Ti[i - 1], self.Ti[i]
                a = (T - T0) / (T1 - T0)
                w = (1 - a) * W[i - 1] + a * W[i]
                wp = (1 - a) * Wp[i - 1] + a * Wp[i]
                wm = (1 - a) * Wm[i - 1] + a * Wm[i]
                wT = (W[i] - W[i - 1]) / (T1 - T0)
            wy = (wp - wm) / (2 * H_Y)
            wyy = (wp - 2 * w + wm) / (H_Y * H_Y)
            denom = 1 - (y / w) * wy + 0.25 * (-0.25 - 1 / w + y * y / (w * w)) * wy * wy + 0.5 * wyy
            core = (y > -0.5) & (y < 0.4)
            g = (denom < DENOM_FLOOR) | (wT <= 0)
            var = np.maximum(wT, 1e-8) / np.maximum(denom, DENOM_FLOOR)
            s = np.sqrt(var)
            g |= (s < LV_MIN) | (s > LV_MAX)
            guarded += int((g & core).sum())
            total += int(core.sum())
            lv[:, j] = np.clip(s, LV_MIN, LV_MAX)
        self.lv = lv
        self.guarded_share = guarded / max(total, 1)
        self._params = params
        self._r_of_T = r_of_T

    def _iv_cache_fn(self, K, T):
        F_T = self.spot * np.exp((self._r_of_T(T) - self.q) * T)
        y = np.array([np.log(K / F_T)])
        W = np.maximum.accumulate(np.array([_w_slice(y, *p) for p in self._params]), axis=0)[:, 0]
        i = np.searchsorted(self.Ti, T)
        if i == 0:
            w = W[0] * T / self.Ti[0]
        elif i >= len(self.Ti):
            w = W[-1] * T / self.Ti[-1]
        else:
            a = (T - self.Ti[i - 1]) / (self.Ti[i] - self.Ti[i - 1])
            w = (1 - a) * W[i - 1] + a * W[i]
        return float(np.sqrt(w / T))

    def mesh_vol(self):
        """A vol level wide enough for the PDE domain: 90th percentile of local vol near the money."""
        core = (self.strikes > self.spot * 0.7) & (self.strikes < self.spot * 1.3)
        return float(max(0.3, np.percentile(self.lv[core], 90)))

    def implied_vol(self, K, T):
        """Implied vol of the (calendar-fixed, T-interpolated) input surface at strike K, tenor T.
        This is the sticky-strike surface the Phase 9 'LocalVol' column really priced."""
        return self._iv_cache_fn(K, T)

    def ql_surface(self, ref_date, shift=0.0):
        """FixedLocalVolSurface anchored at ref_date whose time axis is the calibration time axis
        shifted by `shift` years (shift = (d - t)/365 out of sample)."""
        keep = self.times > shift + 1e-6
        times = list(self.times[keep] - shift)
        m = ql.Matrix(len(self.strikes), len(times))
        cols = np.where(keep)[0]
        for i in range(len(self.strikes)):
            for jj, j in enumerate(cols):
                m[i][jj] = float(self.lv[i, j])
        return ql.FixedLocalVolSurface(ref_date, times, list(self.strikes), m, ql.Actual365Fixed())


def effective_T(eval_date, T):
    """Year fraction to the expiry date the pricer actually uses (T rounded to whole days)."""
    ed = eval_date + ql.Period(int(round(T * 365)), ql.Days)
    return ql.Actual365Fixed().yearFraction(eval_date, ed)


def price_lv(surface_ql, spot, r_ts, q_ts, eval_date, K, T, is_call, grid=200, greeks=False, mesh_vol=0.5):
    sq = ql.SimpleQuote(spot)
    # With localVol=True the operator uses ONLY the local vol surface; the Black vol handle is
    # required by the process signature and is used by the mesher to size the spatial domain,
    # so callers set mesh_vol to about the surface's own high vol level (mesh_vol()) rather than a market value.
    dummy_bv = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_date, ql.NullCalendar(), mesh_vol, ql.Actual365Fixed()))
    proc = ql.GeneralizedBlackScholesProcess(ql.QuoteHandle(sq), q_ts, r_ts, dummy_bv,
                                             ql.LocalVolTermStructureHandle(surface_ql))
    eng = ql.FdBlackScholesVanillaEngine(proc, grid, grid, 0, ql.FdmSchemeDesc.Douglas(), True)
    ed = eval_date + ql.Period(int(round(T * 365)), ql.Days)
    o = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K), ql.EuropeanExercise(ed))
    o.setPricingEngine(eng)
    if greeks:
        return o.NPV(), o.delta(), o.gamma()
    return o.NPV()
