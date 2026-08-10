"""
Phase 5: SABR per-expiry smile calibration. Design decisions:

1. beta FIXED at 1.0 (lognormal SABR), not jointly fitted with alpha. Same remedy as Phase 2's
   Vasicek a/b and Phase 4's Heston kappa: alpha and beta trade off against each other and are
   poorly identified jointly from a single smile slice -- standard industry practice (and
   Hagan's own original paper) is to fix beta a priori. beta=1 suits an equity index (returns
   are roughly percentage-based) better than beta=0 (more common for rates).
2. Formula: QuantLib's ql.sabrVolatility(strike, forward, T, alpha, beta, nu, rho) -- Hagan's
   closed-form approximation, production-tested C++ implementation. Not hand-rolled, to avoid
   the real risk of a subtle transcription error in a formula with this many terms.
3. Objective: vega-weighted squared IV error, same choice as Heston for consistency and the
   same rationale (down-weight uninformative low-vega deep ITM/OTM points).
4. Fit PER EXPIRY independently (alpha, rho, nu), not one shared set across the whole date's
   surface -- this is SABR's whole point (a separate smile-fit per maturity slice), unlike
   Heston's single full-surface fit.
5. No warm-starting needed: each per-expiry fit is a small, fast, well-behaved 3-parameter
   closed-form optimization (no numerical integration/PDE like Heston), unlikely to need the
   elaborate cold/warm/fallback infrastructure Heston required.
"""
import numpy as np
import QuantLib as ql
from scipy.optimize import minimize

BETA = 1.0
BOUNDS = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]  # alpha, rho, nu


def sabr_iv(strike, forward, T, alpha, rho, nu, beta=BETA):
    return ql.sabrVolatility(strike, forward, T, alpha, beta, nu, rho)


def fit_sabr_slice(strikes, forward, T, market_ivs, vegas):
    atm_iv_seed = np.interp(forward, strikes, market_ivs)
    x0 = [max(atm_iv_seed, 1e-3), -0.5, 0.5]

    def resid_sq_sum(x):
        alpha, rho, nu = x
        err = 0.0
        for k, iv, veg in zip(strikes, market_ivs, vegas):
            try:
                model_iv = sabr_iv(k, forward, T, alpha, rho, nu)
            except RuntimeError:
                return 1e6
            if not np.isfinite(model_iv):
                return 1e6
            err += max(veg, 1e-4) * (model_iv - iv) ** 2
        return err

    res = minimize(resid_sq_sum, x0, method="L-BFGS-B", bounds=BOUNDS,
                    options=dict(maxiter=200, eps=1e-6))
    alpha, rho, nu = res.x
    model_ivs = [sabr_iv(k, forward, T, alpha, rho, nu) for k in strikes]
    iv_errors = np.array(model_ivs) - np.array(market_ivs)
    return dict(alpha=alpha, rho=rho, nu=nu, beta=BETA, objective=res.fun,
                converged=bool(res.success), n_strikes=len(strikes),
                mean_iv_error=np.mean(np.abs(iv_errors)), max_iv_error=np.max(np.abs(iv_errors)))
