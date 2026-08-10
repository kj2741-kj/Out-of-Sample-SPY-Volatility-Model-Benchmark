"""
Phase 3: baseline vanilla option pricers, shared by every later phase.

- bs_price          European Black-Scholes-Merton (spot-based, continuous dividend yield q)
- black76_price     Black-76 (forward-based). Mathematically the same model as BSM
                     reparametrized on the forward F=S*exp((r-q)T) -- included as its own
                     pricer specifically so cross-validation against bs_price is a real check
                     of both implementations, not just one formula copy-pasted twice.
- gbm_mc_price      GBM Monte Carlo, terminal-value sampling (exact for GBM, no need to
                     simulate the full path for a European payoff) with antithetic variates.
- crr_binomial_price Cox-Ross-Rubinstein binomial tree, is_american toggle for early exercise.
- baw_american_price Barone-Adesi-Whaley (1987) closed-form American approximation, quadratic
                     approximation to the early-exercise premium. A known approximation, not
                     exact -- cross-validated against CRR-American in phase 3's script.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S, K, T, r, q, sigma, is_call):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_implied_vol(price, S, K, T, r, q, is_call):
    """Invert bs_price for sigma via Brent's method. Used to read any model's price
    (Heston, SABR, SVI, market mid) back into IV-space for apples-to-apples comparison."""
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


def black76_price(F, K, T, r, sigma, is_call):
    if T <= 0 or sigma <= 0:
        payoff = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return np.exp(-r * T) * payoff
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if is_call:
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def gbm_mc_price(S, K, T, r, q, sigma, is_call, n_paths=200_000, seed=0):
    if T <= 0 or sigma <= 0:
        payoff = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return payoff, 0.0
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    z = rng.standard_normal(half)
    z = np.concatenate([z, -z])  # antithetic variates
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    S_T = S * np.exp(drift + diffusion)
    payoff = np.maximum(S_T - K, 0.0) if is_call else np.maximum(K - S_T, 0.0)
    disc_payoff = np.exp(-r * T) * payoff
    price = disc_payoff.mean()
    stderr = disc_payoff.std(ddof=1) / np.sqrt(len(disc_payoff))
    return price, stderr


def crr_binomial_price(S, K, T, r, q, sigma, is_call, is_american, n_steps=500):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0 < p < 1):
        raise ValueError(f"Risk-neutral probability out of bounds: p={p:.4f} (check sigma/dt/rate inputs)")

    j = np.arange(n_steps + 1)
    S_T = S * (u ** (n_steps - j)) * (d ** j)
    values = np.maximum(S_T - K, 0.0) if is_call else np.maximum(K - S_T, 0.0)

    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if is_american:
            j = np.arange(step + 1)
            S_t = S * (u ** (step - j)) * (d ** j)
            intrinsic = np.maximum(S_t - K, 0.0) if is_call else np.maximum(K - S_t, 0.0)
            values = np.maximum(values, intrinsic)
    return values[0]


def baw_american_price(S, K, T, r, q, sigma, is_call):
    """Barone-Adesi-Whaley (1987) quadratic approximation. Falls back to BS if q<=0 and
    is_call (no early-exercise premium for calls with no dividend -- American=European)."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if is_call and q <= 0:
        return bs_price(S, K, T, r, q, sigma, True)

    M = 2 * r / sigma**2
    N = 2 * (r - q) / sigma**2
    K_baw = 1 - np.exp(-r * T)
    sqrt_term = np.sqrt((N - 1) ** 2 + 4 * M / K_baw)

    def d1(s):
        return (np.log(s / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if is_call:
        q2 = (-(N - 1) + sqrt_term) / 2

        def f(s):
            c = bs_price(s, K, T, r, q, sigma, True)
            return s - K - c - (1 - np.exp(-q * T) * norm.cdf(d1(s))) * s / q2

        s_lo, s_hi = K * 1.0001, K * 50
        if f(s_lo) > 0:
            s_star = s_lo
        else:
            f_hi = f(s_hi)
            while f_hi < 0 and s_hi < K * 1e6:
                s_hi *= 2
                f_hi = f(s_hi)
            s_star = brentq(f, s_lo, s_hi, xtol=1e-8, maxiter=200)
        if S >= s_star:
            return S - K
        c_at_star = bs_price(s_star, K, T, r, q, sigma, True)
        A2 = (s_star / q2) * (1 - np.exp(-q * T) * norm.cdf(d1(s_star)))
        c_bs = bs_price(S, K, T, r, q, sigma, True)
        return c_bs + A2 * (S / s_star) ** q2
    else:
        q1 = (-(N - 1) - sqrt_term) / 2

        def f(s):
            p = bs_price(s, K, T, r, q, sigma, False)
            return K - s - p + (1 - np.exp(-q * T) * norm.cdf(-d1(s))) * s / q1

        s_lo, s_hi = K * 1e-4, K * 0.9999
        if f(s_hi) > 0:
            s_star = s_hi
        else:
            f_lo = f(s_lo)
            while f_lo < 0 and s_lo > K * 1e-8:
                s_lo /= 2
                f_lo = f(s_lo)
            s_star = brentq(f, s_lo, s_hi, xtol=1e-8, maxiter=200)
        if S <= s_star:
            return K - S
        p_at_star = bs_price(s_star, K, T, r, q, sigma, False)
        A1 = -(s_star / q1) * (1 - np.exp(-q * T) * norm.cdf(-d1(s_star)))
        p_bs = bs_price(S, K, T, r, q, sigma, False)
        return p_bs + A1 * (S / s_star) ** q1
