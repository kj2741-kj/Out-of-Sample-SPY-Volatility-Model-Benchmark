"""
Phase 8: Merton (1976) Jump-Diffusion pricer. Closed-form: a Poisson-weighted sum of ordinary
Black-Scholes prices, truncated at a reasonable N (Poisson weights decay factorially, so this
converges fast and stays numerically well-behaved -- unlike Heston's characteristic-function
quadrature or Andreasen-Huge's PDE calibration, there's no fragile numerical machinery here).
No native QuantLib "JumpDiffusionEngine" is exposed in this build (checked), so hand-implemented
directly on top of a vectorized Black-Scholes -- low risk given the formula is simple.

dS/S = (r - q - lambda*k) dt + sigma dW + dJ,  J = compound Poisson, ln(1+jump) ~ N(mu_j, sigma_j^2)
k = E[jump size] = exp(mu_j + 0.5*sigma_j^2) - 1  (compensator, keeps risk-neutral drift correct)

Vectorized over the n=0..N_TERMS-1 summation (all terms computed in one batch of numpy ops, not
a Python loop) -- this was the actual bottleneck (a plain Python loop made even a single-date
calibration take minutes; batch would have taken hours). Still loops over contracts in Python
(small, ~45/date) since S,K,T,type differ per contract.
"""
import numpy as np
from scipy.stats import norm
from scipy.special import gammaln

N_TERMS = 25  # Poisson weights decay factorially; safe within this project's lambda bounds
              # ([0,5]/yr) x T (<=3yr) -- max lambda*T=15, needs ~30 terms for 5-sigma tail
              # coverage in the worst case, 25 keeps meaningful margin for typical (much lower)
              # calibrated lambda while roughly halving pricer cost vs the original 50


def merton_price(S, K, T, r, q, sigma, lam, mu_j, sigma_j, is_call, n_terms=N_TERMS):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    lam_prime = lam * (1 + k)

    n = np.arange(n_terms)
    log_poisson_w = -lam_prime * T + n * np.log(max(lam_prime * T, 1e-300)) - gammaln(n + 1)
    w = np.exp(log_poisson_w)

    sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
    r_n = r - lam * k + (n * np.log(1 + k) / T if k > -1 else 0.0)

    d1 = (np.log(S / K) + (r_n - q + 0.5 * sigma_n**2) * T) / (sigma_n * np.sqrt(T))
    d2 = d1 - sigma_n * np.sqrt(T)
    if is_call:
        bs_n = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r_n * T) * norm.cdf(d2)
    else:
        bs_n = K * np.exp(-r_n * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return float(np.sum(w * bs_n))


def merton_price_batch(S, K, T, r, q, sigma, lam, mu_j, sigma_j, is_call, n_terms=N_TERMS):
    """Same formula, vectorized across a whole contract array at once (K, T, is_call are
    arrays of length M; S, r, q, sigma, lam, mu_j, sigma_j are scalars for one calibration
    date/parameter-vector). Eliminates the Python-level per-contract loop that made a single
    date's calibration take ~130s -- this is what actually made Phase 8 batch-feasible, the
    earlier per-contract-vectorized-over-n version wasn't enough on its own."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    M = len(K)

    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    lam_prime = lam * (1 + k)

    n = np.arange(n_terms)[np.newaxis, :]          # (1, n_terms)
    Tc = T[:, np.newaxis]                            # (M, 1)
    Kc = K[:, np.newaxis]
    # r may be scalar OR a per-contract array (tenor-matched zero rates, audit fix 2026-08-09).
    # Reshape to (M,1) so it broadcasts against the (M, n_terms) jump-term grid; a bare (M,)
    # array would fail to broadcast and a scalar must stay scalar.
    rc = np.asarray(r, dtype=float)
    rc = rc[:, np.newaxis] if rc.ndim == 1 else rc

    lam_prime_T = np.maximum(lam_prime * Tc, 1e-300)
    log_poisson_w = -lam_prime * Tc + n * np.log(lam_prime_T) - gammaln(n + 1)
    w = np.exp(log_poisson_w)                        # (M, n_terms)

    sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / Tc)
    r_n = rc - lam * k + (n * np.log(1 + k) / Tc if k > -1 else 0.0)

    d1 = (np.log(S / Kc) + (r_n - q + 0.5 * sigma_n**2) * Tc) / (sigma_n * np.sqrt(Tc))
    d2 = d1 - sigma_n * np.sqrt(Tc)

    call_price = S * np.exp(-q * Tc) * norm.cdf(d1) - Kc * np.exp(-r_n * Tc) * norm.cdf(d2)
    put_price = Kc * np.exp(-r_n * Tc) * norm.cdf(-d2) - S * np.exp(-q * Tc) * norm.cdf(-d1)
    bs_n = np.where(is_call[:, np.newaxis], call_price, put_price)  # (M, n_terms)

    return np.sum(w * bs_n, axis=1)  # (M,)
