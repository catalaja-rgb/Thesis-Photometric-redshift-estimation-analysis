"""
Tipos de galaxias
Esto responde a ¿qué formas distintas puede tener un espectro?
Lo que hacemos:
 - Inventamos 6 espectros falsos
 - Cada uno tiene:
    · Una pendiente distinta
    · Un bultito
    · Un breal
No son galaxias reales, son jueguetes

Mentalmente:
Galaxia tipo 0, tipo 1, tipo 2...
"""
import numpy as np

def log_gaussian_like_with_amplitude(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian log-likelihood after fitting amplitude a for each candidate mu.

    x: (B,)
    mu: (..., B)
    sigma: (B,)  (per-band std)
    Returns:
      logL: (...,)
      a_hat: (...,)   best-fit amplitudes
    """
    w = 1.0 / (sigma**2)                 # (B,)
    # Compute a_hat = argmin_a sum_b w_b (x_b - a mu_b)^2
    # a_hat = (sum w mu x) / (sum w mu^2)
    num = np.sum(mu * (w[None, :] * x[None, :]), axis=-1)
    den = np.sum(mu * (w[None, :] * mu), axis=-1) + 1e-300
    a_hat = num / den

    resid = x[None, :] - a_hat[..., None] * mu
    chi2 = np.sum((resid**2) * w[None, :], axis=-1)

    # logL ∝ -0.5 chi2 - sum log sigma  (constants ignored)
    logL = -0.5 * chi2 - np.sum(np.log(sigma))
    return logL, a_hat

def log_gaussian_likelihood(x: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """
    Si yo esperaba ver mu pero he observado x,
    ¿how likely is this, assuming gaussian noise
    with deviation sigma?
    Entradas:
    - x: observed fluxes
    - mu: model prediction
    - sigma: noise/error
    Salida:
    log-likelihood array for each mu
    """
    B = x.shape[-1]
    resid2 = np.sum((mu - x[None, :]) ** 2, axis=-1)
    return -0.5 * resid2 / (sigma ** 2) - B * np.log(sigma)  # constants ignored except sigma term

def fit_template_grid(x, mu_grid, z_grid, sigma, log_prior_zt=None):
    Z, T, B = mu_grid.shape
    assert x.shape == (B,)

    # make sigma a vector (B,)
    if np.isscalar(sigma):
        sigma_vec = np.full(B, float(sigma))
    else:
        sigma_vec = np.asarray(sigma, dtype=float)
        assert sigma_vec.shape == (B,)

    logL_zt = np.zeros((Z, T), dtype=float)
    a_zt = np.zeros((Z, T), dtype=float)

    for ti in range(T):
        logL_zt[:, ti], a_zt[:, ti] = log_gaussian_like_with_amplitude(
            x=x,
            mu=mu_grid[:, ti, :],   # (Z,B)
            sigma=sigma_vec
        )

    # MLE in (z,t)
    idx = np.unravel_index(np.argmax(logL_zt), logL_zt.shape)
    z_mle = float(z_grid[idx[0]])
    t_mle = int(idx[1])
    a_mle = float(a_zt[idx])

    # posterior on (z,t)
    if log_prior_zt is None:
        log_post_zt = logL_zt
    else:
        assert log_prior_zt.shape == (Z, T)
        log_post_zt = logL_zt + log_prior_zt

    # marginalize templates: p(z|x)
    amax = np.max(log_post_zt, axis=1, keepdims=True)
    pz_unnorm = np.sum(np.exp(log_post_zt - amax), axis=1) * np.exp(amax[:, 0])
    p_z = pz_unnorm / np.sum(pz_unnorm)

    z_map = float(z_grid[int(np.argmax(p_z))])

    return {
        "z_mle": z_mle,
        "t_mle": t_mle,
        "a_mle": a_mle,
        "logL_zt": logL_zt,
        "p_z": p_z,
        "z_map": z_map,
        "a_zt": a_zt,
    }


def simple_redshift_prior(z_grid: np.ndarray, alpha: float = 2.0, z0: float = 0.7) -> np.ndarray:
    """
    A simple analytic prior over z: p(z) ∝ z^alpha * exp(-z/z0).
    Returns normalized prior over z grid.
    """
    z = z_grid
    p = (z ** alpha) * np.exp(-z / z0)
    p[0] = 0.0  # avoid odd behavior at z=0 if alpha>0
    p = p / np.sum(p)
    return p

def expand_prior_to_zt(prior_z: np.ndarray, T: int) -> np.ndarray:
    """
    Make log prior over (z,t) assuming uniform over templates:
      p(z,t)=p(z)/T
    """
    Z = prior_z.size
    prior_zt = np.repeat(prior_z[:, None], T, axis=1) / T 
    return np.log(prior_zt + 1e-300)

