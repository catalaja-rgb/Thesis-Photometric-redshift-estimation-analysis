import numpy as np
from typing import Optional


LOG2PI = np.log(2.0 * np.pi)

def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m).squeeze(axis=axis)

def template_fit_one(
    x: np.ndarray,            # (B,)
    sig: np.ndarray,          # (B,)
    mu_grid: np.ndarray,      # (Z,T,B)
    z_grid: np.ndarray,       # (Z,)
    prior_z: Optional[np.ndarray] = None
,  # (Z,)
):
    """
    Vectorized grid fit for ONE galaxy. Marginalizes over templates.
    Returns: z_mle, z_map, z_mean0, z_mean1, t_mle, t_map
    """
    Z, T, B = mu_grid.shape
    assert x.shape == (B,)
    assert sig.shape == (B,)

    w = 1.0 / (sig**2)                     # (B,)
    sumlog = np.sum(np.log(sig**2))        # scalar
    wx2 = np.sum(w * x * x)                # scalar

    # Flatten (Z,T,B) -> (ZT,B)
    mu = mu_grid.reshape(Z * T, B)

    # denom = sum_b w_b mu_b^2
    denom = np.sum(w[None, :] * mu * mu, axis=1)  # (ZT,)

    # numer = sum_b w_b x_b mu_b
    numer = np.sum((w * x)[None, :] * mu, axis=1) # (ZT,)

    # a_hat >= 0
    a_hat = np.where(denom > 0, numer / denom, 0.0)
    a_hat = np.maximum(a_hat, 0.0)                # (ZT,)

    # chi2 = sum_b w (x - a mu)^2
    # Use quadratic form: chi2 = wx2 - 2 a*numer + a^2*denom
    chi2 = wx2 - 2.0 * a_hat * numer + (a_hat * a_hat) * denom

    ll = -0.5 * (chi2 + sumlog + B * LOG2PI)      # (ZT,)
    ll_zt = ll.reshape(Z, T)

    # p(z|x) without prior: log p(z) = log sum_t exp(ll(z,t))
    log_pz_noprior = _logsumexp(ll_zt, axis=1)    # (Z,)

    zi_mle = int(np.argmax(log_pz_noprior))
    z_mle = float(z_grid[zi_mle])
    t_mle = int(np.argmax(ll_zt[zi_mle]))

    # normalize for posterior mean (no prior)
    lp0 = log_pz_noprior - np.max(log_pz_noprior)
    pz0 = np.exp(lp0)
    pz0 /= pz0.sum()
    z_mean0 = float(np.sum(pz0 * z_grid))

    # prior
    if prior_z is None:
        log_prior = np.zeros(Z)
    else:
        prior = np.asarray(prior_z, float)
        prior = np.clip(prior, 1e-300, None)
        prior /= prior.sum()
        log_prior = np.log(prior)

    log_pz_prior = log_pz_noprior + log_prior

    zi_map = int(np.argmax(log_pz_prior))
    z_map = float(z_grid[zi_map])
    t_map = int(np.argmax(ll_zt[zi_map]))

    lp1 = log_pz_prior - np.max(log_pz_prior)
    pz1 = np.exp(lp1)
    pz1 /= pz1.sum()
    z_mean1 = float(np.sum(pz1 * z_grid))

    return z_mle, z_map, z_mean0, z_mean1, t_mle, t_map


def batch_template_fit(ds, mu_grid, z_grid, prior_z=None, progress_every: int = 100):
    """
    Batch version over all galaxies.
    """
    x = ds.x
    sig = ds.sigma

    N, B = x.shape
    z_mle = np.zeros(N)
    z_map = np.zeros(N)
    z_mean0 = np.zeros(N)
    z_mean1 = np.zeros(N)
    t_mle = np.zeros(N, dtype=int)
    t_map = np.zeros(N, dtype=int)

    for i in range(N):
        if progress_every and (i % progress_every == 0):
            print(f"[template-fit] {i}/{N}", flush=True)

        z_mle[i], z_map[i], z_mean0[i], z_mean1[i], t_mle[i], t_map[i] = template_fit_one(
            x[i], sig[i], mu_grid, z_grid, prior_z=prior_z
        )

    print(f"[template-fit] {N}/{N} done", flush=True)
    return z_mle, z_map, z_mean0, z_mean1, t_mle, t_map
