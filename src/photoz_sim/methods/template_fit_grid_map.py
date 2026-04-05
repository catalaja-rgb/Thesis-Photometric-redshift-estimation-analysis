import numpy as np

LOG2PI = np.log(2.0 * np.pi)


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m).squeeze(axis=axis)


def template_fit_one_map(
    x: np.ndarray,        # (B,)
    sig: np.ndarray,      # (B,)
    mu_grid: np.ndarray,  # (Z,T,B)
    z_grid: np.ndarray,   # (Z,)
    log_prior_z: np.ndarray = None,  # (Z,) or None
    log_prior_t: np.ndarray = None,  # (T,) or None
):
    """
    Grid-based MAP template fitting for ONE galaxy.
    Model: x_b ~ Normal(a * mu_{z,t,b}, sig_b^2)
    We analytically maximize over a >= 0.

    Posterior (up to constant):
        log p(z,t|x) = log p(x|z,t) + log p(z) + log p(t)

    Returns
    -------
    z_map : float
    t_map : int
    log_pz_post : (Z,) array  (log p(z|x) up to additive constant, marginalized over templates)
    """
    Z, T, B = mu_grid.shape
    x = np.asarray(x, float)
    sig = np.asarray(sig, float)
    assert x.shape == (B,)
    assert sig.shape == (B,)

    if log_prior_z is None:
        log_prior_z = np.zeros(Z, float)
    else:
        log_prior_z = np.asarray(log_prior_z, float)
        assert log_prior_z.shape == (Z,)

    if log_prior_t is None:
        log_prior_t = np.zeros(T, float)
    else:
        log_prior_t = np.asarray(log_prior_t, float)
        assert log_prior_t.shape == (T,)

    # weights
    w = 1.0 / (sig**2)             # (B,)
    sumlog = np.sum(np.log(sig**2))
    wx2 = np.sum(w * x * x)

    mu = mu_grid.reshape(Z * T, B)

    denom = np.sum(w[None, :] * mu * mu, axis=1)          # (ZT,)
    numer = np.sum((w * x)[None, :] * mu, axis=1)         # (ZT,)

    a_hat = np.where(denom > 0, numer / denom, 0.0)
    a_hat = np.maximum(a_hat, 0.0)

    chi2 = wx2 - 2.0 * a_hat * numer + (a_hat * a_hat) * denom

    ll = -0.5 * (chi2 + sumlog + B * LOG2PI)              # (ZT,)
    ll_zt = ll.reshape(Z, T)

    # add priors
    log_post_zt = ll_zt + log_prior_z[:, None] + log_prior_t[None, :]

    # marginalize over templates to get log p(z|x)
    log_pz_post = _logsumexp(log_post_zt, axis=1)         # (Z,)

    zi_map = int(np.argmax(log_pz_post))
    z_map = float(z_grid[zi_map])
    t_map = int(np.argmax(log_post_zt[zi_map]))

    return z_map, t_map, log_pz_post


def batch_template_fit_map(
    ds,
    mu_grid: np.ndarray,
    z_grid: np.ndarray,
    log_prior_z: np.ndarray = None,
    log_prior_t: np.ndarray = None,
    progress_every: int = 200,
):
    x = ds.x
    sig = ds.sigma
    N = x.shape[0]

    z_map = np.zeros(N, float)
    t_map = np.zeros(N, int)

    for i in range(N):
        if progress_every and (i % progress_every == 0):
            print(f"[template-fit MAP] {i}/{N}", flush=True)

        zi, ti, _ = template_fit_one_map(
            x[i], sig[i], mu_grid, z_grid,
            log_prior_z=log_prior_z,
            log_prior_t=log_prior_t,
        )
        z_map[i] = zi
        t_map[i] = ti

    print(f"[template-fit MAP] {N}/{N} done", flush=True)
    return z_map, t_map