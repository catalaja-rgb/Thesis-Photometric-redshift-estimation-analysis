import numpy as np

LOG2PI = np.log(2.0 * np.pi)

def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-sum-exp."""
    m = np.max(a, axis=axis, keepdims=True)
    return (np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m).squeeze(axis=axis)

def template_fit_one_mle(
    x: np.ndarray,        # (B,)
    sig: np.ndarray,      # (B,)
    mu_grid: np.ndarray,  # (Z,T,B)
    z_grid: np.ndarray,   # (Z,)
):
    Z, T, B = mu_grid.shape
    x = np.asarray(x, float)
    sig = np.asarray(sig, float)
    assert x.shape == (B,)
    assert sig.shape == (B,)

    # weights and constants
    w = 1.0 / (sig**2)             # (B,)
    sumlog = np.sum(np.log(sig**2))
    wx2 = np.sum(w * x * x)

    # Flatten (Z,T,B) -> (ZT,B)
    mu = mu_grid.reshape(Z * T, B)

    # denom = sum_b w_b mu_b^2
    denom = np.sum(w[None, :] * mu * mu, axis=1)          # (ZT,)
    # numer = sum_b w_b x_b mu_b
    numer = np.sum((w * x)[None, :] * mu, axis=1)         # (ZT,)

    # a_hat (>=0)
    a_hat = np.where(denom > 0, numer / denom, 0.0)
    a_hat = np.maximum(a_hat, 0.0)

    # chi2 = sum_b w (x - a mu)^2 = wx2 - 2 a numer + a^2 denom
    chi2 = wx2 - 2.0 * a_hat * numer + (a_hat * a_hat) * denom

    # log-likelihood per (z,t)
    ll = -0.5 * (chi2 + sumlog + B * LOG2PI)              # (ZT,)
    ll_zt = ll.reshape(Z, T)

    # marginalize templates: log p(z|x) ∝ log sum_t exp(ll(z,t))
    log_pz = _logsumexp(ll_zt, axis=1)                    # (Z,)

    zi_mle = int(np.argmax(log_pz))
    z_mle = float(z_grid[zi_mle])
    t_mle = int(np.argmax(ll_zt[zi_mle]))

    return z_mle, t_mle, log_pz

def batch_template_fit_mle(
    ds,
    mu_grid: np.ndarray,
    z_grid: np.ndarray,
    progress_every: int = 200,
):
    x = ds.x
    sig = ds.sigma
    N = x.shape[0]

    z_mle = np.zeros(N, float)
    t_mle = np.zeros(N, int)

    for i in range(N):
        if progress_every and (i % progress_every == 0):
            print(f"[template-fit MLE] {i}/{N}", flush=True)

        zi, ti, _ = template_fit_one_mle(x[i], sig[i], mu_grid, z_grid)
        z_mle[i] = zi
        t_mle[i] = ti

    return z_mle, t_mle