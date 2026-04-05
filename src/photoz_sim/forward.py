import numpy as np

def redshift_sed(wavelengths: np.ndarray, S_rest: np.ndarray, z: float) -> np.ndarray:
    """
    Apply cosmological redshift to a rest-frame spectral energy distribution.

    This implements:
        S_obs(λ) = (1 / (1 + z)) * S_rest(λ / (1 + z))

    where the (1+z) factor accounts for photon stretching.
    """
    wl = wavelengths
    wl_rest = wl / (1.0 + z)

    # Interpolate rest-frame SED onto shifted grid
    f_shift = np.interp(wl_rest, wl, S_rest, left=0.0, right=0.0)

    return (1.0 / (1.0 + z)) * f_shift


def predict_fluxes(
    wavelengths: np.ndarray,
    S_rest: np.ndarray,
    R: np.ndarray,
    z: float
) -> np.ndarray:
    """
    Compute photometric fluxes through a set of filters.

    Each band is obtained via numerical integration:
        x_b = ∫ S_obs(λ) R_b(λ) dλ

    using a trapezoidal rule on the wavelength grid.
    """
    f_obs = redshift_sed(wavelengths, S_rest, z)

    # Integrate flux within each filter band
    return np.trapz(R * f_obs[None, :], wavelengths, axis=1)


def grid_mu(
    wavelengths: np.ndarray,
    templates: np.ndarray,
    R: np.ndarray,
    z_grid: np.ndarray
) -> np.ndarray:
    """
    Precompute model fluxes over a (z, template) grid.

    Output shape:
        (n_z, n_templates, n_filters)

    This forms the forward model lookup table used in
    template-fitting inference.
    """
    n_z = z_grid.size
    n_templates = templates.shape[0]
    n_filters = R.shape[0]

    mu = np.zeros((n_z, n_templates, n_filters), dtype=float)

    for zi, z in enumerate(z_grid):
        for ti in range(n_templates):
            mu[zi, ti] = predict_fluxes(
                wavelengths,
                templates[ti],
                R,
                z
            )

    return mu
