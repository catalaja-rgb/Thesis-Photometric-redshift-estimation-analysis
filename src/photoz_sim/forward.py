import numpy as np

def redshift_sed(wavelengths: np.ndarray, S_rest: np.ndarray, z: float) -> np.ndarray:
    wl = wavelengths
    wl_rest = wl / (1.0 + z)
    f_rest = S_rest
    f_shift = np.interp(wl_rest, wl, f_rest, left=0.0, right=0.0)
    return (1.0 / (1.0 + z)) * f_shift

def predict_fluxes(wavelengths: np.ndarray, S_rest: np.ndarray, R: np.ndarray, z: float) -> np.ndarray:
    f_obs = redshift_sed(wavelengths, S_rest, z)
    # Integration in each band
    mu = np.trapz(R * f_obs[None, :], wavelengths, axis=1)
    return mu

def grid_mu(wavelengths: np.ndarray, templates: np.ndarray, R: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    Z = z_grid.size
    T = templates.shape[0]
    B = R.shape[0]
    out = np.zeros((Z, T, B), dtype=float)
    for zi, z in enumerate(z_grid):
        for ti in range(T):
            out[zi, ti] = predict_fluxes(wavelengths, templates[ti], R, z)
    return out
