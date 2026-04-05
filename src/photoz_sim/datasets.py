"""
Synthetic dataset generation for photometric redshift experiments.

The main entry point is `make_dataset`, which generates noisy observations,
associated uncertainties, and an SNR-based observation mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Dataset:
    """
    Container for a simulated photometric dataset.

    Attributes
    ----------
    x : np.ndarray
        Noisy observed fluxes with shape (N, B).
    z : np.ndarray
        True redshifts with shape (N,).
    t : np.ndarray
        Template indices with shape (N,).
    a : np.ndarray
        Multiplicative amplitudes with shape (N,).
    sigma : np.ndarray
        Per-band observational uncertainties with shape (N, B).
    mask : np.ndarray
        Boolean detection / usability mask with shape (N, B).
    """

    x: np.ndarray
    z: np.ndarray
    t: np.ndarray
    a: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray


def _normalize_probabilities(probs: np.ndarray, name: str) -> np.ndarray:
    """
    Normalize a probability vector after validating positivity of total mass.

    Parameters
    ----------
    probs : np.ndarray
        Input probability vector.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        Normalized probability vector.
    """
    probs = np.asarray(probs, dtype=float)
    total = float(np.sum(probs))

    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must sum to a positive finite value.")

    return probs / total


def _nearest_redshift_indices(z_samples: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """
    Map continuous sampled redshifts to the nearest points on a discrete grid.

    Parameters
    ----------
    z_samples : np.ndarray
        Continuous redshift samples of shape (N,).
    z_grid : np.ndarray
        Discrete redshift grid of shape (Z,).

    Returns
    -------
    np.ndarray
        Integer grid indices of shape (N,).
    """
    indices = np.searchsorted(z_grid, z_samples, side="left")
    indices = np.clip(indices, 1, len(z_grid) - 1)

    left = z_grid[indices - 1]
    right = z_grid[indices]
    choose_left = (z_samples - left) < (right - z_samples)

    return np.where(choose_left, indices - 1, indices).astype(int)


def _validate_inputs(
    mu_grid: np.ndarray,
    z_grid: np.ndarray,
    z_probs: Optional[np.ndarray],
    t_probs: Optional[np.ndarray],
    z_max: float,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Validate and standardize core dataset-generation inputs.

    Parameters
    ----------
    mu_grid : np.ndarray
        Mean flux grid with shape (Z, T, B).
    z_grid : np.ndarray
        Redshift grid with shape (Z,).
    z_probs : Optional[np.ndarray]
        Optional redshift probabilities over the grid.
    t_probs : Optional[np.ndarray]
        Optional template probabilities.
    z_max : float
        Maximum redshift allowed in sampling.

    Returns
    -------
    tuple
        Validated versions of `(mu_grid, z_grid, z_probs, t_probs)`.
    """
    mu_grid = np.asarray(mu_grid, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)

    if mu_grid.ndim != 3:
        raise ValueError("mu_grid must have shape (Z, T, B).")
    if z_grid.ndim != 1:
        raise ValueError("z_grid must be one-dimensional.")
    if mu_grid.shape[0] != z_grid.shape[0]:
        raise ValueError("z_grid length must match the first dimension of mu_grid.")
    if z_max <= 0:
        raise ValueError("z_max must be strictly positive.")

    if z_probs is not None:
        z_probs = np.asarray(z_probs, dtype=float)
        if z_probs.shape != z_grid.shape:
            raise ValueError("z_probs must have the same shape as z_grid.")

    if t_probs is not None:
        t_probs = np.asarray(t_probs, dtype=float)
        if t_probs.ndim != 1 or t_probs.shape[0] != mu_grid.shape[1]:
            raise ValueError("t_probs must have shape (T,), matching mu_grid.shape[1].")

    positive_entries = mu_grid[mu_grid > 0]
    if positive_entries.size == 0:
        raise ValueError("mu_grid must contain at least one positive entry.")

    return mu_grid, z_grid, z_probs, t_probs


def make_dataset(
    rng: np.random.Generator,
    mu_grid: np.ndarray,
    z_grid: np.ndarray,
    N: int = 500,
    z_max: float = 2.0,
    z_probs: Optional[np.ndarray] = None,
    amp_mean_log10: float = 0.0,
    amp_sigma_log10: float = 0.7,
    a_min: float = 1e-3,
    a_max: float = 1e6,
    sigma0_frac: float = 0.05,
    k: float = 0.02,
    snr_min: Optional[float] = 2.0,
    t_probs: Optional[np.ndarray] = None,
    clip_x_nonneg: bool = False,
) -> Dataset:
    """
    Generate a synthetic photometric dataset from a precomputed flux grid.

    The underlying model is:
        f = a * mu(z, t)
        x = f + epsilon
        epsilon ~ N(0, sigma^2)

    where the observational uncertainty is defined elementwise as:
        sigma = sqrt(sigma0^2 + (k * |f|)^2)

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.
    mu_grid : np.ndarray
        Mean flux grid with shape (Z, T, B), where:
        - Z is the number of redshift grid points,
        - T is the number of templates,
        - B is the number of photometric bands.
    z_grid : np.ndarray
        Redshift grid with shape (Z,).
    N : int, default=500
        Number of samples to generate.
    z_max : float, default=2.0
        Maximum redshift allowed when sampling.
    z_probs : Optional[np.ndarray], default=None
        Optional sampling probabilities over `z_grid`. If None, redshifts are
        sampled uniformly on [0, z_max] and then snapped to the nearest grid point.
    amp_mean_log10 : float, default=0.0
        Mean of log10(amplitude).
    amp_sigma_log10 : float, default=0.7
        Standard deviation of log10(amplitude).
    a_min : float, default=1e-3
        Minimum allowed amplitude.
    a_max : float, default=1e6
        Maximum allowed amplitude.
    sigma0_frac : float, default=0.05
        Baseline noise level, defined relative to a typical flux scale.
    k : float, default=0.02
        Multiplicative noise coefficient.
    snr_min : Optional[float], default=2.0
        Minimum signal-to-noise ratio for a band to be marked as valid in the mask.
        If None, all entries are marked valid.
    t_probs : Optional[np.ndarray], default=None
        Optional sampling probabilities over templates. If None, templates are
        sampled uniformly.
    clip_x_nonneg : bool, default=False
        If True, negative observed fluxes are clipped to zero.

    Returns
    -------
    Dataset
        Simulated dataset with observed fluxes, latent variables, uncertainties,
        and an SNR-based mask.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if a_min <= 0 or a_max <= 0 or a_min > a_max:
        raise ValueError("Amplitude bounds must satisfy 0 < a_min <= a_max.")
    if sigma0_frac < 0:
        raise ValueError("sigma0_frac must be non-negative.")
    if k < 0:
        raise ValueError("k must be non-negative.")
    if snr_min is not None and snr_min < 0:
        raise ValueError("snr_min must be non-negative or None.")

    mu_grid, z_grid, z_probs, t_probs = _validate_inputs(
        mu_grid=mu_grid,
        z_grid=z_grid,
        z_probs=z_probs,
        t_probs=t_probs,
        z_max=z_max,
    )

    Z, T, _ = mu_grid.shape

    if t_probs is None:
        t_probs = np.full(T, 1.0 / T, dtype=float)
    else:
        t_probs = _normalize_probabilities(t_probs, name="t_probs")

    valid_z = z_grid <= z_max
    if not np.any(valid_z):
        raise ValueError("No redshift grid points satisfy z_grid <= z_max.")

    positive_entries = mu_grid[mu_grid > 0]
    mu_scale = float(np.median(positive_entries))
    typical_amplitude = 10.0 ** float(amp_mean_log10)
    sigma0 = float(sigma0_frac * typical_amplitude * mu_scale)

    if z_probs is None:
        z_cont = rng.uniform(0.0, z_max, size=N).astype(float)
        z_indices = _nearest_redshift_indices(z_cont, z_grid)
        z = z_grid[z_indices].astype(float)
    else:
        z_probs = np.where(valid_z, z_probs, 0.0)
        z_probs = _normalize_probabilities(z_probs, name="z_probs")
        z_indices = rng.choice(np.arange(Z), size=N, p=z_probs).astype(int)
        z = z_grid[z_indices].astype(float)

    t = rng.choice(np.arange(T), size=N, p=t_probs).astype(int)

    log10_amplitude = rng.normal(
        loc=amp_mean_log10,
        scale=amp_sigma_log10,
        size=N,
    )
    a = np.clip(10.0 ** log10_amplitude, a_min, a_max).astype(float)

    mu = mu_grid[z_indices, t, :]
    f = a[:, None] * mu

    sigma = np.sqrt(sigma0**2 + (k * np.abs(f)) ** 2)
    x = f + rng.normal(loc=0.0, scale=sigma)

    if clip_x_nonneg:
        x = np.clip(x, 0.0, None)

    if snr_min is None:
        mask = np.ones_like(x, dtype=bool)
    else:
        snr = np.abs(f) / np.maximum(sigma, 1e-30)
        mask = snr >= float(snr_min)

    return Dataset(
        x=x,
        z=z,
        t=t,
        a=a,
        sigma=sigma,
        mask=mask,
    )