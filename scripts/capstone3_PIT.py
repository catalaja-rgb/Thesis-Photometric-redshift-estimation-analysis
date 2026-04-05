import os
import json
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from sklearn.model_selection import train_test_split

from photoz_sim.datasets import make_dataset
from photoz_sim.forward import grid_mu
from photoz_sim.methods.template_fit_grid_mle import batch_template_fit_mle
from photoz_sim.methods.template_fit_grid_map import batch_template_fit_map
from photoz_sim.templates_eazy import load_eazy_templates_from_spectra_param
from photoz_sim.filters_eazy import (
    find_eazy_filters_res,
    load_eazy_filters_res,
    select_filters_in_range,
    auto_wavelength_grid_from_filters,
    build_R_matrix_on_grid,
)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.linewidth": 1.0,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

METHOD_STYLE = {
    "MLE": {"color": "#1b3a57", "marker": "o", "linestyle": "-",  "lw": 1.8, "ms": 5.8},
    "MAP": {"color": "#2a6f97", "marker": "s", "linestyle": "-",  "lw": 1.8, "ms": 5.6},
    "MAP_mismatch": {"color": "#e07b39", "marker": "s", "linestyle": "--", "lw": 1.8, "ms": 5.6},
    "kNN": {"color": "#7a8b5d", "marker": "^", "linestyle": "--", "lw": 1.7, "ms": 5.8},
    "RF":  {"color": "#8c4f5f", "marker": "D", "linestyle": "--", "lw": 1.7, "ms": 5.2},
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig_both(fig, path_no_ext: str) -> None:
    fig.savefig(path_no_ext + ".png")
    fig.savefig(path_no_ext + ".pdf")


def style_axes_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("0.15")
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(False)


def compute_metrics(z_true: np.ndarray, z_pred: np.ndarray) -> dict:
    z_true = np.asarray(z_true, float)
    z_pred = np.asarray(z_pred, float)
    dz = (z_pred - z_true) / (1.0 + z_true)
    return {
        "mae": float(np.mean(np.abs(z_pred - z_true))),
        "bias": float(np.mean(dz)),
        "scatter": float(np.std(dz)),
        "outlier": float(np.mean(np.abs(dz) > 0.15)),
    }


def make_features(x: np.ndarray, x_err: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(x_err, eps, None)


def fit_knn(X_train, y_train, X_test, k=25, weights="distance"):
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=k, weights=weights)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_random_forest(X_train, y_train, X_test,
                      n_estimators=300, random_state=0,
                      max_depth=None, min_samples_leaf=2):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def compute_pit_values(log_pz_array: np.ndarray,
                       z_grid: np.ndarray,
                       z_true: np.ndarray) -> np.ndarray:
    """
    Compute PIT values for a set of posteriors.

    Parameters
    ----------
    log_pz_array : (N, Z) array of log posterior values (unnormalised ok)
    z_grid       : (Z,) redshift grid
    z_true       : (N,) true redshifts

    Returns
    -------
    pit : (N,) array of PIT values in [0, 1]
    """
    log_pz_array = np.asarray(log_pz_array, float)
    z_grid = np.asarray(z_grid, float)
    z_true = np.asarray(z_true, float)

    # Normalise each posterior to a proper probability mass function
    # Subtract max for numerical stability before exponentiating
    log_pz_array = log_pz_array - log_pz_array.max(axis=1, keepdims=True)
    pz = np.exp(log_pz_array)

    # Normalise so each row sums to 1 (using trapezoidal rule)
    norms = np.trapz(pz, z_grid, axis=1)          # (N,)
    pz = pz / norms[:, None]

    # For each galaxy compute CDF up to z_true
    pit = np.zeros(len(z_true), float)
    for i in range(len(z_true)):
        # Find index just past z_true[i]
        idx = np.searchsorted(z_grid, z_true[i], side="right")
        idx = np.clip(idx, 1, len(z_grid))
        # Integrate from 0 to z_true using trapz on the truncated grid
        pit[i] = np.trapz(pz[i, :idx], z_grid[:idx])

    return pit


def build_filters_and_grid(B=5, wlmin_nm=300.0, wlmax_nm=1100.0, step_nm=0.5):
    eazy_root = "data/external/eazy-photoz"
    filters_res = find_eazy_filters_res(eazy_root)
    all_filters = load_eazy_filters_res(str(filters_res))
    chosen_filters = select_filters_in_range(
        all_filters, wlmin_nm=wlmin_nm, wlmax_nm=wlmax_nm, n=B,
    )
    wl = auto_wavelength_grid_from_filters(
        chosen_filters, step_nm=step_nm,
        wl_min_nm_floor=wlmin_nm, wl_max_nm_ceil=wlmax_nm,
    )
    R = build_R_matrix_on_grid(wl, chosen_filters, normalize=True)
    return wl, R


def load_templates(label: str, wl: np.ndarray):
    eazy_root = "data/external/eazy-photoz"
    if label in ("eazy_v1.3", "cww+kin", "pegase13"):
        param_file = f"{eazy_root}/templates/{label}.spectra.param"
        templates, names = load_eazy_templates_from_spectra_param(
            wavelengths_nm=wl,
            spectra_param_file=param_file,
            base_dir=eazy_root,
            max_templates=None,
            normalize="integral",
        )
    else:
        raise ValueError(f"Unknown template label: {label}")
    return templates, names


def make_redshift_prior_probs(z_grid, z_max=2.0, alpha=2.0, z0=0.35):
    """Matched prior — same as used for data generation."""
    z = np.asarray(z_grid, float)
    p = (z ** alpha) * np.exp(-z / z0)
    p = np.where(z <= z_max, p, 0.0)
    p[0] = 0.0
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Prior has zero mass.")
    return p / s


def make_lowz_prior_probs(z_grid, z_max=2.0, alpha=2.0, z0=0.15):
    """
    Misspecified low-z prior.
    Same functional form but z0 is much smaller, concentrating
    probability mass at low redshift and strongly suppressing z > 0.5.
    """
    z = np.asarray(z_grid, float)
    p = (z ** alpha) * np.exp(-z / z0)
    p = np.where(z <= z_max, p, 0.0)
    p[0] = 0.0
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Prior has zero mass.")
    return p / s


def plot_recovery(ax, z_true, z_pred, metrics, title, color="#1b3a57"):
    ax.scatter(z_true, z_pred, s=2.5, alpha=0.35, color=color,
               linewidths=0, rasterized=True)
    lim = [0.0, 2.0]
    ax.plot(lim, lim, "k-", lw=0.9, zorder=5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"True redshift $z_\mathrm{true}$")
    ax.set_ylabel(r"Predicted redshift $\hat{z}$")
    ax.set_title(title, pad=7)
    txt = (f"MAE = {metrics['mae']:.3f}\n"
           f"Bias = {metrics['bias']:.3f}\n"
           f"Scatter = {metrics['scatter']:.3f}\n"
           f"Outliers = {metrics['outlier']:.3f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.7))
    style_axes_box(ax)


def plot_residuals(ax, z_true, z_pred, title, color="#1b3a57"):
    dz = (z_pred - z_true) / (1.0 + z_true)
    ax.scatter(z_true, dz, s=2.5, alpha=0.35, color=color,
               linewidths=0, rasterized=True)
    ax.axhline(0.0,  color="k",    lw=0.9)
    ax.axhline( 0.15, color="0.4", lw=0.8, ls="--")
    ax.axhline(-0.15, color="0.4", lw=0.8, ls="--")
    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([-0.55, 0.55])
    ax.set_xlabel(r"True redshift $z_\mathrm{true}$")
    ax.set_ylabel(r"$\Delta z = (\hat{z}-z_\mathrm{true})/(1+z_\mathrm{true})$")
    ax.set_title(title, pad=7)
    style_axes_box(ax)


def plot_pit_histogram(ax, pit_values, title, color="#1b3a57", n_bins=20):
    """
    Plot PIT histogram with a uniform reference line.
    A perfectly calibrated posterior yields a flat histogram at y=1.
    Deviations reveal systematic miscalibration.
    """
    counts, edges = np.histogram(pit_values, bins=n_bins, range=(0.0, 1.0))
    # Normalise so that a uniform distribution gives height 1.0
    expected = len(pit_values) / n_bins
    heights = counts / expected

    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    ax.bar(centres, heights, width=width * 0.88,
           color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="k", lw=1.0, ls="--", zorder=5, label="Uniform (ideal)")

    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel("PIT value  $u_i$")
    ax.set_ylabel("Normalised frequency")
    ax.set_title(title, pad=7)
    ax.legend(frameon=False, fontsize=9)
    style_axes_box(ax)


def batch_template_fit_mle_with_posteriors(ds, mu_grid, z_grid, progress_every=200):
    """
    Wrapper around existing MLE function that additionally
    returns the full log_pz array (N, Z) needed for PIT computation.
    Calls template_fit_one_mle directly so we can collect log_pz.
    """
    from photoz_sim.methods.template_fit_grid_mle import template_fit_one_mle

    x = ds.x
    sig = ds.sigma
    N = x.shape[0]

    z_mle = np.zeros(N, float)
    t_mle = np.zeros(N, int)
    log_pz_all = np.zeros((N, len(z_grid)), float)

    for i in range(N):
        if progress_every and (i % progress_every == 0):
            print(f"[MLE+posterior] {i}/{N}", flush=True)
        zi, ti, log_pz = template_fit_one_mle(x[i], sig[i], mu_grid, z_grid)
        z_mle[i] = zi
        t_mle[i] = ti
        log_pz_all[i] = log_pz

    return z_mle, t_mle, log_pz_all


def batch_template_fit_map_with_posteriors(ds, mu_grid, z_grid,
                                           log_prior_z, log_prior_t,
                                           progress_every=200):
    """
    Same as above but for MAP — returns full log posterior over z.
    Assumes your template_fit_one_map returns (z_map, t_map, log_post_z).
    If it doesn't, we reconstruct log_post_z = log_pz_likelihood + log_prior_z.
    """
    from photoz_sim.methods.template_fit_grid_mle import template_fit_one_mle

    x = ds.x
    sig = ds.sigma
    N = x.shape[0]

    z_map = np.zeros(N, float)
    t_map = np.zeros(N, int)
    log_post_z_all = np.zeros((N, len(z_grid)), float)

    for i in range(N):
        if progress_every and (i % progress_every == 0):
            print(f"[MAP+posterior] {i}/{N}", flush=True)

        # Get likelihood log_pz from MLE routine
        _, _, log_pz = template_fit_one_mle(x[i], sig[i], mu_grid, z_grid)

        # Add redshift prior to get log posterior (marginalised over templates)
        log_post_z = log_pz + log_prior_z
        log_post_z_all[i] = log_post_z

        zi_map = int(np.argmax(log_post_z))
        z_map[i] = float(z_grid[zi_map])
        t_map[i] = 0  # template selection secondary here

    return z_map, t_map, log_post_z_all


def main():
    rng = np.random.default_rng(0)
    N = 5000
    test_size = 0.30
    z_grid = np.linspace(0.0, 2.0, 401)

    template_library = "eazy_v1.3"
    sigma0_frac = 0.05
    kappa = 0.02

    outdir = "outputs/exp4_prior_misspecification"
    ensure_dir(outdir)

    # ------------------------------------------------------------------ #
    # Infrastructure
    # ------------------------------------------------------------------ #
    wl, R = build_filters_and_grid()
    templates, _ = load_templates(template_library, wl)
    mu_grid = grid_mu(wl, templates, R, z_grid)
    T = templates.shape[0]

    # ------------------------------------------------------------------ #
    # Priors
    # ------------------------------------------------------------------ #
    matched_probs = make_redshift_prior_probs(z_grid, z_max=2.0, alpha=2.0, z0=0.35)
    lowz_probs    = make_lowz_prior_probs(z_grid,    z_max=2.0, alpha=2.0, z0=0.15)

    log_prior_z_matched = np.log(matched_probs + 1e-300)
    log_prior_z_lowz    = np.log(lowz_probs    + 1e-300)
    log_prior_t         = np.log(np.ones(T, dtype=float) / T + 1e-300)

    # ------------------------------------------------------------------ #
    # Data generation — always uses matched prior
    # ------------------------------------------------------------------ #
    ds = make_dataset(
        rng=rng,
        mu_grid=mu_grid,
        z_grid=z_grid,
        N=N,
        z_max=2.0,
        z_probs=matched_probs,
        sigma0_frac=sigma0_frac,
        k=kappa,
        t_probs=np.ones(T, float) / T,
    )

    idx = np.arange(N)
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=0)

    X_train = make_features(ds.x[idx_train], ds.sigma[idx_train])
    y_train = ds.z[idx_train]
    X_test  = make_features(ds.x[idx_test],  ds.sigma[idx_test])
    y_test  = ds.z[idx_test]

    ds_test = SimpleNamespace(
        x=ds.x[idx_test],
        sigma=ds.sigma[idx_test],
        z=ds.z[idx_test],
    )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    print("\n--- MLE ---")
    z_mle, _, log_pz_mle = batch_template_fit_mle_with_posteriors(
        ds_test, mu_grid, z_grid)

    print("\n--- MAP matched prior ---")
    z_map_matched, _, log_post_matched = batch_template_fit_map_with_posteriors(
        ds_test, mu_grid, z_grid,
        log_prior_z=log_prior_z_matched,
        log_prior_t=log_prior_t,
    )

    print("\n--- MAP low-z prior ---")
    z_map_lowz, _, log_post_lowz = batch_template_fit_map_with_posteriors(
        ds_test, mu_grid, z_grid,
        log_prior_z=log_prior_z_lowz,
        log_prior_t=log_prior_t,
    )

    print("\n--- kNN ---")
    z_knn = fit_knn(X_train, y_train, X_test)

    print("\n--- Random Forest ---")
    z_rf = fit_random_forest(X_train, y_train, X_test)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    metrics = {
        "MLE":         compute_metrics(y_test, z_mle),
        "MAP_matched": compute_metrics(y_test, z_map_matched),
        "MAP_lowz":    compute_metrics(y_test, z_map_lowz),
        "kNN":         compute_metrics(y_test, z_knn),
        "RF":          compute_metrics(y_test, z_rf),
    }

    print("\n=== Performance metrics ===")
    for name, m in metrics.items():
        print(f"  {name:15s}  MAE={m['mae']:.3f}  bias={m['bias']:.4f}"
              f"  scatter={m['scatter']:.3f}  outlier={m['outlier']:.3f}")

    # ------------------------------------------------------------------ #
    # PIT values
    # ------------------------------------------------------------------ #
    pit_mle           = compute_pit_values(log_pz_mle,       z_grid, y_test)
    pit_map_matched   = compute_pit_values(log_post_matched,  z_grid, y_test)
    pit_map_lowz      = compute_pit_values(log_post_lowz,     z_grid, y_test)

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    summary = {
        "template_library": template_library,
        "N_total": int(N),
        "N_train": int(len(idx_train)),
        "N_test":  int(len(idx_test)),
        "sigma0_frac": sigma0_frac,
        "kappa": kappa,
        "matched_prior_z0": 0.35,
        "lowz_prior_z0":    0.15,
        "metrics": metrics,
    }
    with open(os.path.join(outdir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.save(os.path.join(outdir, "pit_mle.npy"),         pit_mle)
    np.save(os.path.join(outdir, "pit_map_matched.npy"), pit_map_matched)
    np.save(os.path.join(outdir, "pit_map_lowz.npy"),    pit_map_lowz)

    # ------------------------------------------------------------------ #
    # Figure 1 — Recovery plots (4 panels, same as your existing exp4)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.5))
    axes = axes.flatten()

    plot_recovery(axes[0], y_test, z_mle,
                  metrics["MLE"], "MLE",
                  color=METHOD_STYLE["MLE"]["color"])
    plot_recovery(axes[1], y_test, z_map_lowz,
                  metrics["MAP_lowz"], "MAP (low-z prior)",
                  color=METHOD_STYLE["MAP"]["color"])
    plot_recovery(axes[2], y_test, z_rf,
                  metrics["RF"], "Random Forest",
                  color=METHOD_STYLE["RF"]["color"])
    plot_recovery(axes[3], y_test, z_knn,
                  metrics["kNN"], "kNN",
                  color=METHOD_STYLE["kNN"]["color"])

    fig.suptitle("Prior misspecification: predicted vs true redshift", y=1.01)
    fig.tight_layout()
    savefig_both(fig, os.path.join(outdir, "recovery_prior_mismatch"))
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Figure 2 — Residual plots (4 panels)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.5))
    axes = axes.flatten()

    plot_residuals(axes[0], y_test, z_mle,
                   "MLE", color=METHOD_STYLE["MLE"]["color"])
    plot_residuals(axes[1], y_test, z_map_lowz,
                   "MAP (low-z prior)", color=METHOD_STYLE["MAP"]["color"])
    plot_residuals(axes[2], y_test, z_rf,
                   "Random Forest", color=METHOD_STYLE["RF"]["color"])
    plot_residuals(axes[3], y_test, z_knn,
                   "kNN", color=METHOD_STYLE["kNN"]["color"])

    fig.suptitle("Prior misspecification: residual errors", y=1.01)
    fig.tight_layout()
    savefig_both(fig, os.path.join(outdir, "residuals_prior_mismatch"))
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Figure 3 — PIT histograms (3 panels)
    # Three conditions: MLE likelihood, MAP matched, MAP low-z
    # This is the key diagnostic figure for calibration
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    plot_pit_histogram(
        axes[0], pit_mle,
        title="MLE (likelihood as posterior)",
        color=METHOD_STYLE["MLE"]["color"],
    )
    plot_pit_histogram(
        axes[1], pit_map_matched,
        title="MAP — matched prior",
        color=METHOD_STYLE["MAP"]["color"],
    )
    plot_pit_histogram(
        axes[2], pit_map_lowz,
        title="MAP — low-z prior (misspecified)",
        color="#e07b39",
    )

    fig.suptitle("Posterior calibration: probability integral transform (PIT)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    savefig_both(fig, os.path.join(outdir, "pit_histograms"))
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Figure 4 — Prior comparison: matched vs misspecified
    # Shows the two priors on the same axes 
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(z_grid, matched_probs, color=METHOD_STYLE["MAP"]["color"],
            lw=1.8, label=r"Matched prior ($z_0=0.35$)")
    ax.plot(z_grid, lowz_probs,    color="#e07b39",
            lw=1.8, ls="--", label=r"Low-z prior ($z_0=0.15$)")
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel("Prior probability $\\pi(z)$")
    ax.set_title("Matched vs misspecified redshift prior", pad=8)
    ax.legend(frameon=False)
    style_axes_box(ax)
    fig.tight_layout()
    savefig_both(fig, os.path.join(outdir, "prior_comparison"))
    plt.close(fig)

    print("\nAll outputs saved to:", outdir)


if __name__ == "__main__":
    main()
