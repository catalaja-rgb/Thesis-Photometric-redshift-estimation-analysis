"""
Experiments 1 and 2 for the photometric redshift simulation study.

Experiment 1:
    Baseline comparison under matched conditions.

Experiment 2:
    Performance as observational noise increases, still under matched prior
    conditions.

The script compares four estimators:
    - template-fitting MLE
    - template-fitting MAP
    - k-nearest neighbours
    - random forest

Outputs are written to the outputs/ directory as figures and JSON summaries.
"""

import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from sklearn.model_selection import train_test_split

from photoz_sim.datasets import make_dataset
from photoz_sim.filters_eazy import (
    auto_wavelength_grid_from_filters,
    build_R_matrix_on_grid,
    find_eazy_filters_res,
    load_eazy_filters_res,
    select_filters_in_range,
)
from photoz_sim.forward import grid_mu
from photoz_sim.methods.template_fit_grid_map import batch_template_fit_map
from photoz_sim.methods.template_fit_grid_mle import batch_template_fit_mle
from photoz_sim.templates_bpz import load_bpz_templates_from_list
from photoz_sim.templates_eazy import load_eazy_templates_from_spectra_param


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "axes.linewidth": 0.9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

METHOD_STYLE = {
    "MLE": {"color": "black", "marker": "o", "linestyle": "-"},
    "MAP": {"color": "#4C78A8", "marker": "s", "linestyle": "-"},
    "kNN": {"color": "#4C8C4A", "marker": "^", "linestyle": "--"},
    "RF": {"color": "#B05A5A", "marker": "D", "linestyle": "--"},
}


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def savefig_both(fig, path_no_ext: str) -> None:
    """Save a figure to both PNG and PDF."""
    fig.savefig(path_no_ext + ".png")
    fig.savefig(path_no_ext + ".pdf")


def save_json(data: dict, path: str) -> None:
    """Write a dictionary to JSON with readable formatting."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def compute_metrics(z_true: np.ndarray, z_pred: np.ndarray) -> dict:
    """
    Compute the evaluation metrics used throughout the thesis.

    The outlier definition follows the usual normalised residual threshold
    |dz| > 0.15.
    """
    z_true = np.asarray(z_true, dtype=float)
    z_pred = np.asarray(z_pred, dtype=float)

    dz = (z_pred - z_true) / (1.0 + z_true)

    return {
        "mae": float(np.mean(np.abs(z_pred - z_true))),
        "bias": float(np.mean(dz)),
        "scatter": float(np.std(dz)),
        "outlier": float(np.mean(np.abs(dz) > 0.15)),
    }


def make_features(fluxes: np.ndarray, flux_errors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Build features for the ML baselines.

    Here the fluxes are scaled by their reported uncertainty so that the model
    sees something closer to a signal-to-noise representation rather than raw
    flux alone.
    """
    fluxes = np.asarray(fluxes, dtype=float)
    flux_errors = np.asarray(flux_errors, dtype=float)
    return fluxes / np.clip(flux_errors, eps, None)


def make_redshift_prior_probs(
    z_grid: np.ndarray,
    z_max: float = 2.0,
    alpha: float = 2.0,
    z0: float = 0.35,
) -> np.ndarray:
    """
    Construct a discrete redshift prior on the same grid used for inference.

    The prior is truncated at z_max and then normalised to sum to one.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    probs = (z_grid ** alpha) * np.exp(-z_grid / z0)
    probs = np.where(z_grid <= z_max, probs, 0.0)
    probs[0] = 0.0

    total_mass = float(np.sum(probs))
    if total_mass <= 0:
        raise ValueError("Redshift prior has zero mass.")

    return probs / total_mass


# ---------------------------------------------------------------------
# ML baselines
# ---------------------------------------------------------------------

def fit_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_neighbors: int = 25,
    weights: str = "distance",
) -> np.ndarray:
    """Fit a kNN regressor and return predictions on the test set."""
    from sklearn.neighbors import KNeighborsRegressor

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 300,
    random_state: int = 0,
    max_depth=None,
    min_samples_leaf: int = 2,
) -> np.ndarray:
    """
    Fit a random forest regressor and return predictions on the test set.

    A small minimum leaf size is used to avoid overly jagged fits on the
    simulated training sample.
    """
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


# ---------------------------------------------------------------------
# Template / filter setup
# ---------------------------------------------------------------------

def load_templates(template_label: str, wavelength_grid_nm: np.ndarray):
    """Load one of the template libraries used in the thesis."""
    eazy_root = "data/external/eazy-photoz"
    bpz_sed_dir = "data/external/bpz/bpz-1.99.3/SED"

    if template_label == "eazy_v1.3":
        templates, names = load_eazy_templates_from_spectra_param(
            wavelengths_nm=wavelength_grid_nm,
            spectra_param_file=f"{eazy_root}/templates/eazy_v1.3.spectra.param",
            base_dir=eazy_root,
            max_templates=None,
            normalize="integral",
        )
    elif template_label == "cww+kin":
        templates, names = load_eazy_templates_from_spectra_param(
            wavelengths_nm=wavelength_grid_nm,
            spectra_param_file=f"{eazy_root}/templates/cww+kin.spectra.param",
            base_dir=eazy_root,
            max_templates=None,
            normalize="integral",
        )
    elif template_label == "pegase13":
        templates, names = load_eazy_templates_from_spectra_param(
            wavelengths_nm=wavelength_grid_nm,
            spectra_param_file=f"{eazy_root}/templates/pegase13.spectra.param",
            base_dir=eazy_root,
            max_templates=None,
            normalize="integral",
        )
    elif template_label == "bpz_cwwsb4":
        templates, names = load_bpz_templates_from_list(
            wavelengths_nm=wavelength_grid_nm,
            bpz_sed_dir=bpz_sed_dir,
            list_file="CWWSB4.list",
            wl_unit="A",
        )
    else:
        raise ValueError(f"Unknown template label: {template_label}")

    return templates, names


def build_filters_and_grid(
    n_filters: int = 5,
    wavelength_min_nm: float = 300.0,
    wavelength_max_nm: float = 1100.0,
    step_nm: float = 0.5,
):
    """
    Select a small filter set within the wavelength range of interest and
    build the wavelength grid used by the forward model.
    """
    eazy_root = "data/external/eazy-photoz"
    filters_res = find_eazy_filters_res(eazy_root)
    all_filters = load_eazy_filters_res(str(filters_res))

    chosen_filters = select_filters_in_range(
        all_filters,
        wlmin_nm=wavelength_min_nm,
        wlmax_nm=wavelength_max_nm,
        n=n_filters,
    )

    wavelength_grid_nm = auto_wavelength_grid_from_filters(
        chosen_filters,
        step_nm=step_nm,
        wl_min_nm_floor=wavelength_min_nm,
        wl_max_nm_ceil=wavelength_max_nm,
    )

    filter_matrix = build_R_matrix_on_grid(
        wavelength_grid_nm,
        chosen_filters,
        normalize=True,
    )

    return wavelength_grid_nm, filter_matrix


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def style_axes_box(ax) -> None:
    """Apply a consistent boxed axis style."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor("0.15")
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def add_metrics_box(ax, metrics: dict) -> None:
    """Add a compact summary of the main metrics to a scatter panel."""
    text = (
        rf"$\mathrm{{MAE}}={metrics['mae']:.3f}$" "\n"
        rf"$\mathrm{{Bias}}={metrics['bias']:.3f}$" "\n"
        rf"$\mathrm{{Scatter}}={metrics['scatter']:.3f}$" "\n"
        rf"$\mathrm{{Outliers}}={metrics['outlier']:.3f}$"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="0.65",
            linewidth=0.8,
            alpha=0.96,
        ),
    )


def scatter_panel(ax, z_true, z_pred, title, z_min, z_max, metrics) -> None:
    """Draw one true-vs-predicted scatter panel with the 1:1 line."""
    ax.scatter(
        z_true,
        z_pred,
        s=9,
        alpha=0.24,
        color="0.25",
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([z_min, z_max], [z_min, z_max], color="black", lw=1.15, zorder=3)

    ax.set_title(title, pad=8)
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(z_min, z_max)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(False)

    style_axes_box(ax)
    add_metrics_box(ax, metrics)


# ---------------------------------------------------------------------
# Shared experiment logic
# ---------------------------------------------------------------------

def build_train_test_split(dataset, test_size: float):
    """Create a fixed train/test split and the associated ML features."""
    indices = np.arange(len(dataset.z))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=0,
    )

    X_train = make_features(dataset.x[idx_train], dataset.sigma[idx_train])
    y_train = dataset.z[idx_train]

    X_test = make_features(dataset.x[idx_test], dataset.sigma[idx_test])
    y_test = dataset.z[idx_test]

    dataset_test = SimpleNamespace(
        x=dataset.x[idx_test],
        sigma=dataset.sigma[idx_test],
        z=dataset.z[idx_test],
    )

    return idx_train, idx_test, X_train, y_train, X_test, y_test, dataset_test


def evaluate_all_methods(
    dataset_test,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mu_grid: np.ndarray,
    z_grid: np.ndarray,
    log_prior_z: np.ndarray,
    log_prior_t: np.ndarray,
):
    """
    Run all four estimators on a common split and return predictions and metrics.
    """
    predictions = {}
    metrics = {}

    z_mle, _ = batch_template_fit_mle(
        dataset_test,
        mu_grid,
        z_grid,
    )
    predictions["MLE"] = z_mle
    metrics["MLE"] = compute_metrics(y_test, z_mle)

    z_map, _ = batch_template_fit_map(
        dataset_test,
        mu_grid,
        z_grid,
        log_prior_z=log_prior_z,
        log_prior_t=log_prior_t,
    )
    predictions["MAP"] = z_map
    metrics["MAP"] = compute_metrics(y_test, z_map)

    z_knn = fit_knn(X_train, y_train, X_test, n_neighbors=25, weights="distance")
    predictions["kNN"] = z_knn
    metrics["kNN"] = compute_metrics(y_test, z_knn)

    z_rf = fit_random_forest(
        X_train,
        y_train,
        X_test,
        n_estimators=300,
        random_state=0,
        max_depth=None,
        min_samples_leaf=2,
    )
    predictions["RF"] = z_rf
    metrics["RF"] = compute_metrics(y_test, z_rf)

    return predictions, metrics


def make_experiment_config(
    experiment_name: str,
    template_label: str,
    z_grid: np.ndarray,
    N: int,
    test_size: float,
    extra: dict,
) -> dict:
    """Create a config dictionary to save with the experiment outputs."""
    config = {
        "experiment": experiment_name,
        "seed": 0,
        "template_library": template_label,
        "z_grid_min": float(np.min(z_grid)),
        "z_grid_max": float(np.max(z_grid)),
        "z_grid_size": int(len(z_grid)),
        "N_total": int(N),
        "test_size": float(test_size),
    }
    config.update(extra)
    return config


# ---------------------------------------------------------------------
# Experiment 1: baseline
# ---------------------------------------------------------------------

def run_baseline_experiment_with_prior(
    rng,
    wavelength_grid_nm,
    templates,
    filter_matrix,
    z_grid,
    template_label: str,
    N: int = 5000,
    test_size: float = 0.30,
):
    """
    Baseline comparison under matched conditions.

    The data-generating prior and the inference prior are the same here, so
    this acts as the clean reference case before introducing stronger noise.
    """
    outdir = f"outputs/exp1_baseline_with_prior_{template_label}"
    ensure_dir(outdir)

    mu_grid = grid_mu(wavelength_grid_nm, templates, filter_matrix, z_grid)

    z_probs = make_redshift_prior_probs(z_grid, z_max=2.0, alpha=2.0, z0=0.35)
    log_prior_z = np.log(z_probs + 1e-300)

    n_templates = templates.shape[0]
    template_probs = np.ones(n_templates, dtype=float) / n_templates
    log_prior_t = np.log(template_probs + 1e-300)

    sigma0_frac = 0.005
    noise_slope = 0.002

    dataset = make_dataset(
        rng=rng,
        mu_grid=mu_grid,
        z_grid=z_grid,
        N=N,
        z_max=2.0,
        z_probs=z_probs,
        sigma0_frac=sigma0_frac,
        k=noise_slope,
        t_probs=template_probs,
    )

    idx_train, idx_test, X_train, y_train, X_test, y_test, dataset_test = build_train_test_split(
        dataset, test_size=test_size
    )

    predictions, metrics = evaluate_all_methods(
        dataset_test=dataset_test,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        mu_grid=mu_grid,
        z_grid=z_grid,
        log_prior_z=log_prior_z,
        log_prior_t=log_prior_t,
    )

    summary = {
        "template_library": template_label,
        "N_total": int(N),
        "N_train": int(len(idx_train)),
        "N_test": int(len(idx_test)),
        "sigma0_frac": sigma0_frac,
        "noise_slope": noise_slope,
        "methods": metrics,
    }
    save_json(summary, os.path.join(outdir, "metrics_summary.json"))

    config = make_experiment_config(
        experiment_name="exp1_baseline_with_prior",
        template_label=template_label,
        z_grid=z_grid,
        N=N,
        test_size=test_size,
        extra={
            "sigma0_frac": sigma0_frac,
            "noise_slope": noise_slope,
            "prior_alpha": 2.0,
            "prior_z0": 0.35,
            "n_templates": int(n_templates),
        },
    )
    save_json(config, os.path.join(outdir, "config.json"))

    z_min = float(z_grid.min())
    z_max = float(z_grid.max())

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.7), sharex=True, sharey=True)
    axes = axes.ravel()

    scatter_panel(axes[0], y_test, predictions["MLE"], "MLE", z_min, z_max, metrics["MLE"])
    scatter_panel(axes[1], y_test, predictions["MAP"], "MAP", z_min, z_max, metrics["MAP"])
    scatter_panel(axes[2], y_test, predictions["kNN"], "kNN", z_min, z_max, metrics["kNN"])
    scatter_panel(axes[3], y_test, predictions["RF"], "Random Forest", z_min, z_max, metrics["RF"])

    axes[2].set_xlabel(r"True redshift $z_{\mathrm{true}}$")
    axes[3].set_xlabel(r"True redshift $z_{\mathrm{true}}$")
    axes[0].set_ylabel(r"Predicted redshift $\hat{z}$")
    axes[2].set_ylabel(r"Predicted redshift $\hat{z}$")

    fig.subplots_adjust(top=0.92, wspace=0.14, hspace=0.14)
    savefig_both(fig, os.path.join(outdir, "baseline_2x2_comparison"))
    plt.close(fig)


# ---------------------------------------------------------------------
# Experiment 2: noise
# ---------------------------------------------------------------------

def run_noise_experiment_with_prior(
    rng,
    wavelength_grid_nm,
    templates,
    filter_matrix,
    z_grid,
    template_label: str,
    N: int = 5000,
    test_size: float = 0.30,
):
    """
    Performance as noise increases, keeping the redshift prior matched.

    This isolates the effect of observational degradation without mixing it
    with explicit prior misspecification.
    """
    outdir = f"outputs/exp2_noise_with_prior_{template_label}"
    ensure_dir(outdir)

    mu_grid = grid_mu(wavelength_grid_nm, templates, filter_matrix, z_grid)

    z_probs = make_redshift_prior_probs(z_grid, z_max=2.0, alpha=2.0, z0=0.35)
    log_prior_z = np.log(z_probs + 1e-300)

    n_templates = templates.shape[0]
    template_probs = np.ones(n_templates, dtype=float) / n_templates
    log_prior_t = np.log(template_probs + 1e-300)

    noise_grid = [
        {"sigma0_frac": 0.01, "noise_slope": 0.005},
        {"sigma0_frac": 0.03, "noise_slope": 0.010},
        {"sigma0_frac": 0.05, "noise_slope": 0.020},
        {"sigma0_frac": 0.08, "noise_slope": 0.040},
        {"sigma0_frac": 0.12, "noise_slope": 0.060},
    ]

    results = {}

    for i, noise_cfg in enumerate(noise_grid, start=1):
        sigma0_frac = noise_cfg["sigma0_frac"]
        noise_slope = noise_cfg["noise_slope"]

        dataset = make_dataset(
            rng=rng,
            mu_grid=mu_grid,
            z_grid=z_grid,
            N=N,
            z_max=2.0,
            z_probs=z_probs,
            sigma0_frac=sigma0_frac,
            k=noise_slope,
            t_probs=template_probs,
        )

        idx_train, idx_test, X_train, y_train, X_test, y_test, dataset_test = build_train_test_split(
            dataset, test_size=test_size
        )

        _, metrics = evaluate_all_methods(
            dataset_test=dataset_test,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            mu_grid=mu_grid,
            z_grid=z_grid,
            log_prior_z=log_prior_z,
            log_prior_t=log_prior_t,
        )

        results[f"noise_{i}"] = {
            "sigma0_frac": sigma0_frac,
            "noise_slope": noise_slope,
            "MLE": metrics["MLE"],
            "MAP": metrics["MAP"],
            "kNN": metrics["kNN"],
            "RF": metrics["RF"],
        }

    save_json(results, os.path.join(outdir, "noise_metrics.json"))

    config = make_experiment_config(
        experiment_name="exp2_noise_with_prior",
        template_label=template_label,
        z_grid=z_grid,
        N=N,
        test_size=test_size,
        extra={
            "prior_alpha": 2.0,
            "prior_z0": 0.35,
            "n_templates": int(n_templates),
            "noise_grid": noise_grid,
        },
    )
    save_json(config, os.path.join(outdir, "config.json"))

    sorted_keys = sorted(results.keys(), key=lambda key: results[key]["noise_slope"])
    x_values = [results[key]["noise_slope"] for key in sorted_keys]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharex=True)

    metric_specs = [
        ("mae", "Mean absolute error", "MAE as a function of noise"),
        ("outlier", "Catastrophic outlier rate", "Outlier rate as a function of noise"),
    ]

    for ax, (metric_name, ylabel, title) in zip(axes, metric_specs):
        for method in ["MLE", "MAP", "kNN", "RF"]:
            y_values = [results[key][method][metric_name] for key in sorted_keys]
            style = METHOD_STYLE[method]

            ax.plot(
                x_values,
                y_values,
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                lw=1.7,
                ms=5.2,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=0.7,
            )

        ax.set_xlabel(r"Noise level $k$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(False, axis="x")
        style_axes_box(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=2.0,
        columnspacing=1.5,
    )
    fig.subplots_adjust(top=0.88, wspace=0.27)
    savefig_both(fig, os.path.join(outdir, "noise_combined_mae_outlier"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    for method in ["MLE", "MAP", "kNN", "RF"]:
        y_values = [results[key][method]["scatter"] for key in sorted_keys]
        style = METHOD_STYLE[method]

        ax.plot(
            x_values,
            y_values,
            label=method,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            lw=1.7,
            ms=5.2,
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=0.7,
        )

    ax.set_xlabel(r"Noise level $k$")
    ax.set_ylabel("Scatter")
    ax.set_title("Scatter as a function of noise", pad=8)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(False, axis="x")
    ax.legend(frameon=False, loc="best")
    style_axes_box(ax)

    savefig_both(fig, os.path.join(outdir, "scatter_vs_noise"))
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)

    N = 5000
    test_size = 0.30
    z_grid = np.linspace(0.0, 2.0, 401)
    template_label = "bpz_cwwsb4"

    wavelength_grid_nm, filter_matrix = build_filters_and_grid()
    templates, template_names = load_templates(template_label, wavelength_grid_nm)

    run_baseline_experiment_with_prior(
        rng=rng,
        wavelength_grid_nm=wavelength_grid_nm,
        templates=templates,
        filter_matrix=filter_matrix,
        z_grid=z_grid,
        template_label=template_label,
        N=N,
        test_size=test_size,
    )

    run_noise_experiment_with_prior(
        rng=rng,
        wavelength_grid_nm=wavelength_grid_nm,
        templates=templates,
        filter_matrix=filter_matrix,
        z_grid=z_grid,
        template_label=template_label,
        N=N,
        test_size=test_size,
    )


if __name__ == "__main__":
    main()