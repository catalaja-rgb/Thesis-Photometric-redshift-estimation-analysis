import os
import json
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split

from photoz_sim.datasets import make_dataset
from photoz_sim.forward import grid_mu
from photoz_sim.methods.template_fit_grid_mle import batch_template_fit_mle
from photoz_sim.methods.template_fit_grid_map import batch_template_fit_map
from photoz_sim.templates_eazy import load_eazy_templates_from_spectra_param
from photoz_sim.templates_bpz import load_bpz_templates_from_list
from photoz_sim.filters_eazy import (
    find_eazy_filters_res,
    load_eazy_filters_res,
    select_filters_in_range,
    auto_wavelength_grid_from_filters,
    build_R_matrix_on_grid,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11,
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
})

PANEL_STYLE = {
    "MLE": {"title": "MLE", "color": "0.15"},
    "MAP_lowz": {"title": "MAP (low-z prior)", "color": "0.15"},
    "RF": {"title": "Random Forest", "color": "0.15"},
    "kNN": {"title": "kNN", "color": "0.15"},
}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def savefig_both(fig, path_no_ext: str) -> None:
    fig.savefig(path_no_ext + ".png")
    fig.savefig(path_no_ext + ".pdf")

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
    x = np.asarray(x, float)
    x_err = np.asarray(x_err, float)
    return x / np.clip(x_err, eps, None)

def fit_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 25,
    weights: str = "distance",
) -> np.ndarray:
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=k, weights=weights)
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

def make_redshift_prior_probs(z_grid, z_max=2.0, alpha=2.0, z0=0.35):
    z = np.asarray(z_grid, float)
    p = (z ** alpha) * np.exp(-z / z0)
    p = np.where(z <= z_max, p, 0.0)
    p[0] = 0.0
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Prior has zero mass.")
    return p / s

def make_lowz_misspecified_prior(z_grid, z_max=2.0, alpha=1.0, z0=0.18):
    z = np.asarray(z_grid, float)
    p = (z ** alpha) * np.exp(-z / z0)
    p = np.where(z <= z_max, p, 0.0)
    p[0] = 0.0
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Low-z prior has zero mass.")
    return p / s

def load_templates(label: str, wl: np.ndarray):
    eazy_root = "data/external/eazy-photoz"
    bpz_sed_dir = "data/external/bpz/bpz-1.99.3/SED"

    if label == "eazy_v1.3":
        templates, names = load_eazy_templates_from_spectra_param(
            wavelengths_nm=wl,
            spectra_param_file=f"{eazy_root}/templates/eazy_v1.3.spectra.param",
            base_dir=eazy_root,
            max_templates=None,
            normalize="integral",
        )
    elif label == "cww+kin":
        templates, names = load_eazy_templates_from_spectra_param(
            wavelengths_nm=wl,
            spectra_param_file=f"{eazy_root}/templates/cww+kin.spectra.param",
            base_dir=eazy_root,
            max_templates=None,
            normalize="integral",
        )
    elif label == "pegase13":
        templates, names = load_eazy_templates_from_spectra_param(
            wavelengths_nm=wl,
            spectra_param_file=f"{eazy_root}/templates/pegase13.spectra.param",
            base_dir=eazy_root,
            max_templates=None,
            normalize="integral",
        )
    elif label == "bpz_cwwsb4":
        templates, names = load_bpz_templates_from_list(
            wavelengths_nm=wl,
            bpz_sed_dir=bpz_sed_dir,
            list_file="CWWSB4.list",
            wl_unit="A",
        )
    else:
        raise ValueError(f"Unknown template label: {label}")

    return templates, names

def build_filters_and_grid(
    B: int = 5,
    wlmin_nm: float = 300.0,
    wlmax_nm: float = 1100.0,
    step_nm: float = 0.5,
):
    eazy_root = "data/external/eazy-photoz"
    filters_res = find_eazy_filters_res(eazy_root)
    all_filters = load_eazy_filters_res(str(filters_res))

    chosen_filters = select_filters_in_range(
        all_filters,
        wlmin_nm=wlmin_nm,
        wlmax_nm=wlmax_nm,
        n=B,
    )

    print("Using FILTERS.RES:", str(filters_res))
    print(f"Chosen {B} filters (eff nm):")
    for f in chosen_filters:
        print(f"  {f.name}  eff={f.eff_wavelength_A() / 10.0:.1f} nm")

    wl = auto_wavelength_grid_from_filters(
        chosen_filters,
        step_nm=step_nm,
        wl_min_nm_floor=wlmin_nm,
        wl_max_nm_ceil=wlmax_nm,
    )
    R = build_R_matrix_on_grid(wl, chosen_filters, normalize=True)
    return wl, R

def style_axes_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor("0.15")
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

def add_metrics_box(ax, metrics: dict):
    txt = (
        rf"$\mathrm{{MAE}}={metrics['mae']:.3f}$" "\n"
        rf"$\mathrm{{Bias}}={metrics['bias']:.3f}$" "\n"
        rf"$\mathrm{{Scatter}}={metrics['scatter']:.3f}$" "\n"
        rf"$\mathrm{{Outliers}}={metrics['outlier']:.3f}$"
    )
    ax.text(
        0.04, 0.96, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="white",
            edgecolor="0.72",
            linewidth=0.75,
            alpha=0.96,
        ),
    )

def recovery_panel(ax, z_true, z_pred, metrics, panel_key, zmin, zmax):
    style = PANEL_STYLE[panel_key]
    ax.scatter(
        z_true,
        z_pred,
        s=7,
        alpha=0.20,
        color=style["color"],
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([zmin, zmax], [zmin, zmax], color="black", lw=1.05, zorder=3)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(style["title"], pad=7, weight="medium")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(False)
    style_axes_box(ax)
    add_metrics_box(ax, metrics)

def residual_panel(ax, z_true, z_pred, metrics, panel_key, zmin, zmax, ylims):
    style = PANEL_STYLE[panel_key]
    dz = (z_pred - z_true) / (1.0 + z_true)
    outmask = np.abs(dz) > 0.15

    ax.scatter(
        z_true[~outmask],
        dz[~outmask],
        s=7,
        alpha=0.18,
        color=style["color"],
        edgecolors="none",
        rasterized=True,
    )

    if np.any(outmask):
        ax.scatter(
            z_true[outmask],
            dz[outmask],
            s=11,
            alpha=0.65,
            facecolors="white",
            edgecolors="black",
            linewidths=0.45,
            marker="o",
            rasterized=True,
        )

    ax.axhline(0.0, color="black", lw=1.0, zorder=3)
    ax.axhline(0.15, color="0.45", lw=0.85, linestyle="--", zorder=2)
    ax.axhline(-0.15, color="0.45", lw=0.85, linestyle="--", zorder=2)

    ax.set_xlim(zmin, zmax)
    ax.set_ylim(*ylims)
    ax.set_title(style["title"], pad=7, weight="medium")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.grid(False, axis="x")
    style_axes_box(ax)
    add_metrics_box(ax, metrics)

def compute_common_residual_limits(results_dict):
    all_dz = []
    for key in results_dict:
        z_true = results_dict[key]["z_true"]
        z_pred = results_dict[key]["z_pred"]
        dz = (z_pred - z_true) / (1.0 + z_true)
        all_dz.append(dz)
    all_dz = np.concatenate(all_dz)
    q01, q99 = np.quantile(all_dz, [0.01, 0.99])
    ymin = min(-0.25, q01 - 0.03)
    ymax = max(0.25, q99 + 0.03)
    return (ymin, ymax)

def plot_recovery_figure(results_dict, outdir, z_grid):
    zmin = float(z_grid.min())
    zmax = float(z_grid.max())

    fig, axes = plt.subplots(2, 2, figsize=(9.1, 8.4), sharex=True, sharey=True)
    axes = axes.ravel()

    panel_order = ["MLE", "MAP_lowz", "RF", "kNN"]

    for ax, panel_key in zip(axes, panel_order):
        recovery_panel(
            ax,
            results_dict[panel_key]["z_true"],
            results_dict[panel_key]["z_pred"],
            results_dict[panel_key]["metrics"],
            panel_key,
            zmin,
            zmax,
        )

    axes[2].set_xlabel(r"True redshift $z_{\mathrm{true}}$")
    axes[3].set_xlabel(r"True redshift $z_{\mathrm{true}}$")
    axes[0].set_ylabel(r"Predicted redshift $\hat{z}$")
    axes[2].set_ylabel(r"Predicted redshift $\hat{z}$")

    fig.subplots_adjust(wspace=0.12, hspace=0.14)
    savefig_both(fig, os.path.join(outdir, "recovery_comparison_2x2"))
    plt.close(fig)

def plot_residual_figure(results_dict, outdir, z_grid):
    zmin = float(z_grid.min())
    zmax = float(z_grid.max())
    ylims = compute_common_residual_limits(results_dict)

    fig, axes = plt.subplots(2, 2, figsize=(9.1, 8.4), sharex=True, sharey=True)
    axes = axes.ravel()

    panel_order = ["MLE", "MAP_lowz", "kNN", "RF"]

    for ax, panel_key in zip(axes, panel_order):
        residual_panel(
            ax,
            results_dict[panel_key]["z_true"],
            results_dict[panel_key]["z_pred"],
            results_dict[panel_key]["metrics"],
            panel_key,
            zmin,
            zmax,
            ylims,
        )

    axes[2].set_xlabel(r"True redshift $z_{\mathrm{true}}$")
    axes[3].set_xlabel(r"True redshift $z_{\mathrm{true}}$")
    axes[0].set_ylabel(r"$\Delta z = (\hat{z}-z_{\mathrm{true}})/(1+z_{\mathrm{true}})$")
    axes[2].set_ylabel(r"$\Delta z = (\hat{z}-z_{\mathrm{true}})/(1+z_{\mathrm{true}})$")

    fig.subplots_adjust(wspace=0.12, hspace=0.14)
    savefig_both(fig, os.path.join(outdir, "residual_comparison_2x2"))
    plt.close(fig)

def main():
    rng = np.random.default_rng(0)
    N = 5000
    test_size = 0.30
    z_grid = np.linspace(0.0, 2.0, 401)

    template_label = "bpz_cwwsb4"
    sigma0_frac = 0.05
    kappa = 0.02

    outdir = f"outputs/exp4_lowz_prior_comparison_{template_label}"
    ensure_dir(outdir)

    wl, R = build_filters_and_grid()

    templates, _ = load_templates(template_label, wl)
    mu_grid = grid_mu(wl, templates, R, z_grid)

    z_probs_true = make_redshift_prior_probs(z_grid, z_max=2.0, alpha=2.0, z0=0.35)
    z_probs_lowz = make_lowz_misspecified_prior(z_grid, z_max=2.0, alpha=1.0, z0=0.18)

    log_prior_z_lowz = np.log(z_probs_lowz + 1e-300)

    T = templates.shape[0]
    t_probs = np.ones(T, dtype=float) / T
    log_prior_t = np.log(t_probs + 1e-300)

    ds = make_dataset(
        rng=rng,
        mu_grid=mu_grid,
        z_grid=z_grid,
        N=N,
        z_max=2.0,
        z_probs=z_probs_true,
        sigma0_frac=sigma0_frac,
        k=kappa,
        t_probs=t_probs,
    )

    print("\nGenerated fixed dataset for low-z prior comparison:")
    print("  template library =", template_label)
    print("  x shape          =", ds.x.shape)
    print("  z min/max        =", float(ds.z.min()), float(ds.z.max()))

    idx = np.arange(N)
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=0,
    )

    X_train = make_features(ds.x[idx_train], ds.sigma[idx_train])
    y_train = ds.z[idx_train]
    X_test = make_features(ds.x[idx_test], ds.sigma[idx_test])
    y_test = ds.z[idx_test]

    ds_test = SimpleNamespace(
        x=ds.x[idx_test],
        sigma=ds.sigma[idx_test],
        z=ds.z[idx_test],
    )

    z_mle, _ = batch_template_fit_mle(
        ds_test,
        mu_grid,
        z_grid,
        progress_every=200,
    )
    metrics_mle = compute_metrics(y_test, z_mle)

    z_map_lowz, _ = batch_template_fit_map(
        ds_test,
        mu_grid,
        z_grid,
        log_prior_z=log_prior_z_lowz,
        log_prior_t=log_prior_t,
        progress_every=200,
    )
    metrics_map_lowz = compute_metrics(y_test, z_map_lowz)

    z_knn = fit_knn(X_train, y_train, X_test, k=25, weights="distance")
    metrics_knn = compute_metrics(y_test, z_knn)

    z_rf = fit_random_forest(
        X_train, y_train, X_test,
        n_estimators=300,
        random_state=0,
        max_depth=None,
        min_samples_leaf=2,
    )
    metrics_rf = compute_metrics(y_test, z_rf)

    summary = {
        "template_library": template_label,
        "N_total": int(N),
        "N_train": int(len(idx_train)),
        "N_test": int(len(idx_test)),
        "sigma0_frac": sigma0_frac,
        "kappa": kappa,
        "generation_prior": "matched",
        "map_inference_prior": "low_z_concentrated",
        "results": {
            "MLE": metrics_mle,
            "MAP_lowz": metrics_map_lowz,
            "kNN": metrics_knn,
            "RF": metrics_rf,
        },
    }

    with open(os.path.join(outdir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    results_dict = {
        "MLE": {
            "z_true": y_test,
            "z_pred": z_mle,
            "metrics": metrics_mle,
        },
        "MAP_lowz": {
            "z_true": y_test,
            "z_pred": z_map_lowz,
            "metrics": metrics_map_lowz,
        },
        "RF": {
            "z_true": y_test,
            "z_pred": z_rf,
            "metrics": metrics_rf,
        },
        "kNN": {
            "z_true": y_test,
            "z_pred": z_knn,
            "metrics": metrics_knn,
        },
    }

    plot_recovery_figure(results_dict, outdir, z_grid)
    plot_residual_figure(results_dict, outdir, z_grid)

    print("\nSaved metrics to:")
    print(" ", os.path.join(outdir, "metrics_summary.json"))
    print("Saved figures:")
    print(" ", os.path.join(outdir, "recovery_comparison_2x2.png"))
    print(" ", os.path.join(outdir, "recovery_comparison_2x2.pdf"))
    print(" ", os.path.join(outdir, "residual_comparison_2x2.png"))
    print(" ", os.path.join(outdir, "residual_comparison_2x2.pdf"))

if __name__ == "__main__":
    main()