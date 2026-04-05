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
from photoz_sim.templates_bpz import load_bpz_templates_from_list
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


def make_redshift_prior_probs(z_grid, z_max=2.0, alpha=2.0, z0=0.35):
    z = np.asarray(z_grid, float)
    p = (z ** alpha) * np.exp(-z / z0)
    p = np.where(z <= z_max, p, 0.0)
    p[0] = 0.0
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Prior has zero mass.")
    return p / s


def plot_metric_panel(ax, x, libs_short, results, metric_key, ylabel, title):
    for method in ["MLE", "MAP", "kNN", "RF"]:
        y = [results[lib][method][metric_key] for lib in inference_libraries]
        style = METHOD_STYLE[method]
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            lw=style["lw"],
            ms=style["ms"],
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
            label=method,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(libs_short)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    style_axes_box(ax)


def main():
    rng = np.random.default_rng(0)
    N = 5000
    test_size = 0.30
    z_grid = np.linspace(0.0, 2.0, 401)

    generation_library = "eazy_v1.3"
    global inference_libraries
    inference_libraries = [
        "eazy_v1.3",
        "bpz_cwwsb4",
        "cww+kin",
        "pegase13",
    ]

    library_labels = {
        "eazy_v1.3": "EAZY",
        "bpz_cwwsb4": "BPZ",
        "cww+kin": "CWW+KIN",
        "pegase13": "PEGASE",
    }

    sigma0_frac = 0.05
    kappa = 0.02

    outdir = f"outputs/exp3_template_mismatch_gen_{generation_library}"
    ensure_dir(outdir)

    wl, R = build_filters_and_grid()

    gen_templates, _ = load_templates(generation_library, wl)
    mu_grid_gen = grid_mu(wl, gen_templates, R, z_grid)

    z_probs = make_redshift_prior_probs(z_grid, z_max=2.0, alpha=2.0, z0=0.35)
    log_prior_z = np.log(z_probs + 1e-300)

    T_gen = gen_templates.shape[0]
    t_probs_gen = np.ones(T_gen, dtype=float) / T_gen

    ds = make_dataset(
        rng=rng,
        mu_grid=mu_grid_gen,
        z_grid=z_grid,
        N=N,
        z_max=2.0,
        z_probs=z_probs,
        sigma0_frac=sigma0_frac,
        k=kappa,
        t_probs=t_probs_gen,
    )

    print("\nGenerated fixed dataset:")
    print("  generation library =", generation_library)
    print("  x shape            =", ds.x.shape)
    print("  z min/max          =", float(ds.z.min()), float(ds.z.max()))

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

    results = {}

    for inf_lib in inference_libraries:
        print(f"\n=== Inference with template library: {inf_lib} ===")

        inf_templates, _ = load_templates(inf_lib, wl)
        mu_grid_inf = grid_mu(wl, inf_templates, R, z_grid)

        T_inf = inf_templates.shape[0]
        log_prior_t = np.log(np.ones(T_inf, dtype=float) / T_inf + 1e-300)

        z_mle, _ = batch_template_fit_mle(
            ds_test,
            mu_grid_inf,
            z_grid,
            progress_every=200,
        )
        metrics_mle = compute_metrics(y_test, z_mle)

        z_map, _ = batch_template_fit_map(
            ds_test,
            mu_grid_inf,
            z_grid,
            log_prior_z=log_prior_z,
            log_prior_t=log_prior_t,
            progress_every=200,
        )
        metrics_map = compute_metrics(y_test, z_map)

        results[inf_lib] = {
            "MLE": metrics_mle,
            "MAP": metrics_map,
            "kNN": metrics_knn,
            "RF": metrics_rf,
        }

    summary = {
        "generation_library": generation_library,
        "inference_libraries": inference_libraries,
        "N_total": int(N),
        "N_train": int(len(idx_train)),
        "N_test": int(len(idx_test)),
        "sigma0_frac": sigma0_frac,
        "kappa": kappa,
        "results": results,
    }

    with open(os.path.join(outdir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved metrics to:")
    print(" ", os.path.join(outdir, "metrics_summary.json"))

    x = np.arange(len(inference_libraries))
    libs_short = [library_labels[lib] for lib in inference_libraries]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), sharex=True)

    plot_metric_panel(
        axes[0],
        x,
        libs_short,
        results,
        metric_key="mae",
        ylabel="Mean absolute error",
        title="Accuracy under template mismatch",
    )

    plot_metric_panel(
        axes[1],
        x,
        libs_short,
        results,
        metric_key="outlier",
        ylabel="Catastrophic outlier rate",
        title="Outlier sensitivity under template mismatch",
    )

    axes[0].set_xlabel("Inference template library")
    axes[1].set_xlabel("Inference template library")

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

    combined_path = os.path.join(outdir, "template_mismatch_mae_outlier_combined")
    savefig_both(fig, combined_path)
    plt.close(fig)

    print("Saved figures:")
    print(" ", combined_path + ".png")
    print(" ", combined_path + ".pdf")


if __name__ == "__main__":
    main()