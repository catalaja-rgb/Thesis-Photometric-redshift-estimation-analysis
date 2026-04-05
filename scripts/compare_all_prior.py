import os
import numpy as np
import matplotlib.pyplot as plt

from photoz_sim.datasets import make_dataset
from photoz_sim.forward import grid_mu

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


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def compute_metrics(z_true: np.ndarray, z_est: np.ndarray):
    dz = (z_est - z_true) / (1.0 + z_true)
    mae = float(np.mean(np.abs(z_est - z_true)))
    bias = float(np.mean(dz))
    scatter = float(np.std(dz))
    outlier = float(np.mean(np.abs(dz) > 0.15))
    return {"mae": mae, "bias": bias, "scatter": scatter, "outlier": outlier}


def make_simple_redshift_prior(z_grid, alpha=2.0, z0=0.7):
    """
    A toy but standard-ish shape: p(z) ∝ z^alpha exp(-z/z0), truncated to the grid.
    This is NOT 'the' BPZ prior (that uses magnitude/type), but it shows the effect cleanly.
    """
    z = np.asarray(z_grid, float)
    p = (z ** alpha) * np.exp(-z / z0)
    p[0] = 0.0  # avoid any weirdness at z=0
    p = p / np.sum(p)
    return np.log(p + 1e-300)  # stable log


def run_one(label, templates, wl, R, z_grid, rng, N, log_prior_z, log_prior_t):
    mu_grid = grid_mu(wl, templates, R, z_grid)
    ds = make_dataset(rng, mu_grid, z_grid, N=N)  # NOTE: if z_true ~ Uniform, a 'correct' prior is flat.

    z_hat, t_hat = batch_template_fit_map(
        ds, mu_grid, z_grid,
        log_prior_z=log_prior_z,
        log_prior_t=log_prior_t,
        progress_every=200
    )

    metrics = compute_metrics(ds.z, z_hat)

    print(f"\n=== {label} (MAP) ===")
    print("templates:", templates.shape)
    print(metrics)

    return {
        "label": label,
        "T": int(templates.shape[0]),
        "z_true": ds.z,
        "z_hat": z_hat,
        "metrics": metrics,
    }


def format_metrics_text(r):
    m = r["metrics"]
    return (
        f"T={r['T']}\n"
        f"MAE={m['mae']:.3f}\n"
        f"bias={m['bias']:.3f}\n"
        f"scatter={m['scatter']:.3f}\n"
        f"outlier={m['outlier']:.3f}"
    )


def main():
    outdir = "outputs/2figures_all_templates_realfilters"
    ensure_dir(outdir)

    rng = np.random.default_rng(0)
    N = 1000
    z_grid = np.linspace(0.0, 2.0, 401)

    # ---- PRIORS ----
    # This is where priors actually enter.
    # If you set log_prior_z = zeros, you get back MLE behavior.
    log_prior_z = make_simple_redshift_prior(z_grid, alpha=2.0, z0=0.7)

    # template prior: start uniform (zeros). Later you can bias types if you want.
    log_prior_t = None

    # -----------------------
    # Real filters
    # -----------------------
    eazy_root = "data/external/eazy-photoz"
    filters_res = find_eazy_filters_res(eazy_root)
    all_filters = load_eazy_filters_res(str(filters_res))

    B = 5
    wlmin_nm, wlmax_nm = 300.0, 1100.0
    chosen_filters = select_filters_in_range(all_filters, wlmin_nm=wlmin_nm, wlmax_nm=wlmax_nm, n=B)

    print("Using FILTERS.RES:", str(filters_res))
    print(f"Chosen {B} filters (eff nm):")
    for f in chosen_filters:
        print(f"  {f.name}   eff={f.eff_wavelength_A()/10.0:.1f} nm")

    wl = auto_wavelength_grid_from_filters(
        chosen_filters,
        step_nm=0.5,
        wl_min_nm_floor=wlmin_nm,
        wl_max_nm_ceil=wlmax_nm,
    )
    R = build_R_matrix_on_grid(wl, chosen_filters, normalize=True)

    runs = []

    # 1) EAZY v1.3
    templates, _ = load_eazy_templates_from_spectra_param(
        wavelengths_nm=wl,
        spectra_param_file=f"{eazy_root}/templates/eazy_v1.3.spectra.param",
        base_dir=eazy_root,
        max_templates=None,
        normalize="integral",
    )
    runs.append(run_one("EAZY v1.3", templates, wl, R, z_grid, rng, N, log_prior_z, log_prior_t))

    # 2) CWW+KIN
    templates, _ = load_eazy_templates_from_spectra_param(
        wavelengths_nm=wl,
        spectra_param_file=f"{eazy_root}/templates/cww+kin.spectra.param",
        base_dir=eazy_root,
        max_templates=None,
        normalize="integral",
    )
    runs.append(run_one("CWW+KIN", templates, wl, R, z_grid, rng, N, log_prior_z, log_prior_t))

    # 3) PEGASE13
    templates, _ = load_eazy_templates_from_spectra_param(
        wavelengths_nm=wl,
        spectra_param_file=f"{eazy_root}/templates/pegase13.spectra.param",
        base_dir=eazy_root,
        max_templates=None,
        normalize="integral",
    )
    runs.append(run_one("PEGASE13", templates, wl, R, z_grid, rng, N, log_prior_z, log_prior_t))

    # 4) BPZ CWWSB4.list
    bpz_sed_dir = "data/external/bpz/bpz-1.99.3/SED"
    templates, _ = load_bpz_templates_from_list(
        wavelengths_nm=wl,
        bpz_sed_dir=bpz_sed_dir,
        list_file="CWWSB4.list",
        wl_unit="A",
    )
    runs.append(run_one("BPZ CWWSB4", templates, wl, R, z_grid, rng, N, log_prior_z, log_prior_t))

    # -----------------------
    # Plot 2x2
    # -----------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    zmin, zmax = float(z_grid.min()), float(z_grid.max())

    for ax, r in zip(axes, runs):
        ax.scatter(r["z_true"], r["z_hat"], s=6, alpha=0.25)
        ax.plot([zmin, zmax], [zmin, zmax], lw=1.5)
        ax.set_title(r["label"] + " (MAP)")

        ax.text(
            0.03, 0.97, format_metrics_text(r),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.85)
        )

        ax.set_xlim(zmin, zmax)
        ax.set_ylim(zmin, zmax)

    for ax in axes[2:]:
        ax.set_xlabel("z_true")
    for ax in axes[0::2]:
        ax.set_ylabel("z_hat")

    fig.suptitle("Template fitting with redshift prior (MAP) — same filters, same z-grid", y=0.98)
    plt.tight_layout()

    outpath = os.path.join(outdir, "compare_4panel_scatter_MAP_with_prior.png")
    plt.savefig(outpath, dpi=250)
    plt.close()

    print("\nSaved:", outpath)


if __name__ == "__main__":
    main()