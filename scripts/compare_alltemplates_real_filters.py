import os
import numpy as np
import matplotlib.pyplot as plt

from photoz_sim.datasets import make_dataset
from photoz_sim.forward import grid_mu
from photoz_sim.methods.template_fit_grid_mle import batch_template_fit_mle

from photoz_sim.templates_eazy import load_eazy_templates_from_spectra_param
from photoz_sim.templates_bpz import load_bpz_templates, load_bpz_templates_from_list

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
    return mae, bias, scatter, outlier


def run_one(label, templates, wl, R, z_grid, rng, N, outdir):
    mu_grid = grid_mu(wl, templates, R, z_grid)
    ds = make_dataset(rng, mu_grid, z_grid, N=N)

    z_mle, t_mle = batch_template_fit_mle(ds, mu_grid, z_grid, progress_every=200)

    mae, bias, scatter, outlier = compute_metrics(ds.z, z_mle)

    print(f"\n=== {label} ===")
    print("templates:", templates.shape)
    print("Done.")
    print("Mean |z_MLE - z_true|:", mae)
    print("bias (mean dz):", bias)
    print("scatter (std dz):", scatter)
    print("outlier rate |dz|>0.15:", outlier)

    # save scatter plot
    plt.figure()
    plt.scatter(ds.z, z_mle, s=6, alpha=0.3)
    plt.plot([z_grid.min(), z_grid.max()], [z_grid.min(), z_grid.max()])
    plt.xlabel("z_true")
    plt.ylabel("z_MLE")
    plt.title(f"{label} (MLE) — real filters")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_scatter.png"), dpi=200)
    plt.close()

    return {"label": label, "T": int(templates.shape[0]), "mae": mae, "bias": bias, "scatter": scatter, "outlier": outlier}


def main():
    outdir = "outputs/figures_all_templates_realfilters"
    ensure_dir(outdir)

    rng = np.random.default_rng(0)
    N = 1000
    z_grid = np.linspace(0.0, 2.0, 401)

    # -----------------------
    # Real filters (EAZY FILTER.RES.latest)
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

    results = []

    # -----------------------
    # EAZY template sets
    # -----------------------
    eazy_sets = [
        ("eazy_v1.3",  f"{eazy_root}/templates/eazy_v1.3.spectra.param"),
        ("cww+kin",    f"{eazy_root}/templates/cww+kin.spectra.param"),
        ("pegase13",   f"{eazy_root}/templates/pegase13.spectra.param"),
    ]

    for label, param in eazy_sets:
        templates, names = load_eazy_templates_from_spectra_param(
            wavelengths_nm=wl,
            spectra_param_file=param,
            base_dir=eazy_root,
            max_templates=None,
            normalize="integral",
        )
        results.append(run_one(label, templates, wl, R, z_grid, rng, N, outdir))

    # -----------------------
    # BPZ templates (same wl, same R)
    # -----------------------
    bpz_sed_dir = "data/external/bpz/bpz-1.99.3/SED"

    # A) BPZ: first 6 .sed files (quick baseline)
    bpz6_templates, bpz6_names = load_bpz_templates(
        wavelengths_nm=wl,
        bpz_sed_dir=bpz_sed_dir,
        max_templates=6,
        wl_unit="A",
    )
    results.append(run_one("bpz_first6", bpz6_templates, wl, R, z_grid, rng, N, outdir))

    # B) BPZ: CWWSB4.list (classic)
    cw_templates, cw_names = load_bpz_templates_from_list(
        wavelengths_nm=wl,
        bpz_sed_dir=bpz_sed_dir,
        list_file="CWWSB4.list",
        wl_unit="A",
    )
    results.append(run_one("bpz_CWWSB4", cw_templates, wl, R, z_grid, rng, N, outdir))

    print("\n=== SUMMARY ===")
    for r in results:
        print(r)
    print("\nSaved figures to:", outdir)


if __name__ == "__main__":
    main()
