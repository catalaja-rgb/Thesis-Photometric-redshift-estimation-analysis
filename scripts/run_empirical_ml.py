# scripts/run_empirical_ml.py
import os
import numpy as np
import json
import matplotlib.pyplot as plt

from photoz_sim.datasets import make_dataset
from photoz_sim.forward import grid_mu
from photoz_sim.templates_bpz import load_bpz_templates_from_list
from photoz_sim.filters_eazy import load_eazy_filters_res

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def build_R_from_eazy_filters(chosen, wl_nm: np.ndarray) -> np.ndarray:
    R = np.stack(
        [np.interp(wl_nm, f.wl_A / 10.0, f.R, left=0.0, right=0.0) for f in chosen],
        axis=0,
    )
    area = np.trapz(R, wl_nm, axis=1)
    return R / np.clip(area[:, None], 1e-30, None)


def metrics(z_true: np.ndarray, z_pred: np.ndarray) -> dict:
    dz = (z_pred - z_true) / (1.0 + z_true)
    return {
        "mae": float(np.mean(np.abs(z_pred - z_true))),
        "bias": float(np.mean(dz)),
        "scatter": float(np.std(dz)),
        "outlier": float(np.mean(np.abs(dz) > 0.15)),
    }


def pick_filters_by_prefix(filters, wanted_prefixes):
    chosen, missing = [], []
    for pref in wanted_prefixes:
        matches = [f for f in filters if f.name.startswith(pref)]
        if not matches:
            missing.append(pref)
        else:
            chosen.append(matches[0])
    if missing:
        examples = [filters[i].name for i in range(min(25, len(filters)))]
        raise RuntimeError(
            f"Missing filters (by prefix) in FILTER.RES: {missing}\n"
            f"First {len(examples)} filter names:\n  " + "\n  ".join(examples)
        )
    return chosen

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def dz_norm(z_true, z_pred):
    z_true = np.asarray(z_true)
    z_pred = np.asarray(z_pred)
    return (z_pred - z_true) / (1.0 + z_true)


def save_slide_figures(outdir, y_test, y_knn, y_rf, y_gbdt):
    ensure_dir(outdir)

    zt = np.asarray(y_test)
    preds = {
        "kNN": np.asarray(y_knn),
        "RF": np.asarray(y_rf),
        "HGBR": np.asarray(y_gbdt),
    }

    # ---- save metrics JSON (nice for your report)
    def metrics_local(z_true, z_pred):
        dz = dz_norm(z_true, z_pred)
        return {
            "mae": float(np.mean(np.abs(z_pred - z_true))),
            "bias": float(np.mean(dz)),
            "scatter": float(np.std(dz)),
            "outlier": float(np.mean(np.abs(dz) > 0.15)),
        }

    m = {name: metrics_local(zt, zp) for name, zp in preds.items()}
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)

    # consistent axes limits
    zmin = float(np.min(zt))
    zmax = float(np.max(zt))

    # ========= Figure 1: z_pred vs z_true (1×3) =========
    fig = plt.figure(figsize=(12, 4))
    for i, name in enumerate(["kNN", "RF", "HGBR"], start=1):
        ax = plt.subplot(1, 3, i)
        ax.scatter(zt, preds[name], s=7, alpha=0.25)
        ax.plot([zmin, zmax], [zmin, zmax], linewidth=1.0)
        ax.set_xlim(zmin, zmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel(r"$z_{\mathrm{true}}$")
        ax.set_ylabel(r"$z_{\mathrm{pred}}$")
        ax.set_title(
            f"{name}\n"
            f"MAE={m[name]['mae']:.3f}  bias={m[name]['bias']:.3f}\n"
            f"scat={m[name]['scatter']:.3f}  out={m[name]['outlier']:.3f}",
            fontsize=10
        )
        ax.grid(alpha=0.15)

    fig.tight_layout()
    p1 = os.path.join(outdir, "pred_vs_true_3models.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)

    # ========= Figure 2: dz histogram =========
    fig = plt.figure(figsize=(7.5, 4.5))
    bins = np.linspace(-0.5, 0.5, 80)
    for name in ["kNN", "RF", "HGBR"]:
        dz = dz_norm(zt, preds[name])
        plt.hist(dz, bins=bins, alpha=0.35, label=name)
    plt.xlabel(r"$\Delta z = (z_{\mathrm{pred}}-z_{\mathrm{true}})/(1+z_{\mathrm{true}})$")
    plt.ylabel("Count")
    plt.title(r"Normalized redshift error distribution")
    plt.grid(alpha=0.15)
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(outdir, "dz_hist_3models.png")
    plt.savefig(p2, dpi=300)
    plt.close()

    # ========= Figure 3: Outlier rate vs z_true =========
    # Define z bins
    nbins = 12
    edges = np.linspace(zmin, zmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=(7.5, 4.5))
    for name in ["kNN", "RF", "HGBR"]:
        dz = dz_norm(zt, preds[name])
        out = (np.abs(dz) > 0.15).astype(float)

        out_rate = np.empty(nbins)
        out_rate[:] = np.nan
        for b in range(nbins):
            mask = (zt >= edges[b]) & (zt < edges[b + 1])
            if np.any(mask):
                out_rate[b] = float(np.mean(out[mask]))

        plt.plot(centers, out_rate, marker="o", linewidth=1.5, label=name)

    plt.xlabel(r"$z_{\mathrm{true}}$ (binned)")
    plt.ylabel(r"Outlier rate  $\mathbb{P}(|\Delta z|>0.15)$")
    plt.ylim(0, 1)
    plt.title("Where do models fail? Outliers vs redshift")
    plt.grid(alpha=0.15)
    plt.legend()
    plt.tight_layout()
    p3 = os.path.join(outdir, "outlier_rate_vs_z.png")
    plt.savefig(p3, dpi=300)
    plt.close()

    print("Saved slide figures:")
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)


def main():
    rng = np.random.default_rng(0)

    wl = np.linspace(300.0, 1100.0, 1601)  # nm
    z_grid = np.linspace(0.0, 2.0, 401)

    templates, names = load_bpz_templates_from_list(
        wavelengths_nm=wl,
        bpz_sed_dir="data/external/bpz/bpz-1.99.3/SED",
        list_file="CWWSB4.list",
        wl_unit="A",
    )
    print("Templates:", templates.shape, "example:", names[:5])

    FILTER_RES = "data/external/eazy-photoz/filters/FILTER.RES.latest"
    filters = load_eazy_filters_res(FILTER_RES)

    WANTED_PREFIX = [
        "hst/wfpc2_f300w.dat",
        "hst/wfc3/UVIS/f475x.dat",
        "hst/wfpc2_f702w.dat",
        "REST_FRAME/Gunn_z.dat",
        "Euclid_NISP.Y.dat",
    ]
    chosen = pick_filters_by_prefix(filters, WANTED_PREFIX)

    print("\nChosen 5 filters (eff nm):")
    for f in chosen:
        print(f"  {f.name}   eff={float(f.eff_wavelength_A())/10.0:.1f} nm")

    R = build_R_from_eazy_filters(chosen, wl)

    mu_grid = grid_mu(wl, templates, R, z_grid)

    N = 5000
    ds = make_dataset(rng, mu_grid, z_grid, N=N)
    print("\nDataset:", ds.x.shape, "z range:", float(ds.z.min()), float(ds.z.max()))

    # RAW FLUX FEATURES (back to the good setup)
    X = ds.x
    y = ds.z

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # kNN (scaled)
    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=25, weights="distance")),
        ]
    )
    knn.fit(X_train, y_train)
    y_knn = knn.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=0,
        n_jobs=-1,
        min_samples_leaf=3,
    )
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)

    # Gradient Boosting
    gbdt = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=1e-2,
        random_state=0,
    )
    gbdt.fit(X_train, y_train)
    y_gbdt = gbdt.predict(X_test)

    print("\n=== Empirical ML results (train/test split) ===")
    print("kNN  :", metrics(y_test, y_knn))
    print("RF   :", metrics(y_test, y_rf))
    print("HGBR :", metrics(y_test, y_gbdt))
    outdir = "outputs/empirical_ml_slidefigs"
    save_slide_figures(outdir, y_test, y_knn, y_rf, y_gbdt)


if __name__ == "__main__":
    main()
