import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def _read_template_2col(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Template file {path} does not look like 2 columns.")
    wl = arr[:, 0].astype(float)
    f = arr[:, 1].astype(float)
    return wl, f


def _resolve_eazy_path(relpath: str, param_path: Path, base_dir: Optional[str]) -> Path:
    rp = Path(relpath)
    if rp.is_absolute():
        return rp
    param_dir = param_path.parent          # .../eazy-photoz/templates
    repo_root = param_dir.parent           # .../eazy-photoz
    user_base = Path(base_dir) if base_dir else None
    candidates: List[Path] = []
    if user_base is not None:
        candidates.append(user_base / rp)
    candidates.append(param_dir / rp)
    candidates.append(repo_root / rp)
    parts = rp.parts
    if len(parts) >= 2 and parts[0].lower() == "templates":
        rp_stripped = Path(*parts[1:])
        if user_base is not None:
            candidates.append(user_base / rp_stripped)
        candidates.append(param_dir / rp_stripped)
        candidates.append(repo_root / rp_stripped)

    for c in candidates:
        if c.exists():
            return c

    return (user_base / rp) if user_base is not None else (repo_root / rp)


def load_eazy_templates_from_spectra_param(
    wavelengths_nm: np.ndarray,
    spectra_param_file: str,
    base_dir: Optional[str] = None,
    max_templates: Optional[int] = None,
    normalize: str = "integral",          # "integral" or "ref"
    ref_wavelength_nm: float = 750.0,

) -> Tuple[np.ndarray, List[str]]:
    wl_target = np.asarray(wavelengths_nm, float)
    param_path = Path(spectra_param_file)
    if not param_path.exists():
        raise FileNotFoundError(f"spectra param file not found: {param_path}")

    if base_dir is None:
        # if param is .../eazy-photoz/templates/*.param, repo root is parent of templates
        base_dir_guess = str(param_path.parent.parent)
        base_dir = base_dir_guess

    lines = param_path.read_text().splitlines()

    entries: List[Tuple[str, float]] = []
    for line in lines:
        s = line.strip()
        if (not s) or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue

        relpath = parts[1]
        scale_to_A = float(parts[2])
        entries.append((relpath, scale_to_A))

    if not entries:
        raise ValueError(f"No valid template entries found in {param_path}")

    if max_templates is not None:
        entries = entries[: int(max_templates)]

    templates: List[np.ndarray] = []
    names: List[str] = []

    for relpath, scale_to_A in entries:
        p = _resolve_eazy_path(relpath, param_path, base_dir)
        if not p.exists():
            raise FileNotFoundError(
                f"Template file not found: {p}\n"
                f"  from entry '{relpath}' in {param_path}\n"
                f"  base_dir was: {base_dir}"
            )

        wl_raw, f_raw = _read_template_2col(p)

        # Convert wavelength: (raw * scale_to_A) -> Angstrom -> nm
        wl_nm = (wl_raw * scale_to_A) / 10.0

        f_on = np.interp(wl_target, wl_nm, f_raw, left=0.0, right=0.0)
        f_on = np.clip(f_on, 1e-12, None)

        if normalize == "integral":
            area = np.trapz(f_on, wl_target)
            f_on = f_on / max(area, 1e-30)
        elif normalize == "ref":
            fref = float(np.interp(float(ref_wavelength_nm), wl_target, f_on))
            f_on = f_on / max(fref, 1e-30)
        else:
            raise ValueError("normalize must be 'integral' or 'ref'")

        templates.append(f_on)
        names.append(p.name)

    return np.stack(templates, axis=0), names
