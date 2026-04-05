import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

def load_bpz_template_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="#")
    wl = arr[:, 0].astype(float)
    f = arr[:, 1].astype(float)
    return wl, f


def load_bpz_templates(
    wavelengths_nm: np.ndarray,
    bpz_sed_dir: str,
    max_templates: Optional[int] = None,
    wl_unit: str = "A",  # BPZ is usually Angstrom
) -> Tuple[np.ndarray, List[str]]:

    sed_path = Path(bpz_sed_dir)

    files = sorted([p for p in sed_path.iterdir()
                    if p.is_file() and p.suffix.lower() == ".sed"])

    if len(files) == 0:
        raise FileNotFoundError(
            f"No .sed templates found in {bpz_sed_dir}. "
            
        )

    if max_templates is not None:
        files = files[:max_templates]

    wl_target = np.asarray(wavelengths_nm, float)

    templates = []
    names = []

    for p in files:
        wl_i, f_i = load_bpz_template_file(p)

        # --- Convert BPZ wavelength units to nm to match wl_target ---
        if wl_unit.lower() in ["a", "ang", "angstrom", "angstroms"]:
            wl_i = wl_i / 10.0
        elif wl_unit.lower() == "nm":
            pass
        else:
            raise ValueError("wl_unit must be 'A' (Angstrom) or 'nm'")

        # Interpolate onto common grid (nm)
        f_on = np.interp(wl_target, wl_i, f_i, left=0.0, right=0.0)

        # Ensure positivity and normalize (shape matters)
        f_on = np.clip(f_on, 1e-12, None)
        area = np.trapz(f_on, wl_target)
        f_on = f_on / max(area, 1e-30)

        templates.append(f_on)
        names.append(p.name)

    return np.stack(templates, axis=0), names

from pathlib import Path
import numpy as np
from typing import Tuple, List

def load_bpz_templates_from_list(
    wavelengths_nm: np.ndarray,
    bpz_sed_dir: str,
    list_file: str,
    wl_unit: str = "A",
) -> Tuple[np.ndarray, List[str]]:
    sed_path = Path(bpz_sed_dir)
    list_path = sed_path / list_file
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")

    names = []
    for line in list_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # BPZ list files usually contain filenames (sometimes with extra columns)
        parts = s.split()
        names.append(parts[0])

    templates = []
    wl_target = np.asarray(wavelengths_nm, float)

    for name in names:
        p = sed_path / name
        arr = np.loadtxt(p, comments="#")
        wl_i = arr[:, 0].astype(float)
        f_i = arr[:, 1].astype(float)

        if wl_unit.lower() in ["a", "ang", "angstrom", "angstroms"]:
            wl_i = wl_i / 10.0  # Å -> nm

        f_on = np.interp(wl_target, wl_i, f_i, left=0.0, right=0.0)
        f_on = np.clip(f_on, 1e-12, None)
        area = np.trapz(f_on, wl_target)
        f_on = f_on / max(area, 1e-30)

        templates.append(f_on)

    return np.stack(templates, axis=0), names
