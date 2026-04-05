import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class EazyFilter:
    def __init__(self, name: str, wl_A: np.ndarray, R: np.ndarray):
        self.name = name
        self.wl_A = wl_A.astype(float)
        self.R = R.astype(float)

    def eff_wavelength_A(self) -> float:
        # Simple effective wavelength: ∫ λ R dλ / ∫ R dλ
        num = np.trapz(self.wl_A * self.R, self.wl_A)
        den = np.trapz(self.R, self.wl_A)
        return float(num / max(den, 1e-30))


def find_eazy_filters_res(eazy_root: str) -> Path:
    """
    Try to locate an EAZY filters-res file inside eazy_root/filters.
    Common filenames include FILTER.RES.latest, FILTER.RES, master.FILTERS.RES, etc.
    """
    filters_dir = Path(eazy_root) / "filters"
    if not filters_dir.exists():
        raise FileNotFoundError(f"Filters directory not found: {filters_dir}")

    preferred = [
        "FILTER.RES.latest",
        "FILTER.RES",
        "FILTERS.RES",
        "master.FILTERS.RES",
        "master.FILTER.RES",
    ]
    for name in preferred:
        p = filters_dir / name
        if p.exists():
            return p

    # fallback: pick any *RES* file
    candidates = sorted(list(filters_dir.glob("*.RES*")) + list(filters_dir.glob("*.res*")))
    if not candidates:
        raise FileNotFoundError(f"No FILTER*.RES* files found in {filters_dir}")
    return candidates[0]


def load_eazy_filters_res(path: str) -> List[EazyFilter]:
    """
    Parse an EAZY/HYPERZ-style FILTERS.RES file:
      Header line:  N  description...
      Next N lines: i  lambda(Angstrom)  R(lambda)

    Returns a list of EazyFilter entries in file order.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"filters_res file not found: {p}")

    lines = p.read_text(errors="ignore").splitlines()
    out: List[EazyFilter] = []
    k = 0
    nlines = len(lines)

    def is_header(line: str) -> bool:
        s = line.strip()
        if not s or s.startswith("#"):
            return False
        parts = s.split()
        if len(parts) < 2:
            return False
        # first token should be int N
        try:
            int(parts[0])
            return True
        except Exception:
            return False

    while k < nlines:
        line = lines[k].strip()
        if not is_header(line):
            k += 1
            continue

        parts = line.split()
        n = int(parts[0])
        name = " ".join(parts[1:]).strip()
        k += 1

        wl = []
        rr = []
        for _ in range(n):
            if k >= nlines:
                break
            row = lines[k].strip()
            k += 1
            if not row or row.startswith("#"):
                continue
            cols = row.split()
            if len(cols) < 3:
                continue
            # cols: i, lambda_A, R
            try:
                wl.append(float(cols[1]))
                rr.append(float(cols[2]))
            except Exception:
                continue

        wl_A = np.asarray(wl, float)
        R = np.asarray(rr, float)

        if wl_A.size < 5:
            continue

        # clean
        R = np.clip(R, 0.0, None)
        out.append(EazyFilter(name=name, wl_A=wl_A, R=R))

    if not out:
        raise ValueError(f"No filters parsed from {p}")

    return out


def build_R_matrix_on_grid(
    wavelengths_nm: np.ndarray,
    filters: List[EazyFilter],
    normalize: bool = True,
) -> np.ndarray:
    """
    Interpolate each filter response onto the wavelength grid (nm).
    Returns R matrix of shape (B, L).
    """
    wl_nm = np.asarray(wavelengths_nm, float)
    B = len(filters)
    L = wl_nm.size

    Rmat = np.zeros((B, L), dtype=float)

    for b, f in enumerate(filters):
        wl_nm_f = f.wl_A / 10.0  # Å -> nm
        R_on = np.interp(wl_nm, wl_nm_f, f.R, left=0.0, right=0.0)
        R_on = np.clip(R_on, 0.0, None)
        if normalize:
            area = np.trapz(R_on, wl_nm)
            R_on = R_on / max(area, 1e-30)
        Rmat[b] = R_on

    return Rmat


def select_filters_in_range(
    filters: List[EazyFilter],
    wlmin_nm: float,
    wlmax_nm: float,
    n: int = 5,
) -> List[EazyFilter]:
    """
    Choose n filters whose effective wavelengths fall in [wlmin_nm, wlmax_nm],
    spaced roughly evenly in effective wavelength.
    """
    candidates = []
    for f in filters:
        eff_nm = f.eff_wavelength_A() / 10.0
        if wlmin_nm <= eff_nm <= wlmax_nm:
            candidates.append((eff_nm, f))

    if len(candidates) < n:
        raise ValueError(
            f"Only {len(candidates)} filters have eff wavelength in [{wlmin_nm},{wlmax_nm}] nm; need {n}. "
            f"Increase wavelength range or lower n."
        )

    candidates.sort(key=lambda x: x[0])
    effs = np.array([c[0] for c in candidates])
    chosen = []

    # target effective wavelengths evenly spaced
    targets = np.linspace(effs.min(), effs.max(), n)
    used = set()

    for t in targets:
        j = int(np.argmin(np.abs(effs - t)))
        # avoid duplicates: walk outward if needed
        jj = j
        step = 1
        while jj in used and (j - step >= 0 or j + step < len(candidates)):
            left = j - step
            right = j + step
            if left >= 0 and left not in used:
                jj = left
                break
            if right < len(candidates) and right not in used:
                jj = right
                break
            step += 1

        used.add(jj)
        chosen.append(candidates[jj][1])

    # sort chosen by eff wavelength
    chosen.sort(key=lambda f: f.eff_wavelength_A())
    return chosen


def auto_wavelength_grid_from_filters(
    filters: List[EazyFilter],
    padding_frac: float = 0.05,
    step_nm: float = 0.5,
    wl_min_nm_floor: float = 300.0,
    wl_max_nm_ceil: float = 2500.0,
) -> np.ndarray:
    """
    Build a wavelength grid (nm) that covers the selected filters.
    """
    mins = []
    maxs = []
    for f in filters:
        wl_nm = f.wl_A / 10.0
        mins.append(float(wl_nm.min()))
        maxs.append(float(wl_nm.max()))

    lo = min(mins)
    hi = max(maxs)
    span = hi - lo
    lo = max(wl_min_nm_floor, lo - padding_frac * span)
    hi = min(wl_max_nm_ceil, hi + padding_frac * span)

    n = int(np.ceil((hi - lo) / step_nm)) + 1
    return np.linspace(lo, hi, n)

import numpy as np
from typing import List


def _filter_eff_nm(f) -> float:
    """
    Compute effective wavelength in nm for an EAZY filter object.

    Tries common attribute names. If no precomputed value exists,
    computes:
        eff = ∫ λ R(λ) dλ / ∫ R(λ) dλ
    """
    # 1) If the object already stores something usable, try it
    for name in ["eff_nm", "lambda_eff_nm", "lambda_eff", "eff"]:
        if hasattr(f, name):
            val = float(getattr(f, name))
            # if it looks like Angstrom, convert to nm
            if val > 2000:  # crude but safe
                return val / 10.0
            return val

    # 2) Otherwise compute from stored wavelength/response arrays
    # Try common wavelength attribute names (often in Angstrom in EAZY files)
    wl = None
    for name in ["wl", "wave", "wavelength", "lam", "lambda", "wavelength_A", "lambda_A", "wave_A", "wl_A"]:
        if hasattr(f, name):
            wl = np.asarray(getattr(f, name), float)
            break

    if wl is None:
        raise AttributeError("EazyFilter has no wavelength array attribute (wl/wave/lam/...).")

    # Try common response attribute names
    R = None
    for name in ["R", "resp", "response", "throughput", "trans", "t"]:
        if hasattr(f, name):
            R = np.asarray(getattr(f, name), float)
            break

    if R is None:
        raise AttributeError("EazyFilter has no response array attribute (R/resp/response/...).")

    # Compute effective wavelength in same units as wl
    num = np.trapz(wl * R, wl)
    den = np.trapz(R, wl)
    if den <= 0:
        raise ValueError("Filter response integrates to zero; cannot compute effective wavelength.")

    eff = num / den

    # Heuristic: EAZY filter curves are usually in Angstrom
    # If eff ~ thousands, assume Angstrom → convert to nm
    if eff > 2000:
        return float(eff / 10.0)
    return float(eff)


import numpy as np
from typing import List


def pick_filters_by_eff_wavelength(filters: List, B: int):
    """
    Pick B filters evenly spaced in effective wavelength.
    EazyFilter provides eff_wavelength_A (Angstrom).
    """
    if len(filters) < B:
        raise ValueError(f"Requested {B} filters, but only {len(filters)} available.")

    eff_nm = np.array([float(f.eff_wavelength_A()) / 10.0 for f in filters])
    good = (eff_nm > 250) & (eff_nm < 2000)   # optical+NIR
    filters = [f for f, g in zip(filters, good) if g]
    eff_nm = eff_nm[good]
    order = np.argsort(eff_nm)
    filters_sorted = [filters[i] for i in order]


    idx = np.linspace(0, len(filters_sorted) - 1, B).astype(int)
    return [filters_sorted[i] for i in idx]

