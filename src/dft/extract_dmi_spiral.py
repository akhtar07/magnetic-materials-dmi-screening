"""Extract the Dzyaloshinskii-Moriya vector components of the Fe-Fe chain bond in mp-1079285
(Tb6FeBi2) from the four constrained 90-degree spin-spiral calculations staged by
stage_rerun_batches.py (data/dft_staging_noncollinear/mp-1079285/spiral_{n1,n2}_{plus,minus}).

Model: E = sum_<ij> D_ij . (S_i x S_j) over the 4 nearest-neighbour Fe-Fe bonds of the 1x4x1
supercell (one Fe per primitive cell, chain along b). For a 90-degree spiral rotating in the plane
normal to n, S_i x S_j = +-n for every bond, the symmetric exchange contributes identically to both
chiralities, so D.n = [E(plus) - E(minus)] / (2 * 4) per bond (unit spins). n1 ~ a-perp, n2 = b x n1;
the b-component is forced to zero by the Pm mirror perpendicular to b (Moriya rule), so the two
projections determine |D| for this bond.

Checks before trusting a number: both members converged, constraint penalty energy small (the
"E_p" term reported by VASP with I_CONSTRAINED_M), the constrained Fe moments actually point along
their target directions with equal magnitude, and the induced Tb/Bi moments are the same in both.

Usage:
    python extract_dmi_spiral.py
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "dft_staging_noncollinear" / "mp-1079285"
TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-\d.]+)\s*eV")


def outcar(calc: Path) -> Path:
    for p in (calc / "results" / "OUTCAR", calc / "OUTCAR"):
        if p.exists():
            return p
    raise FileNotFoundError(calc)


def noncollinear_moments(txt: str) -> np.ndarray:
    """Per-ion (mx, my, mz) from the last magnetization (x)/(y)/(z) tables."""
    comps = []
    for ax in "xyz":
        i = txt.rfind(f"magnetization ({ax})")
        vals = []
        for line in txt[i:].splitlines()[4:]:
            parts = line.split()
            if len(parts) < 5 or not parts[0].isdigit():
                break
            vals.append(float(parts[-1]))
        comps.append(vals)
    return np.array(comps).T


def penalty_energy(calc: Path) -> float | None:
    # VASP prints "E_p = ..." (constraint penalty) with I_CONSTRAINED_M in OSZICAR (not OUTCAR);
    # take the last value
    for p in (calc / "results" / "OSZICAR", calc / "OSZICAR"):
        if p.exists():
            m = re.findall(r"E_p\s*=\s*([-\d.E+]+)", p.read_text(errors="ignore"))
            return float(m[-1]) if m else None
    return None


def scf_converged(calc: Path, txt: str) -> bool:
    """NSW=0 single point: VASP prints neither 'reached required accuracy' nor 'EDIFF is reached'
    in OUTCAR, so check OSZICAR directly -- last electronic |dE| below EDIFF and NELM not hit."""
    ediff = float(re.search(r"EDIFF\s*=\s*([-\d.E+]+)", txt).group(1))
    nelm = int(re.search(r"NELM\s*=\s*(\d+)", txt).group(1))
    for p in (calc / "results" / "OSZICAR", calc / "OSZICAR"):
        if p.exists():
            steps = re.findall(r"^\s*(?:DAV|RMM):\s+(\d+)\s+\S+\s+(\S+)", p.read_text(errors="ignore"), re.M)
            return bool(steps) and int(steps[-1][0]) < nelm and abs(float(steps[-1][1])) < ediff
    return False


E0_RE = re.compile(r"energy\s+without\s+entropy\s*=\s*([-\d.]+)\s+energy\(sigma->0\)\s*=\s*([-\d.]+)")


def main():
    results = {}
    for name in ("spiral_n1_plus", "spiral_n1_minus", "spiral_n2_plus", "spiral_n2_minus"):
        calc = BASE / name
        txt = outcar(calc).read_text(errors="ignore")
        meta = json.loads((calc / "meta.json").read_text())
        tot = TOTEN_RE.findall(txt)
        e0 = E0_RE.findall(txt)
        mom = noncollinear_moments(txt)
        fe = meta["fe_site_indices_in_chain_order"]
        fe_m = mom[fe]
        targets = np.array([meta["spin_directions"][str(i)] for i in fe])
        along = np.einsum("ij,ij->i", fe_m, targets)
        perp = np.linalg.norm(fe_m - along[:, None] * targets, axis=1)
        results[name] = dict(
            E=float(tot[-1]), E_sigma0=float(e0[-1][1]),
            converged=scf_converged(calc, txt),
            n_scf=len(tot), E_penalty=penalty_energy(calc),
            fe_along=along.round(3).tolist(), fe_perp=perp.round(3).tolist(),
            other_max=float(np.abs(np.delete(mom, fe, axis=0)).max()) if len(mom) > len(fe) else 0.0,
            normal=meta["rotation_plane_normal"], chirality=meta["chirality"])
        r = results[name]
        print(f"{name}: E={r['E']:.6f} eV E(sigma->0)={r['E_sigma0']:.6f} eV n_scf={r['n_scf']} converged={r['converged']} E_p={r['E_penalty']} "
              f"Fe|m|along={r['fe_along']} perp={r['fe_perp']} max|m| non-Fe={r['other_max']:.3f}")

    out = {}
    for n in ("n1", "n2"):
        plus, minus = results[f"spiral_{n}_plus"], results[f"spiral_{n}_minus"]
        dn = (plus["E"] - minus["E"]) / (2 * 4)
        dn0 = (plus["E_sigma0"] - minus["E_sigma0"]) / (2 * 4)
        out[n] = dict(normal=plus["normal"], D_dot_n_meV=dn * 1000.0,
                      D_dot_n_meV_from_E_sigma0=dn0 * 1000.0,
                      E_plus=plus["E"], E_minus=minus["E"],
                      both_converged=plus["converged"] and minus["converged"],
                      penalty_max_eV=max(x for x in (plus["E_penalty"], minus["E_penalty"]) if x is not None) if any(
                          x is not None for x in (plus["E_penalty"], minus["E_penalty"])) else None)
        print(f"D.{n} = {dn*1000:+.4f} meV/bond  (E+ - E- = {(plus['E']-minus['E'])*1000:+.4f} meV per 4 bonds; "
              f"from E(sigma->0): {dn0*1000:+.4f} meV/bond)")
    d1, d2 = out["n1"]["D_dot_n_meV"], out["n2"]["D_dot_n_meV"]
    out["D_magnitude_meV"] = float(np.hypot(d1, d2))
    out["note"] = ("|D| from the two in-plane-perpendicular projections; D_b = 0 by the Pm mirror normal to b. "
                   "Per Fe-Fe nearest-neighbour bond, unit spins, E = sum_<ij> D_ij.(S_i x S_j).")
    print(f"|D| (a-perp, c-perp components) = {out['D_magnitude_meV']:.4f} meV/bond")
    o = ROOT / "outputs" / "dmi_mp1079285_spiral.json"
    o.write_text(json.dumps({"per_calc": results, "dmi": out}, indent=2))
    print(f"wrote {o.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
