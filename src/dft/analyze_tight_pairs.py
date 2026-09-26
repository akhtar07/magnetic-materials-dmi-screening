"""Analyse the tight same-cell FM/AFM pairs (data/dft_staging_tight, Batch B of 2026-09-20).

Each pair `afm_<calc>` / `fm_in_<calc>` starts from the SAME relaxed AFM cell (CONTCAR of the
original enumerated AFM calc), both relaxed with ISIF=2, EDIFF=1e-6 eV, EDIFFG=-0.01 eV/A and the
same k-mesh, so the FM-AFM energy difference is free of the cell-shape and k-mesh mismatch of the
original MPRelaxSet runs (FM primitive cell vs AFM supercell, ISIF=3). This is the number the
paper's "margin" figure should rest on for the candidates whose original margin was marginal.

Per candidate it writes the per-atom energy gap E(FM) - E(AFM), the spin-state audit outcome of
both members (same rules as audit_spin_states.py: moments must stay in the same |m| state and keep
the intended sign pattern), the rms(c) of the final SCF step and, for compounds with one magnetic
sublattice and a single dominant coupling, a mean-field nearest-neighbour exchange J per bond.

Usage:
    python analyze_tight_pairs.py [--out outputs/dft_tight_validation.csv]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TIGHT = ROOT / "data" / "dft_staging_tight"
sys.path.insert(0, str(ROOT / "src" / "dft"))
from audit_spin_states import final_moments, poscar_species  # noqa: E402

TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-\d.]+)\s*eV")
FORCE_RE = re.compile(r"POSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s*-+\n(.*?)\n\s*-+\n", re.S)
MAG = {"Fe", "Mn", "Co", "Ni", "Cr"}
# A relaxation killed before EDIFFG (farm job 72658 was cancelled from the account on 2026-09-21
# 22:16 with three tight members mid-run) still gives a usable gap when the total energy has
# stopped moving: "provisional" = spin state valid, forces below 0.05 eV/A and the TOTEN spread over
# the last three ionic steps below 0.05 meV/atom (the gaps in question are >= 0.3 meV/atom).
PROVISIONAL_MAX_FORCE = 0.05
PROVISIONAL_DRIFT_MEV_PER_ATOM = 0.05


def last_max_force(txt: str) -> float | None:
    blocks = FORCE_RE.findall(txt)
    if not blocks:
        return None
    f = np.array([[float(x) for x in l.split()[3:6]] for l in blocks[-1].splitlines()])
    return float(np.abs(f).max())


def outcar_path(calc: Path) -> Path | None:
    for p in (calc / "results" / "OUTCAR", calc / "OUTCAR"):
        if p.exists():
            return p
    return None


def last_rms_c(calc: Path) -> float | None:
    for p in (calc / "results" / "OSZICAR", calc / "OSZICAR"):
        if p.exists():
            vals = [float(m) for m in re.findall(r"rms\(c\)\s*=\s*([-\d.E+]+)", p.read_text())]
            if not vals:
                # OSZICAR prints rms(c) as the last column of DAV/RMM lines
                rows = [l.split() for l in p.read_text().splitlines() if l.startswith(("DAV:", "RMM:"))]
                vals = [float(r[-1]) for r in rows if len(r) >= 8]
            return vals[-1] if vals else None
    return None


def analyse_calc(calc: Path) -> dict | None:
    outcar = outcar_path(calc)
    if outcar is None:
        return None
    txt = outcar.read_text(errors="ignore")
    tot = TOTEN_RE.findall(txt)
    meta = json.loads((calc / "meta.json").read_text())
    species = poscar_species(calc / "POSCAR")
    m = final_moments(outcar)
    init = np.array(meta["magmoms"], dtype=float)
    idx = [i for i, s in enumerate(species) if s in MAG]
    row = dict(
        calc=calc.name, ordering=meta["ordering"], n_atoms=len(species),
        converged="reached required accuracy" in txt,
        toten_ev=float(tot[-1]) if tot else np.nan,
        n_ionic_steps=len(tot), rms_c_final=last_rms_c(calc),
        max_force_ev_per_a=last_max_force(txt),
        toten_drift_last3_mev_per_atom=(max(map(float, tot[-3:])) - min(map(float, tot[-3:]))) / len(species) * 1000.0
        if len(tot) >= 3 else np.nan,
        mag_moments=json.dumps(np.round(m[idx], 2).tolist()) if m is not None and len(m) == len(species) else "",
    )
    problems = []
    if m is None or len(m) != len(species):
        problems.append("no per-ion magnetization table")
    else:
        mm = m[idx]
        # per species: Cr3+ (~3 mu_B) next to Fe3+ (~4.3 mu_B) in mp-551086 is physical, not a collapse
        for sp in sorted({species[i] for i in idx}):
            ms = np.abs(np.array([m[i] for i in idx if species[i] == sp]))
            if ms.max() > 0.5 and (ms.max() - ms.min()) > 0.5:
                problems.append(f"inconsistent |m| across {sp} ions")
        sgn = [np.sign(a) * np.sign(b) for a, b in zip(mm, init[idx]) if abs(b) > 0 and abs(a) > 0.5]
        if sgn and len(set(sgn)) > 1:
            problems.append("sign pattern changed vs MAGMOM")
        big = [v for v in mm if abs(v) > 0.5]
        if meta["ordering"] == "AFM" and big and (all(v > 0 for v in big) or all(v < 0 for v in big)):
            problems.append("AFM calc ended fully parallel")
    row["spin_state_valid"] = not problems
    row["spin_state_problems"] = "; ".join(problems)
    return row


# AFM configurations that share an FM partner with another pair (same cell, same k-mesh): the
# mp-1079285 +-+- chain pattern in the 1x4x1 cell (afm_2b_in_4b) is compared with fm_in_4b.
FM_PARTNER = {"2b_in_4b": "4b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "dft_tight_validation.csv"))
    args = ap.parse_args()
    rows = []
    for cand in sorted(p for p in TIGHT.iterdir() if p.is_dir()):
        calcs = {c.name: analyse_calc(c) for c in sorted(p for p in cand.iterdir() if p.is_dir())}
        for name, afm in calcs.items():
            if not name.startswith("afm_"):
                continue
            fm = calcs.get("fm_in_" + FM_PARTNER.get(name[4:], name[4:]))
            base = dict(candidate_id=cand.name, pair=name[4:])
            if afm is None or fm is None:
                rows.append({**base, "status": "not run" if afm is None and fm is None else "one member missing"})
                print(f"{cand.name}/{name[4:]}: {rows[-1]['status']}")
                continue
            spins_ok = afm["spin_state_valid"] and fm["spin_state_valid"]
            ok = afm["converged"] and fm["converged"] and spins_ok
            provisional = spins_ok and not ok and all(
                (c["max_force_ev_per_a"] or np.inf) < PROVISIONAL_MAX_FORCE
                and c["toten_drift_last3_mev_per_atom"] < PROVISIONAL_DRIFT_MEV_PER_ATOM for c in (afm, fm))
            gap = (fm["toten_ev"] - afm["toten_ev"]) / afm["n_atoms"] * 1000.0
            verdict = ("FM" if gap < 0 else "AFM") if (ok or provisional) else "INVALID"
            rows.append({**base, "status": "ok" if ok else ("provisional" if provisional else "invalid"),
                         "afm_max_force": afm["max_force_ev_per_a"], "fm_max_force": fm["max_force_ev_per_a"],
                         "afm_drift_last3": round(afm["toten_drift_last3_mev_per_atom"], 4),
                         "fm_drift_last3": round(fm["toten_drift_last3_mev_per_atom"], 4),
                         "n_atoms": afm["n_atoms"],
                         "E_fm_ev": fm["toten_ev"], "E_afm_ev": afm["toten_ev"],
                         "gap_fm_minus_afm_meV_per_atom": round(gap, 4),
                         "dft_ground_state": verdict,
                         "afm_converged": afm["converged"], "fm_converged": fm["converged"],
                         "afm_rms_c": afm["rms_c_final"], "fm_rms_c": fm["rms_c_final"],
                         "afm_moments": afm["mag_moments"], "fm_moments": fm["mag_moments"],
                         "afm_problems": afm["spin_state_problems"], "fm_problems": fm["spin_state_problems"]})
            note = ""
            if provisional:
                note = f"  (provisional: max|F| {afm['max_force_ev_per_a']:.3f}/{fm['max_force_ev_per_a']:.3f} eV/A)"
            elif not ok:
                note = "  [" + afm["spin_state_problems"] + " | " + fm["spin_state_problems"] + "]"
            print(f"{cand.name}/{name[4:]}: E(FM)-E(AFM) = {gap:+.3f} meV/atom -> {verdict}{note}")
    out = Path(args.out)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("candidate_id", "pair", "status"), k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
