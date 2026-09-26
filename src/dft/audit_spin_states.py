"""Audit every converged magnetic-ordering calculation for spin-state integrity.

A collinear FM-vs-AFM energy comparison is only meaningful if every magnetic ion ends up in the
SAME spin state (same |moment|) in every configuration, with the intended sign pattern. Two
failure modes silently break that and were found in this project's batch on 2026-09-20:
  * spin-state collapse: some ions drop to low-/intermediate-spin (|m| ~ 0 or ~1.9 uB instead
    of ~3.75) during the SCF/relaxation -- the "AFM" energy is then a different electronic
    state, not an exchange-flipped one (mp-1173143: 140 meV/atom fake gap).
  * sign flip: the SCF re-orders the moments away from the MAGMOM pattern -- an "AFM" cell can
    end fully parallel (mp-682554/1_afm), so it is really a doubled FM cell.
Both pass VASP's own convergence test, so they must be checked from the final per-ion
magnetization table, not from "reached required accuracy".

Usage:
    python audit_spin_states.py [--tol 0.5] [--out outputs/dft_spin_state_audit.csv]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "data" / "dft_staging"
MAG_SPECIES = {"Fe", "Mn", "Co", "Ni", "Cr", "V", "Ti"}


def final_moments(outcar: Path) -> np.ndarray | None:
    txt = outcar.read_text(errors="ignore")
    i = txt.rfind("magnetization (x)")
    if i < 0:
        return None
    mom = []
    for line in txt[i:].splitlines()[4:]:
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            break
        mom.append(float(parts[-1]))
    return np.array(mom)


def poscar_species(poscar: Path) -> list[str]:
    lines = poscar.read_text().splitlines()
    out = []
    for sym, n in zip(lines[5].split(), lines[6].split()):
        out += [sym] * int(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.5,
                    help="max allowed |m| deviation (uB) of any magnetic ion from the per-species "
                         "reference |m| taken from that candidate's FM calc")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "dft_spin_state_audit.csv"))
    args = ap.parse_args()

    rows = []
    for cand in sorted(p for p in STAGING.iterdir() if p.is_dir() and p.name.startswith("mp-")):
        calcs = {}
        for calc in sorted(p for p in cand.iterdir() if p.is_dir()):
            outcar = calc / "results" / "OUTCAR"
            if not outcar.exists():
                outcar = calc / "OUTCAR"
            if not outcar.exists():
                continue
            sp = poscar_species(calc / "POSCAR")
            m = final_moments(outcar)
            if m is None or len(m) != len(sp):
                continue
            meta = json.loads((calc / "meta.json").read_text())
            if meta.get("superseded_by"):
                continue
            init = np.array(meta["magmoms"], dtype=float)
            conv = "reached required accuracy" in outcar.read_text(errors="ignore")
            calcs[calc.name] = dict(species=sp, m=m, init=init, conv=conv, ordering=meta["ordering"])

        # the reference FM calc is the FM-ordered calc carrying the largest moment: a candidate can
        # have several FM calcs (mp-22972: 0_fm collapsed to |m|~0, the 4_fm/5_fm reruns hold 1.6 muB)
        # and a collapsed one must neither set the reference nor drag the healthy ones down with it
        def _max_mag(c):
            return max([abs(c["m"][i]) for i, x in enumerate(c["species"]) if x in MAG_SPECIES] or [0.0])
        fm_calcs = [c for c in calcs.values() if c["ordering"] == "FM"]
        fm = max(fm_calcs, key=_max_mag) if fm_calcs else None
        # reference |m| values per magnetic species = the distinct |m| clusters seen in the FM calc
        # (a species may legitimately occupy two valence/spin states, e.g. Fe2+/Fe3+ in Li2Fe3WO8,
        # so the reference is a SET of values, not one number). If the FM calc itself collapsed to
        # ~0 while other configs carry a moment, the FM calc is flagged and the global max is used.
        ref, fm_collapsed = {}, {}
        for s in MAG_SPECIES:
            vals = [abs(c["m"][i]) for c in calcs.values() for i, x in enumerate(c["species"]) if x == s]
            if not vals:
                continue
            fm_vals = sorted(abs(fm["m"][i]) for i, x in enumerate(fm["species"]) if x == s) if fm else []
            clusters = []
            for v in fm_vals:
                if not clusters or v - clusters[-1][-1] > args.tol:
                    clusters.append([v])
                else:
                    clusters[-1].append(v)
            ref[s] = [float(np.median(cl)) for cl in clusters] or [float(max(vals))]
            fm_collapsed[s] = bool(fm_vals) and max(fm_vals) < 0.5 and max(vals) > 1.0
            if fm_collapsed[s]:
                ref[s] = [float(max(vals))]

        for name, c in calcs.items():
            problems = []
            per_species = {}
            all_big = []
            for s, rvals in ref.items():
                idx = [i for i, x in enumerate(c["species"]) if x == s]
                if not idx:
                    continue
                mm = c["m"][idx]
                per_species[s] = np.round(mm, 2).tolist()
                if c["ordering"] == "FM" and np.abs(mm).max() < 0.5 and max(rvals) > 1.0:
                    problems.append(f"{s}:FM calc collapsed to nonmagnetic (|m|={np.round(np.abs(mm),2).tolist()}, other configs reach {max(rvals):.2f})")
                elif max(rvals) > 0.5:
                    dev = np.array([min(abs(abs(v) - r) for r in rvals) for v in mm])
                    if dev.max() > args.tol:
                        problems.append(f"{s}:collapsed/inconsistent |m| (ref {np.round(rvals,2).tolist()}, got {np.round(np.abs(mm),2).tolist()})")
                ini = c["init"][idx]
                sgn = [np.sign(a) * np.sign(b) for a, b in zip(mm, ini) if abs(b) > 0 and abs(a) > 0.5]
                # a global flip (every sign reversed) is physically identical; only a PARTIAL
                # flip changes the ordering pattern
                if sgn and len(set(np.sign(sgn))) > 1:
                    problems.append(f"{s}:pattern changed vs MAGMOM")
                all_big += [v for v in mm if abs(v) > 0.5]
            # A single magnetic ion per cell (len(all_big) == 1) is the same defect in disguise: the
            # enumerator "antiparallel" partner was a d0 site (e.g. Ti4+ in mp-1216981, |m| ~ 0.06)
            # whose moment does not survive SCF, so the calc is just the FM state again.
            if c["ordering"] in ("AFM", "FiM") and len(all_big) >= 1 and (all(v > 0 for v in all_big) or all(v < 0 for v in all_big)):
                problems.append("labelled %s but every magnetic ion (|m|>0.5) is parallel (%s)" % (
                    c["ordering"], "FM in supercell" if len(all_big) > 1 else "only a non-magnetic d0 site was flipped"))
            rows.append(dict(candidate_id=cand.name, calc=name, ordering=c["ordering"], converged=c["conv"],
                             moments=json.dumps(per_species), reference=json.dumps({k: [round(x, 2) for x in v] for k, v in ref.items()}),
                             valid=not problems, problems="; ".join(problems)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if not r["valid"]]
    print(f"{len(rows)} calcs audited, {len(bad)} invalid -> {out.relative_to(ROOT)}")
    for r in bad:
        print(f"  {r['candidate_id']}/{r['calc']}: {r['problems']}")


if __name__ == "__main__":
    main()
