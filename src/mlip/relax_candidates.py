"""Phase 3 applied to real candidates: relax each backfilled structure (screen_candidates.py +
backfill_structures.py) with the fine-tuned MACE model (Default head, Stage 2/SWA checkpoint) and
use the result as a sanity/ranking signal before anything goes to Phase 4 spin dynamics.

Why this matters as a filter, not just a relaxation step: every input structure here is already
DFT-relaxed *at its source database's own DFT settings* (MP/JARVIS/AFLOW all store relaxed
structures). Re-relaxing with our fine-tuned MACE potential and looking at how far it moves the
structure is a direct, cheap test of whether the potential agrees the structure is a local energy
minimum on ITS learned surface. A small relaxation energy / low final force residual means the
potential and the source DFT broadly agree; a large one means either the candidate is
out-of-distribution for the fine-tuned model or something is physically off -- worth flagging
before spending Phase 4 compute on it, not something to gloss over.

Position-only relaxation (fixed cell) is used deliberately for this first pass: the fine-tuned
model's stress RMSE (9-13, see project_status_report.pdf Sec 4.2) is decent but not validated
enough to trust for cell relaxation of never-before-seen compositions without checking basic
position-relaxation stability first.

Usage:
    python relax_candidates.py [--fmax 0.05] [--max-steps 200]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.optimize import FIRE
from mace.calculators import MACECalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "finetune" / "realrun_run" / "realrun_stagetwo.model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmax", type=float, default=0.05, help="force convergence threshold, eV/A")
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MACE model on {device}: {MODEL_PATH}")
    calc = MACECalculator(model_paths=str(MODEL_PATH), device=device, head="Default", default_dtype="float32")

    with open(ROOT / "data" / "processed" / "candidate_structures.json") as f:
        structures = json.load(f)
    meta = pd.read_csv(ROOT / "outputs" / "candidate_screened_full.csv").set_index("material_id")

    results = []
    n = sum(1 for v in structures.values() if "structure" in v)
    print(f"Relaxing {n} candidates (fmax={args.fmax}, max_steps={args.max_steps})...")

    for i, (mid, entry) in enumerate(structures.items(), 1):
        if "structure" not in entry:
            continue
        try:
            structure = Structure.from_dict(entry["structure"])
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.calc = calc

            e0 = atoms.get_potential_energy()
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=args.fmax, steps=args.max_steps)
            e1 = atoms.get_potential_energy()
            forces = atoms.get_forces()
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))
            n_atoms = len(atoms)

            row = {
                "material_id": mid,
                "n_atoms": n_atoms,
                "initial_energy_per_atom": e0 / n_atoms,
                "final_energy_per_atom": e1 / n_atoms,
                "relaxation_energy_per_atom": (e0 - e1) / n_atoms,
                "final_max_force": max_force,
                "converged": bool(max_force <= args.fmax),
                "n_steps": opt.nsteps,
            }
            if mid in meta.index:
                m = meta.loc[mid]
                row.update({
                    "source_database": m["source_database"], "formula": m["formula"],
                    "ordering_final": m["ordering_final"], "ordering_source": m["ordering_source"],
                    "confidence": m["confidence"], "energy_above_hull": m["energy_above_hull"],
                })
            results.append(row)
        except Exception as e:
            results.append({"material_id": mid, "error": f"{type(e).__name__}: {e}"})

        if i % 25 == 0:
            print(f"  [{i}/{n}] done")

    df = pd.DataFrame(results)
    out_path = ROOT / "outputs" / "candidate_relaxation.csv"
    df.to_csv(out_path, index=False)

    ok = df[df.get("error").isna()] if "error" in df.columns else df
    print(f"\n{len(ok)}/{len(df)} relaxed without error")
    print(f"Converged (max force <= {args.fmax}): {ok['converged'].sum()}/{len(ok)}")
    print(f"\nRelaxation energy per atom stats (eV/atom) -- large values flag OOD/questionable candidates:")
    print(ok["relaxation_energy_per_atom"].describe())

    # Rank: converged first, then small relaxation energy (agreement with source DFT), then
    # thermodynamic stability from the source database where known.
    ok = ok.copy()
    ok["abs_relax_e"] = ok["relaxation_energy_per_atom"].abs()
    ranked = ok.sort_values(by=["converged", "abs_relax_e"], ascending=[False, True])
    ranked.to_csv(ROOT / "outputs" / "candidate_relaxation_ranked.csv", index=False)
    print(f"\nWrote {out_path} and outputs/candidate_relaxation_ranked.csv")
    print("\nTop 15 most MACE-consistent candidates (converged, smallest relaxation shift):")
    cols = ["material_id", "source_database", "formula", "ordering_final", "n_atoms",
            "relaxation_energy_per_atom", "final_max_force", "energy_above_hull"]
    cols = [c for c in cols if c in ranked.columns]
    print(ranked[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
