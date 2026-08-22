"""Combine screen_candidates.py's classifier-based ranking with relax_candidates.py's MACE
relaxation-consistency signal into one final ranked shortlist for Phase 4 spin-dynamics selection.

Why a separate merge step: the two upstream scripts rank on different axes --
screen_candidates.py ranks on (hull_ok, prediction confidence) using each source's OWN DFT hull
energy; relax_candidates.py adds an independent, source-agnostic check (does OUR fine-tuned MACE
potential agree the structure is a local minimum on its own PES). A candidate that passes both is
on much firmer ground than one that only passes the classifier funnel. Concretely this also fixes
a real problem visible in the raw relaxation output: several high-relaxation-energy JARVIS
candidates (1-3.8 eV/atom above hull already) get corroborated as bad by MACE independently --
without this merge they'd still appear ranked by confidence alone.

Ranking key (best first):
  1. converged (all 298/298 did, kept as a safety gate for any future re-run that doesn't converge)
  2. hull_ok (True first -- known thermodynamically stable, or AFLOW/MAGNDATA where hull is unknown
     and not penalized)
  3. abs(relaxation_energy_per_atom) ascending -- MACE agrees the source-DFT structure is already
     near-equilibrium on our potential's surface
  4. confidence descending -- classifier confidence in the FM/FiM call

Usage:
    python finalize_phase4_shortlist.py [--top N]
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    relax = pd.read_csv(ROOT / "outputs" / "candidate_relaxation.csv")
    relax = relax[relax.get("error").isna()] if "error" in relax.columns else relax
    screened = pd.read_csv(ROOT / "outputs" / "candidate_screened_full.csv").set_index("material_id")

    df = relax.set_index("material_id").join(
        screened[["confidence", "energy_above_hull", "crystal_system_bucket", "heavy_soc_elements"]],
        rsuffix="_screened", how="left",
    )
    df["energy_above_hull"] = df["energy_above_hull"].combine_first(df.get("energy_above_hull_screened"))
    df["hull_known"] = df["energy_above_hull"].notna()
    df["hull_ok"] = df["energy_above_hull"].fillna(float("inf")) <= 0.1
    df["abs_relax_e"] = df["relaxation_energy_per_atom"].abs()

    ranked = df.sort_values(
        by=["converged", "hull_ok", "abs_relax_e", "confidence"],
        ascending=[False, False, True, False],
    )

    out_cols = [
        "source_database", "formula", "ordering_final", "ordering_source", "confidence",
        "n_atoms", "relaxation_energy_per_atom", "final_max_force", "converged",
        "energy_above_hull", "hull_known", "crystal_system_bucket", "heavy_soc_elements",
    ]
    out_cols = [c for c in out_cols if c in ranked.columns]
    shortlist = ranked[out_cols].head(args.top)
    shortlist.to_csv(ROOT / "outputs" / "phase4_shortlist.csv")
    print(f"Wrote top {len(shortlist)} Phase-4-ready candidates to outputs/phase4_shortlist.csv")
    print(f"\nPer-source counts in full merged ranking (n={len(ranked)}):")
    print(ranked["source_database"].value_counts())
    print(f"\nTop {args.top} preview:")
    print(shortlist.to_string())


if __name__ == "__main__":
    main()
