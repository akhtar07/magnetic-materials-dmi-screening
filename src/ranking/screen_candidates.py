"""Phase 2: screen the full harmonized database for candidate magnetic materials worth carrying
into Phase 3 (MACE relaxation/energetics) and Phase 4 (spin-dynamics simulation).

Why this exists: Phases 1, 3, and 4 have each been validated in isolation (classifier trained and
evaluated on a held-out split; MACE fine-tuned on the harmonized dataset; spin dynamics run on one
hand-picked Pd/Fe/Ir(111)-type system) but nothing has connected them -- there was no step that
takes the classifier's predictions over the WHOLE database and turns them into a ranked shortlist.
src/ranking/ and src/bias/ existed as empty directories in this repo before this script; this is
the first thing to land in either.

This also resolves the JARVIS ordering-label gap found while retraining the v2 classifier: JARVIS
(19,116 records) has zero populated `ordering` labels and was previously silently unusable. Rather
than hand-picking a heuristic threshold on JARVIS's raw magnetic-moment field (a modeling judgment
call), this script does the more principled thing and simply predicts `ordering` for it (and every
other unlabeled record) with the trained classifier -- that is exactly what a bias-correction
classifier trained on real MAGNDATA ground truth is for (see dataset_specification.md: "the field
the Phase 1 bias classifier corrects").

Candidate criteria (screening for DMI-compatible, bulk skyrmion/spin-spiral-hosting systems, per
the same physical reasoning already used for the Pd/Fe/Ir(111) Phase 4 system -- see
project_status_report.pdf Sec. 5):
  1. Predicted or ground-truth ordering is FM or FiM (a net moment is needed for the kind of
     interfacial/bulk spin-texture readout this project targets; AFM is excluded here even though
     antiferromagnetic skyrmions exist in principle, since Phase 4's existing simulation code
     assumes a net-moment texture).
  2. Noncentrosymmetric space group -- DMI is forbidden by inversion symmetry, so any bulk DMI
     mechanism (Moriya's original criterion) requires this. Verified computationally (not
     hardcoded from memory): checked all 230 space groups' symmetry operations for an inversion
     operator, found 92 centrosymmetric / 138 noncentrosymmetric, matching the textbook count.
  3. Contains at least one heavy, strong-spin-orbit-coupling element (5d transition metals + Bi) --
     the classic DMI-enhancing species in both Fert-Levy interfacial and bulk Moriya mechanisms
     (e.g. Pt, Ir, W, Ta in synthetic multilayers). This is a heuristic filter, not a computed DMI
     value -- flag for anyone downstream expecting an actual DMI magnitude, which nothing in this
     pipeline computes yet.
  4. Thermodynamically plausible: energy_above_hull <= 0.1 eV/atom where known (92.6% coverage,
     MP + JARVIS only); records with unknown hull energy are kept but ranked below known-stable
     ones rather than excluded, since AFLOW/MAGNDATA lack this field entirely and shouldn't be
     penalized to zero for a field their source never provided.

Usage:
    python screen_candidates.py [--top N]
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from pymatgen.symmetry.groups import SpaceGroup
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "baseline"))
from build_features_v2 import (  # noqa: E402
    MAGNETIC_ELEMENTS, composition_features, crystal_system_from_sg, local_structure_features,
)

CATEGORICAL_COLS = ["space_group_number", "crystal_system_bucket"]
HEAVY_SOC_ELEMENTS = {"Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Bi"}


def _is_centrosymmetric(sg_number: int) -> bool | None:
    if sg_number is None or not (1 <= sg_number <= 230):
        return None
    sg = SpaceGroup.from_int_number(int(sg_number))
    return any(np.allclose(op.rotation_matrix, -np.eye(3)) for op in sg.symmetry_ops)


def build_all_features(records: list[dict]) -> pd.DataFrame:
    """Same feature recipe as build_features_v2.py, but for every record with a composition,
    labeled or not -- unlabeled rows are what we need predictions for."""
    rows = []
    for r in records:
        if not r.get("elements"):
            continue
        feats = composition_features(r["elements"])
        if not feats:
            continue
        feats["material_id"] = r["material_id"]
        feats["source_database"] = r["source_database"]
        feats["formula"] = r.get("formula")
        feats["elements_list"] = r["elements"]
        feats["density"] = r.get("density")
        feats["nsites"] = r.get("nsites")
        feats["volume"] = r.get("volume")
        sg = r.get("space_group_number")
        # some sources (JARVIS) store this as a numeric string rather than an int -- coerce.
        try:
            sg = int(float(sg)) if sg is not None else None
        except (TypeError, ValueError):
            sg = None
        feats["space_group_number"] = sg
        feats["crystal_system_bucket"] = crystal_system_from_sg(sg)
        feats["energy_above_hull"] = r.get("energy_above_hull")

        if r.get("structure") is not None:
            feats.update(local_structure_features(r["structure"]))
        else:
            feats.update({
                "n_magnetic_sites": np.nan, "fraction_magnetic_sites": np.nan,
                "mean_magmag_nn_distance": np.nan, "min_magmag_nn_distance": np.nan,
                "std_magmag_nn_distance": np.nan, "mean_magnetic_coordination": np.nan,
            })

        ordering = r.get("ordering")
        feats["ordering"] = ordering if ordering and ordering != "Unknown" else None
        rows.append(feats)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200, help="how many top candidates to write out")
    args = ap.parse_args()

    with open(ROOT / "data" / "processed" / "harmonized_v2.json") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} harmonized records")

    df = build_all_features(records)
    labeled_mask = df["ordering"].notna()
    print(f"{labeled_mask.sum()} labeled, {(~labeled_mask).sum()} unlabeled")

    model_feature_cols = [
        c for c in df.columns
        if c not in ("material_id", "source_database", "formula", "elements_list", "ordering",
                     "energy_above_hull")
    ]
    X_all = df[model_feature_cols].copy()
    for c in CATEGORICAL_COLS:
        X_all[c] = X_all[c].astype("category")

    encoder = LabelEncoder()
    y_labeled = encoder.fit_transform(df.loc[labeled_mask, "ordering"])
    print("Classes:", list(encoder.classes_))

    # Deployment model: train on ALL labeled data (not a train/test split -- that evaluation
    # already happened in train_eval_lightgbm_v2.py, 67.0% accuracy / 68.1% balanced accuracy;
    # this is the final model used to actually screen the unlabeled majority of the database).
    model = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05, objective="multiclass",
        class_weight="balanced", random_state=42, verbosity=-1,
    )
    model.fit(X_all[labeled_mask], y_labeled, categorical_feature=CATEGORICAL_COLS)

    proba = model.predict_proba(X_all)
    pred_idx = proba.argmax(axis=1)
    pred_ordering = encoder.inverse_transform(pred_idx)
    pred_confidence = proba.max(axis=1)

    df["ordering_final"] = np.where(labeled_mask, df["ordering"], pred_ordering)
    df["ordering_source"] = np.where(labeled_mask, "ground_truth", "predicted")
    df["confidence"] = np.where(labeled_mask, 1.0, pred_confidence)

    print("\nordering_final distribution (ground truth + predicted combined):")
    print(df["ordering_final"].value_counts())
    print("\nPredicted-only distribution (the 25,495 previously-unlabeled records, incl. all JARVIS):")
    print(df.loc[~labeled_mask, "ordering_final"].value_counts())

    # --- Screening filters ---
    sg_cache: dict[int, bool | None] = {}

    def noncentro(sg):
        if pd.isna(sg):
            return None
        sg = int(sg)
        if sg not in sg_cache:
            sg_cache[sg] = _is_centrosymmetric(sg)
        centro = sg_cache[sg]
        return None if centro is None else not centro

    df["is_noncentrosymmetric"] = df["space_group_number"].apply(noncentro)
    df["heavy_soc_elements"] = df["elements_list"].apply(
        lambda els: sorted(set(els) & HEAVY_SOC_ELEMENTS) if isinstance(els, list) else []
    )
    df["has_heavy_soc"] = df["heavy_soc_elements"].apply(len).gt(0)

    funnel = {"total": len(df)}
    step = df[df["ordering_final"].isin(["FM", "FiM"])]
    funnel["FM/FiM ordering"] = len(step)
    step = step[step["is_noncentrosymmetric"] == True]  # noqa: E712
    funnel["+ noncentrosymmetric"] = len(step)
    step = step[step["has_heavy_soc"]]
    funnel["+ heavy-SOC element present"] = len(step)
    print("\nScreening funnel:")
    for k, v in funnel.items():
        print(f"  {k}: {v}")

    step = step.copy()
    step["hull_known"] = step["energy_above_hull"].notna()
    step["hull_ok"] = step["energy_above_hull"].fillna(np.inf) <= 0.1
    step = step.sort_values(
        by=["hull_ok", "confidence", "hull_known"], ascending=[False, False, False]
    )

    out_cols = [
        "material_id", "source_database", "formula", "ordering_final", "ordering_source",
        "confidence", "space_group_number", "crystal_system_bucket", "heavy_soc_elements",
        "energy_above_hull", "density", "volume", "nsites",
    ]
    shortlist = step[out_cols].head(args.top)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(out_dir / "candidate_shortlist.csv", index=False)
    print(f"\nWrote top {len(shortlist)} candidates to {out_dir / 'candidate_shortlist.csv'}")
    print("\nTop 15 preview:")
    print(shortlist.head(15).to_string(index=False))

    # Full screened set (not just top-N) for downstream use / re-ranking without retraining.
    step[out_cols].to_csv(out_dir / "candidate_screened_full.csv", index=False)
    print(f"Wrote full screened set ({len(step)} rows) to {out_dir / 'candidate_screened_full.csv'}")


if __name__ == "__main__":
    main()
