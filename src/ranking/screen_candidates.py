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
  4. Thermodynamically plausible: energy_above_hull <= 0.1 eV/atom where known (MP, JARVIS
     [recomputed, see below] and OQMD: 1,340 of the 1,864 funnel survivors on 2026-09-20);
     records with unknown hull energy are kept but ranked below known-stable
     ones rather than excluded, since AFLOW/MAGNDATA lack this field entirely and shouldn't be
     penalized to zero for a field their source never provided.

  Revision 2026-09-20: (a) the JARVIS `ehull` field copied into the harmonized dataset is not a
     hull distance (median 1.7 eV/atom over the whole snapshot; see recompute_jarvis_hull.py) --
     the hull energy of every JARVIS record is now replaced by the value recomputed from JARVIS's
     own formation energies (data/processed/jarvis_ehull_recomputed.csv), which is what the
     "MP + JARVIS" coverage statement in the paper implied all along. (b) `--hull-hard` makes
     criterion 4 an exclusion (known hull > 0.1 eV/atom dropped; unknown kept) and ranks known-
     stable records by hull energy before confidence; its outputs go to *_hullhard.csv so the
     original ranking that the DFT batch was drawn from stays reproducible.
  (c) Tier-2 local-geometry features are now computed for every source whose structure could be
     fetched (fetch_structures_all.py + attach_structures), not only MAGNDATA/2DMatPedia, so the
     deployment model and the screened records use the same feature definition -- previously
     "Tier-2 missing" was a proxy for the source database.

Usage:
    python screen_candidates.py [--top N] [--hull-hard]
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
    MAGNETIC_ELEMENTS, attach_structures, composition_features, crystal_system_from_sg, tier2_features_parallel,
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

        ordering = r.get("ordering")
        feats["ordering"] = ordering if ordering and ordering != "Unknown" else None
        rows.append((feats, r.get("structure")))

    # Tier 2 local-geometry features for every row with a structure (all sources since the
    # 2026-09-20 revision -- see build_features_v2.attach_structures); NaN where none is available.
    for (feats, _), t2 in zip(rows, tier2_features_parallel([st for _, st in rows])):
        feats.update(t2)
    return pd.DataFrame([f for f, _ in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200, help="how many top candidates to write out")
    ap.add_argument("--hull-hard", action="store_true",
                    help="exclude records with a KNOWN hull energy > 0.1 eV/atom and rank stable ones by "
                         "hull energy first; writes *_hullhard.csv")
    args = ap.parse_args()

    with open(ROOT / "data" / "processed" / "harmonized_v2.json") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} harmonized records")
    coverage = attach_structures(records)
    print("structure coverage for Tier-2 features:", {k: f"{v['with_structure']}/{v['n']}" for k, v in coverage.items()})

    df = build_all_features(records)
    jarvis_hull = pd.read_csv(ROOT / "data" / "processed" / "jarvis_ehull_recomputed.csv").set_index("material_id")["ehull_recomputed"]
    is_jarvis = df["source_database"] == "JARVIS"
    df.loc[is_jarvis, "energy_above_hull"] = df.loc[is_jarvis, "material_id"].map(jarvis_hull).values
    print(f"JARVIS hull energies replaced by recomputed values for {is_jarvis.sum()} records "
          f"({df.loc[is_jarvis, 'energy_above_hull'].notna().sum()} defined)")
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
    # already happened in train_eval_lightgbm_v2.py, 67.4% accuracy / 68.0% balanced accuracy with Tier-2 for all sources;
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
    print(f"\nPredicted-only distribution (the {(~labeled_mask).sum():,} unlabeled records, incl. all JARVIS):")
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
    per_source = {"total": df["source_database"].value_counts().to_dict()}
    step = df[df["ordering_final"].isin(["FM", "FiM"])]
    funnel["FM/FiM ordering"] = len(step)
    per_source["FM/FiM ordering"] = step["source_database"].value_counts().to_dict()
    step = step[step["is_noncentrosymmetric"] == True]  # noqa: E712
    funnel["+ noncentrosymmetric"] = len(step)
    per_source["+ noncentrosymmetric"] = step["source_database"].value_counts().to_dict()
    step = step[step["has_heavy_soc"]]
    funnel["+ heavy-SOC element present"] = len(step)
    per_source["+ heavy-SOC element present"] = step["source_database"].value_counts().to_dict()
    print("\nScreening funnel:")
    for k, v in funnel.items():
        print(f"  {k}: {v}")

    step = step.copy()
    step["hull_known"] = step["energy_above_hull"].notna()
    step["hull_ok"] = step["energy_above_hull"].fillna(np.inf) <= 0.1
    funnel["  of which hull known"] = int(step["hull_known"].sum())
    funnel["  of which hull <= 0.1 eV/atom"] = int(step["hull_ok"].sum())
    per_source["  of which hull known"] = step.loc[step["hull_known"], "source_database"].value_counts().to_dict()
    per_source["  of which hull <= 0.1 eV/atom"] = step.loc[step["hull_ok"], "source_database"].value_counts().to_dict()
    if args.hull_hard:
        step = step[step["hull_ok"] | ~step["hull_known"]]
        funnel["+ hull <= 0.1 where known (hard)"] = len(step)
        per_source["+ hull <= 0.1 where known (hard)"] = step["source_database"].value_counts().to_dict()
        step = step.sort_values(
            by=["hull_known", "energy_above_hull", "confidence"], ascending=[False, True, False]
        )
    else:
        step = step.sort_values(
            by=["hull_ok", "confidence", "hull_known"], ascending=[False, False, False]
        )
    print("\nScreening funnel (with hull):")
    for k, v in funnel.items():
        print(f"  {k}: {v}")
    suffix = "_hullhard" if args.hull_hard else ""
    with open(ROOT / "outputs" / f"screening_funnel{suffix}.json", "w") as f:
        json.dump({"funnel": funnel, "per_source": per_source,
                   "labeled": int(labeled_mask.sum()), "unlabeled": int((~labeled_mask).sum())}, f, indent=2)

    out_cols = [
        "material_id", "source_database", "formula", "ordering_final", "ordering_source",
        "confidence", "space_group_number", "crystal_system_bucket", "heavy_soc_elements",
        "energy_above_hull", "density", "volume", "nsites",
    ]
    shortlist = step[out_cols].head(args.top)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(out_dir / f"candidate_shortlist{suffix}.csv", index=False)
    print(f"\nWrote top {len(shortlist)} candidates to {out_dir / f'candidate_shortlist{suffix}.csv'}")
    print("\nTop 15 preview:")
    print(shortlist.head(15).to_string(index=False))

    # Full screened set (not just top-N) for downstream use / re-ranking without retraining.
    step[out_cols].to_csv(out_dir / f"candidate_screened_full{suffix}.csv", index=False)
    print(f"Wrote full screened set ({len(step)} rows) to {out_dir / f'candidate_screened_full{suffix}.csv'}")
    for src in ("MaterialsProject", "JARVIS", "AFLOW"):
        t = step[step.source_database == src].head(100)
        print(f"  top-100 {src}: hull known {t.hull_known.sum()}, hull<=0.1 {t.hull_ok.sum()}")


if __name__ == "__main__":
    main()
