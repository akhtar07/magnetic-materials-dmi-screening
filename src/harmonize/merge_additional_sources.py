"""Merge OQMD, NOMAD, and 2DMatPedia into harmonized_v2.json.

These three were acquired (src/acquisition/acquire_oqmd.py, acquire_nomad.py,
acquire_2dmatpedia.py) in an earlier session but never actually merged in -- the harmonized
dataset and every downstream classifier/screening number in this project have been computed
without them until now. Discovered by checking data/processed/ for their output files directly,
not assumed.

None of the three carry an `ordering` label or populated `magmom_per_site` (verified by
inspecting the actual field values, not just key presence -- OQMD/NOMAD don't expose magnetism
fields at all per their own docstrings; 2DMatPedia has `total_magnetization` and `is_magnetic`
but never populates per-site magmom vectors). Per the same principle already applied to JARVIS
(dataset_specification.md: "the field the Phase 1 bias classifier corrects" -- do not hand-derive
an ordering label from a net-moment threshold, since that can't distinguish FM from FiM or AFM
from NM), these are merged in as unlabeled (ordering=None) and left for screen_candidates.py's
classifier to predict, not heuristically labeled here.

OQMD-specific note: the acquisition script queries by overlapping element_set, so the raw output
has heavy duplication by material_id (7,800 raw rows, 816 unique ids) -- deduplicated here before
merging, keeping first occurrence.

2DMatPedia is the only 2D-materials source in the project (dataset_specification.md flagged "2D
and 3D screening is 3D-only in practice until C2DB/2DMatPedia land" as an open gap) -- this closes
that gap for 2DMatPedia specifically (C2DB remains unacquired).

Usage:
    python merge_additional_sources.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

FIELDS = [
    "material_id", "source_database", "formula", "chemical_system", "elements", "num_elements",
    "space_group", "space_group_number", "crystal_system", "nsites", "density", "volume",
    "band_gap", "formation_energy", "energy_above_hull", "ordering", "total_magnetization",
    "is_magnetic", "is_metal", "structure", "magmom_per_site",
]

SOURCES = {
    "OQMD": PROCESSED / "oqmd_harmonized.json",
    "NOMAD": PROCESSED / "nomad_harmonized.json",
    "2DMatPedia": PROCESSED / "2dmatpedia_harmonized.json",
}


def load_and_normalize(name: str, path: Path) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    if name == "OQMD":
        seen = set()
        deduped = []
        for r in raw:
            mid = r.get("material_id")
            if mid in seen:
                continue
            seen.add(mid)
            deduped.append(r)
        print(f"  OQMD dedup: {len(raw)} raw -> {len(deduped)} unique material_id")
        raw = deduped
    out = [{field: r.get(field) for field in FIELDS} for r in raw]
    for r in out:
        r["source_database"] = name  # normalize (2DMatPedia/NOMAD already say this, but be explicit)
        r["ordering"] = None  # never fabricate a label these sources don't provide, see docstring
    return out


def main():
    harmonized_path = PROCESSED / "harmonized_v2.json"
    with open(harmonized_path) as f:
        harmonized = json.load(f)
    existing_ids = {(r["source_database"], r["material_id"]) for r in harmonized}
    print(f"Starting from {len(harmonized)} records")

    added_total = 0
    for name, path in SOURCES.items():
        if not path.exists():
            print(f"  {name}: {path} not found, skipping")
            continue
        new_recs = load_and_normalize(name, path)
        # guard against accidental re-run double-adding
        new_recs = [r for r in new_recs if (r["source_database"], r["material_id"]) not in existing_ids]
        harmonized.extend(new_recs)
        existing_ids.update((r["source_database"], r["material_id"]) for r in new_recs)
        added_total += len(new_recs)
        print(f"  {name}: added {len(new_recs)} records")

    with open(harmonized_path, "w") as f:
        json.dump(harmonized, f)

    summary = {}
    for rec in harmonized:
        db = rec["source_database"]
        summary[db] = summary.get(db, 0) + 1
    summary["_total"] = len(harmonized)
    summary["_with_structure"] = sum(1 for r in harmonized if r["structure"] is not None)
    with open(PROCESSED / "harmonized_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAdded {added_total} records total. Wrote {len(harmonized)} records to {harmonized_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
