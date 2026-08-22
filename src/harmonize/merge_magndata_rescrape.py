"""Merge the new MAGNDATA rescrape (data/processed/magndata_harmonized.json, ~1,900+ records,
see src/acquisition/acquire_magndata.py) into harmonized_v2.json, replacing the old 79-record
MAGNDATA subset that build_harmonized_dataset.py carried over from master_dataset_v4.json.

Run build_harmonized_dataset.py first if harmonized_v2.json doesn't exist yet.

Usage:
    python merge_magndata_rescrape.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# Schema fields carried over from build_harmonized_dataset.py's FIELDS list, plus
# source_database/magmom_per_site which it also attaches. acquire_magndata.py's output already
# matches this shape except for the extra `source_url` field, which is dropped here.
FIELDS = [
    "material_id", "source_database", "formula", "chemical_system", "elements", "num_elements",
    "space_group", "space_group_number", "crystal_system", "nsites", "density", "volume",
    "band_gap", "formation_energy", "energy_above_hull", "ordering", "total_magnetization",
    "is_magnetic", "is_metal", "structure", "magmom_per_site",
]


def main():
    harmonized_path = PROCESSED / "harmonized_v2.json"
    with open(harmonized_path) as f:
        harmonized = json.load(f)

    old_magndata = [r for r in harmonized if r["source_database"] == "MAGNDATA"]
    kept = [r for r in harmonized if r["source_database"] != "MAGNDATA"]
    print(f"Dropping {len(old_magndata)} old MAGNDATA records (kept {len(kept)} from other sources)")

    with open(PROCESSED / "magndata_harmonized.json") as f:
        new_magndata_raw = json.load(f)
    new_magndata = [{field: r.get(field) for field in FIELDS} for r in new_magndata_raw]
    print(f"Adding {len(new_magndata)} rescraped MAGNDATA records")

    merged = kept + new_magndata
    with open(harmonized_path, "w") as f:
        json.dump(merged, f)

    summary = {}
    for rec in merged:
        db = rec["source_database"]
        summary[db] = summary.get(db, 0) + 1
    summary["_total"] = len(merged)
    summary["_with_structure"] = sum(1 for r in merged if r["structure"] is not None)
    with open(PROCESSED / "harmonized_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(merged)} records to {harmonized_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
