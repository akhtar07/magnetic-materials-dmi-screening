"""Consolidate the reusable prior-run data into one documented, provenance-tracked dataset.

Input (copied read-only from the original repo, see docs/dataset_specification.md):
    data/raw/master_dataset_v4.json -- Materials Project + JARVIS-DFT + AFLOW + MAGNDATA, 59,681 records

NOTE: data/raw/magndata_dataset_labeled.json is NOT merged here even though it's on disk --
it turned out to be a duplicate of the 79 MAGNDATA-tagged records already inside master_dataset_v4.json
(same material_ids, same structures). master_dataset_v4.json also contains 64 stray untagged
(database=None) records that are further duplicates of the same 79 MAGNDATA entries with a different
key shape -- those are dropped. Discovered by inspecting record counts; see roadmap for detail.

Output:
    data/processed/harmonized_v2.json -- one list of records conforming to docs/dataset_specification.md
    data/processed/harmonized_v2_summary.json -- per-source counts for a quick sanity check
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FIELDS = [
    "material_id", "database", "formula", "chemical_system", "elements", "num_elements",
    "space_group", "space_group_number", "crystal_system", "nsites", "density", "volume",
    "band_gap", "formation_energy", "energy_above_hull", "ordering", "total_magnetization",
    "is_magnetic", "is_metal", "structure",
]


def load_source(path: Path) -> list[dict]:
    with open(path) as f:
        records = json.load(f)
    out = []
    for r in records:
        if r.get("database") is None:
            continue  # stray untagged duplicate, see module docstring
        rec = {field: r.get(field) for field in FIELDS}
        rec["source_database"] = rec.pop("database")
        structure = rec["structure"]
        magmoms = None
        if structure and "sites" in structure:
            magmoms = [
                site.get("properties", {}).get("magmom")
                for site in structure["sites"]
            ]
        rec["magmom_per_site"] = magmoms
        out.append(rec)
    return out


def main():
    harmonized = load_source(RAW / "master_dataset_v4.json")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "harmonized_v2.json"
    with open(out_path, "w") as f:
        json.dump(harmonized, f)

    summary = {}
    for rec in harmonized:
        db = rec["source_database"]
        summary[db] = summary.get(db, 0) + 1
    summary["_total"] = len(harmonized)
    summary["_with_structure"] = sum(1 for r in harmonized if r["structure"] is not None)

    with open(PROCESSED / "harmonized_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(harmonized)} records to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
