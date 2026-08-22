"""Acquire 2DMatPedia (2D materials) into the harmonized schema.

IMPORTANT -- why this uses the Figshare mirror rather than the official site:
www.2dmatpedia.org returns HTTP 503 (Service Unavailable) on every endpoint including
/download (verified across http/https and both www and bare domains). That is a server-side
outage on their end, not a client bug -- and is the most likely reason the previous attempt
at this source produced zero records. The authors publish the same dataset on Figshare
(DOI 10.6084/m9.figshare.7699910), which is up and serves the full 29MB JSON.

Download (verified md5 4b201c32a308010d5024f44c3d6709fa):
    curl -L https://ndownloader.figshare.com/files/14399918 -o data/external/2dmatpedia.json

Format note: the file is JSON *Lines* (one JSON object per line), not a single JSON array --
json.load() on the whole file fails. 6,351 records total, 1,862 with |total_magnetization| > 0.01.

Usage:
    python acquire_2dmatpedia.py [--magnetic-only]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "external" / "2dmatpedia.json"
OUT = ROOT / "data" / "processed" / "2dmatpedia_harmonized.json"

MAG_THRESHOLD = 0.01  # |total_magnetization| in mu_B below this is treated as non-magnetic


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_harmonized(r: dict) -> dict:
    sg = r.get("spacegroup") or {}
    return {
        "material_id": r.get("material_id"),
        "source_database": "2DMatPedia",
        "formula": r.get("formula_pretty"),
        "chemical_system": r.get("chemsys"),
        "elements": r.get("elements"),
        "num_elements": r.get("nelements"),
        "space_group": sg.get("symbol"),
        "space_group_number": sg.get("number"),
        "crystal_system": sg.get("crystal_system"),
        "nsites": None,
        "density": None,
        "volume": None,
        "band_gap": r.get("bandgap"),
        "formation_energy": None,
        "energy_above_hull": None,
        # 2DMatPedia gives no magnetic-ordering label (FM/AFM/...), only a magnitude.
        # Leave ordering None rather than inventing a label from magnitude alone -- the
        # Phase 1 bias classifier is what assigns/corrects ordering.
        "ordering": None,
        "total_magnetization": r.get("total_magnetization"),
        "is_magnetic": abs(r.get("total_magnetization") or 0) > MAG_THRESHOLD,
        "is_metal": (r.get("bandgap") == 0.0) if r.get("bandgap") is not None else None,
        "structure": r.get("structure"),
        "magmom_per_site": None,
        # 2D-specific fields -- the reason the proposal wanted this source at all
        "dimensionality": "2D",
        "exfoliation_energy_per_atom": r.get("exfoliation_energy_per_atom"),
        "decomposition_energy_per_atom": r.get("decomposition_energy_per_atom"),
        "discovery_process": r.get("discovery_process"),
        "source_id": r.get("source_id"),  # parent 3D material (e.g. mp-30033) for top-down entries
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--magnetic-only", action="store_true",
                        help="keep only records with |total_magnetization| > %.2f" % MAG_THRESHOLD)
    args = parser.parse_args()

    if not SRC.exists():
        raise SystemExit(f"{SRC} not found -- see download command in this file's docstring")

    records = load_jsonl(SRC)
    print(f"Loaded {len(records)} 2DMatPedia records")

    harmonized = [to_harmonized(r) for r in records]
    if args.magnetic_only:
        harmonized = [h for h in harmonized if h["is_magnetic"]]
        print(f"Filtered to {len(harmonized)} magnetic records")

    with_struct = sum(1 for h in harmonized if h["structure"] is not None)
    with_exf = sum(1 for h in harmonized if h["exfoliation_energy_per_atom"] is not None)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(harmonized, f)

    print(f"Wrote {len(harmonized)} records to {OUT}")
    print(f"  with structure: {with_struct}")
    print(f"  with exfoliation energy: {with_exf}")


if __name__ == "__main__":
    main()
