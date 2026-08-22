"""Acquire NOMAD entries for magnetic-element systems into the harmonized schema.

Why the previous attempt returned nothing: NOMAD's v1 API requires a POST to
/entries/query with a JSON body -- a plain GET (which the old client used) does not
return entries. The /info endpoint responds to GET, which makes a naive connectivity
check look successful while the actual search silently yields nothing.

Also note: passing a `required.include` block with guessed field paths returns HTTP 422
(verified -- e.g. 'results.properties.electronic.band_structure_electronic.band_gap' is
rejected). Field paths must match NOMAD's actual schema; the ones used below were read off
a live entry rather than guessed. Omitting `required` entirely returns full entries and
works fine, at the cost of larger responses.

Scope note: NOMAD's role in the proposal is "supplementary raw DFT data not present in
curated DBs". A query for entries containing any of Fe/Co/Ni/Mn/Cr matches ~3.44M entries,
which is far more than this project needs and mostly duplicates MP/AFLOW/JARVIS. This script
therefore pulls a bounded sample for cross-referencing, not the whole set.

IMPORTANT -- magnetic labels: NOMAD's results.properties for these entries expose
geometry_optimization / structures / available_properties, but NOT a magnetic ordering or
total magnetization field in the general case. So `ordering` and `total_magnetization` are
left None here rather than fabricated. Treat NOMAD as a structural/provenance supplement.

Usage:
    python acquire_nomad.py --max-entries 2000
"""
import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "nomad_harmonized.json"
URL = "https://nomad-lab.eu/prod/v1/api/v1/entries/query"
MAGNETIC_ELEMENTS = ["Fe", "Co", "Ni", "Mn", "Cr"]
PAGE_SIZE = 100


def to_harmonized(e: dict) -> dict:
    results = e.get("results") or {}
    material = results.get("material") or {}
    symmetry = material.get("symmetry") or {}
    return {
        "material_id": f"nomad-{e.get('entry_id')}",
        "source_database": "NOMAD",
        "formula": material.get("chemical_formula_reduced"),
        "chemical_system": "-".join(sorted(material.get("elements") or [])) or None,
        "elements": material.get("elements"),
        "num_elements": material.get("n_elements"),
        "space_group": symmetry.get("space_group_symbol"),
        "space_group_number": symmetry.get("space_group_number"),
        "crystal_system": symmetry.get("crystal_system"),
        "nsites": None,
        "density": None,
        "volume": None,
        "band_gap": None,
        "formation_energy": None,
        "energy_above_hull": None,
        "ordering": None,             # not exposed -- see module docstring
        "total_magnetization": None,  # not exposed -- see module docstring
        "is_magnetic": None,
        "is_metal": None,
        "structure": None,
        "magmom_per_site": None,
        "structural_type": material.get("structural_type"),
        "nomad_material_id": material.get("material_id"),
        "external_db": e.get("external_db"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-entries", type=int, default=2000)
    args = parser.parse_args()

    collected = []
    page_after = None

    while len(collected) < args.max_entries:
        page = {"page_size": min(PAGE_SIZE, args.max_entries - len(collected))}
        if page_after:
            page["page_after_value"] = page_after
        body = {"query": {"results.material.elements": {"any": MAGNETIC_ELEMENTS}},
                "pagination": page}
        try:
            r = requests.post(URL, json=body, timeout=90)
            r.raise_for_status()
        except Exception as e:
            print(f"request failed ({type(e).__name__}); stopping with {len(collected)} collected")
            break

        payload = r.json()
        data = payload.get("data") or []
        if not data:
            break
        collected.extend(to_harmonized(e) for e in data)
        pagination = payload.get("pagination") or {}
        page_after = pagination.get("next_page_after_value")
        print(f"  {len(collected)} / {args.max_entries} (total available: {pagination.get('total')})")
        if not page_after:
            break
        time.sleep(0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(collected, f)
    print(f"Wrote {len(collected)} NOMAD records to {OUT}")


if __name__ == "__main__":
    main()
