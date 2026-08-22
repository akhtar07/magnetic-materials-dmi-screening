"""Backfill full atomic `structure` for shortlisted candidates from screen_candidates.py.

Only MAGNDATA carries `structure` in the harmonized schema (see build_features_v2.py docstring);
MP/JARVIS/AFLOW never had it acquired at all (verified against the raw master_dataset_v4.json --
the field is simply absent from those source records, not null-but-present). Fetching it for the
full ~60k database would be wasteful; dataset_specification.md already anticipated backfilling
"for MP/JARVIS/AFLOW records that get shortlisted" -- this does exactly that, for the output of
screen_candidates.py, via each source's own public data access method:

  - JARVIS: jarvis-tools' bulk `dft_3d` figshare dataset (no API key -- public bulk download,
    cached locally once). Matched by jid, converted via Atoms.pymatgen_converter().
  - AFLOW: the public AFLUX REST API (no API key), queried per-candidate by auid to get the
    `aurl` entry-directory path, then CONTCAR.relax fetched directly and parsed as POSCAR. Same
    "verify against the live service, don't guess the URL scheme" approach used for MAGNDATA --
    confirmed working against a real candidate before writing this loop (see conversation).
  - Materials Project: requires an API key (MPRester) -- none is present in this environment
    (checked env vars, ~/.pmgrc.yaml, shell profiles; found nothing). Reads MP_API_KEY from the
    environment if set; otherwise these candidates are recorded as blocked, not silently skipped.

Usage:
    python backfill_structures.py [--top-per-source N]
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[2]
HEADERS = {"User-Agent": "Mozilla/5.0 (research data acquisition; magnetic-materials-ai project)"}


def backfill_aflow(auids: list[str]) -> dict[str, dict]:
    out = {}
    for auid in auids:
        try:
            q = f"http://aflow.org/API/aflux/?auid('{auid}')"
            r = requests.get(q, timeout=30)
            r.raise_for_status()
            hits = r.json()
            if not hits:
                out[auid] = {"error": "no AFLUX hit for auid"}
                continue
            aurl = hits[0]["aurl"]
            base = "http://" + aurl.replace("aflowlib.duke.edu:", "aflowlib.duke.edu/")
            r2 = requests.get(base + "/CONTCAR.relax", timeout=30)
            r2.raise_for_status()
            structure = Structure.from_str(r2.text, fmt="poscar")
            out[auid] = {"structure": structure.as_dict()}
        except Exception as e:
            out[auid] = {"error": f"{type(e).__name__}: {e}"}
        time.sleep(0.3)
    return out


def backfill_jarvis(jids: list[str]) -> dict[str, dict]:
    from jarvis.core.atoms import Atoms
    from jarvis.db.figshare import data

    print("Loading cached JARVIS dft_3d dataset (downloads once, ~large first time)...")
    dft_3d = data(dataset="dft_3d", store_dir=str(ROOT / "data" / "raw" / "jarvis_cache"))
    by_jid = {e["jid"]: e for e in dft_3d}
    print(f"JARVIS dft_3d loaded: {len(by_jid)} entries")

    out = {}
    for jid in jids:
        entry = by_jid.get(jid)
        if entry is None:
            out[jid] = {"error": "jid not found in dft_3d dataset"}
            continue
        try:
            atoms = Atoms.from_dict(entry["atoms"])
            structure = atoms.pymatgen_converter()
            out[jid] = {"structure": structure.as_dict()}
        except Exception as e:
            out[jid] = {"error": f"{type(e).__name__}: {e}"}
    return out


def backfill_mp(mp_ids: list[str]) -> dict[str, dict]:
    import os
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        return {mid: {"error": "blocked: no MP_API_KEY in environment"} for mid in mp_ids}

    from mp_api.client import MPRester
    out = {}
    with MPRester(api_key) as mpr:
        for mid in mp_ids:
            try:
                structure = mpr.get_structure_by_material_id(mid)
                out[mid] = {"structure": structure.as_dict()}
            except Exception as e:
                out[mid] = {"error": f"{type(e).__name__}: {e}"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-per-source", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "outputs" / "candidate_screened_full.csv")
    print(f"Loaded {len(df)} screened candidates")

    results_path = ROOT / "data" / "processed" / "candidate_structures.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    aflow_ids = df[df.source_database == "AFLOW"]["material_id"].head(args.top_per_source).tolist()
    jarvis_ids = df[df.source_database == "JARVIS"]["material_id"].head(args.top_per_source).tolist()
    mp_ids = df[df.source_database == "MaterialsProject"]["material_id"].head(args.top_per_source).tolist()

    print(f"\nBackfilling AFLOW ({len(aflow_ids)} candidates, no key needed)...")
    aflow_results = backfill_aflow(aflow_ids)
    results.update(aflow_results)
    n_ok = sum(1 for v in aflow_results.values() if "structure" in v)
    print(f"  AFLOW: {n_ok}/{len(aflow_ids)} succeeded")

    print(f"\nBackfilling JARVIS ({len(jarvis_ids)} candidates, no key needed)...")
    jarvis_results = backfill_jarvis(jarvis_ids)
    results.update(jarvis_results)
    n_ok = sum(1 for v in jarvis_results.values() if "structure" in v)
    print(f"  JARVIS: {n_ok}/{len(jarvis_ids)} succeeded")

    print(f"\nBackfilling Materials Project ({len(mp_ids)} candidates, needs MP_API_KEY)...")
    mp_results = backfill_mp(mp_ids)
    results.update(mp_results)
    n_ok = sum(1 for v in mp_results.values() if "structure" in v)
    print(f"  MaterialsProject: {n_ok}/{len(mp_ids)} succeeded")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results))

    total_ok = sum(1 for v in results.values() if "structure" in v)
    print(f"\nWrote {len(results)} results ({total_ok} with structure) to {results_path}")


if __name__ == "__main__":
    main()
