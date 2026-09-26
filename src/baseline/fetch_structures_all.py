"""Fetch atomic structures for EVERY harmonized record so the Tier-2 local-geometry features of
build_features_v2.py can be computed for all sources, not only MAGNDATA/2DMatPedia.

Why (revision, 2026-09-20): in the original v2 pipeline the Tier-2 features (magnetic-magnetic
nearest-neighbour distance, magnetic coordination, ...) were populated for the 3% of records that
carried an embedded `structure` (MAGNDATA, 2DMatPedia) and NaN for everything else. Because every
MAGNDATA row is AFM/FiM-heavy and every MP row had NaN, "Tier-2 is missing" was itself a proxy for
the data source and therefore for the label distribution -- a leakage-like artefact, not a
structural signal. Filling the features for all sources removes that confound.

Sources and how the structure is obtained (all persisted under data/processed/structure_store/,
gitignored, one gzip JSON per source mapping material_id -> pymatgen Structure dict or an error):
  MaterialsProject : mp_api MPRester summary search in chunks (needs MP_API_KEY; read from the
                     environment only, never printed).
  JARVIS           : the raw dft_3d zip already cached in data/raw/jarvis_cache (`atoms` field).
  AFLOW            : AFLUX lookup of auid -> aurl + species, then CONTCAR.relax. Many AFLOW CONTCARs
                     are VASP-4 format without an element line; pymatgen then silently assigns
                     placeholder species (H, He, Li, ...), which corrupted 490 of the first 2,076
                     AFLOW structures fetched (found 2026-09-20 via a species-vs-composition check).
                     The AFLUX `species` list (alphabetical, the CONTCAR order) is now passed as
                     default_names, and every stored structure is validated against the record's
                     element set before use.
  OQMD             : oqmdapi/formationenergy?entry_id=N&fields=unit_cell,sites (public, flaky).
  NOMAD            : entries/<entry_id>/archive/query for results.properties.structures.
2DMatPedia and MAGNDATA already embed their structure in harmonized_v2.json and are not fetched.

Every fetch is resumable: existing entries in the per-source store are kept, only missing ids are
requested. Failures are stored as {"error": ...} so coverage is reported honestly by
build_features_v2.py rather than silently treated as "no structure".

Usage:
    MP_API_KEY=... python fetch_structures_all.py [--sources mp jarvis aflow oqmd nomad] [--limit N]
"""
import argparse
import gzip
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data" / "processed" / "structure_store"
HEADERS = {"User-Agent": "Mozilla/5.0 (research data acquisition; magnetic-materials-ai project)"}
SOURCE_KEY = {"mp": "MaterialsProject", "jarvis": "JARVIS", "aflow": "AFLOW", "oqmd": "OQMD", "nomad": "NOMAD"}


def load_store(name: str) -> dict:
    p = STORE / f"{name}.json.gz"
    if p.exists():
        with gzip.open(p, "rt") as f:
            return json.load(f)
    return {}


def save_store(name: str, store: dict) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    tmp = STORE / f"{name}.json.gz.tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(store, f)
    tmp.replace(STORE / f"{name}.json.gz")


def report(name: str, store: dict, wanted: list[str]) -> None:
    ok = sum(1 for i in wanted if "structure" in store.get(i, {}))
    err = sum(1 for i in wanted if "error" in store.get(i, {}))
    print(f"  {name}: {ok}/{len(wanted)} structures, {err} errors, {len(wanted) - ok - err} not attempted")


# ----------------------------------------------------------------------------------------------
def fetch_mp(ids: list[str], store: dict, chunk: int = 500) -> None:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit("MP_API_KEY not set in the environment (see .mp_api_key.local; never print it)")
    from mp_api.client import MPRester

    todo = [i for i in ids if i not in store]
    print(f"  MP: {len(todo)} ids to fetch in chunks of {chunk}")
    with MPRester(api_key, mute_progress_bars=True) as mpr:
        for k in range(0, len(todo), chunk):
            batch = todo[k:k + chunk]
            for attempt in range(3):
                try:
                    docs = mpr.materials.summary.search(material_ids=batch, fields=["material_id", "structure"])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"    chunk {k}: {type(e).__name__}, retrying"); time.sleep(10)
            got = {str(d.material_id): d.structure for d in docs}
            for i in batch:
                if i in got:
                    store[i] = {"structure": got[i].as_dict()}
                else:
                    # the id may have been deprecated/merged upstream since the dataset snapshot
                    store[i] = {"error": "not returned by summary search (deprecated id?)"}
            save_store("mp", store)
            print(f"    {min(k + chunk, len(todo))}/{len(todo)} ({len(got)} returned in this chunk)")
            time.sleep(0.5)


def fetch_jarvis(ids: list[str], store: dict) -> None:
    zpath = ROOT / "data" / "raw" / "jarvis_cache" / "jdft_3d-9-24-2025.json.zip"
    with zipfile.ZipFile(zpath) as z:
        raw = json.loads(z.read(z.namelist()[0]))
    by_jid = {e["jid"]: e for e in raw}
    print(f"  JARVIS: {len(by_jid)} entries in raw zip")
    for jid in ids:
        if jid in store:
            continue
        e = by_jid.get(jid)
        if e is None:
            store[jid] = {"error": "jid not in raw zip"}
            continue
        a = e["atoms"]
        try:
            s = Structure(Lattice(np.array(a["lattice_mat"], dtype=float)), a["elements"],
                          np.array(a["coords"], dtype=float), coords_are_cartesian=bool(a.get("cartesian", True)))
            store[jid] = {"structure": s.as_dict()}
        except Exception as ex:
            store[jid] = {"error": f"{type(ex).__name__}: {ex}"}
    save_store("jarvis", store)


def fetch_aflow(ids: list[str], store: dict) -> None:
    todo = [i for i in ids if i not in store]
    print(f"  AFLOW: {len(todo)} auids to fetch")
    for n, auid in enumerate(todo, 1):
        try:
            r = requests.get(f"http://aflow.org/API/aflux/?auid('{auid}'),species,composition", timeout=60, headers=HEADERS)
            r.raise_for_status()
            hits = r.json()
            if not hits:
                store[auid] = {"error": "no AFLUX hit for auid"}
            else:
                aurl = hits[0]["aurl"]
                species = hits[0].get("species") or None
                base = "http://" + aurl.replace("aflowlib.duke.edu:", "aflowlib.duke.edu/")
                r2 = requests.get(base + "/CONTCAR.relax", timeout=60, headers=HEADERS)
                r2.raise_for_status()
                lines = r2.text.splitlines()
                has_symbols = all(tok.isalpha() for tok in lines[5].split())
                if has_symbols:
                    st = Structure.from_str(r2.text, fmt="poscar")
                else:
                    # VASP-4 CONTCAR: no element line. AFLUX `species`/`composition` are in the
                    # CONTCAR's species order; verify the atom counts are proportional to
                    # `composition` before trusting that order.
                    counts = [int(x) for x in lines[5].split()]
                    comp = hits[0].get("composition")
                    if not species or not comp or len(counts) != len(species):
                        raise ValueError(f"cannot assign species: counts={counts} species={species}")
                    ratios = {c / k for c, k in zip(counts, comp)}
                    if len(ratios) != 1:
                        raise ValueError(f"CONTCAR counts {counts} not proportional to AFLUX composition {comp}")
                    st = Poscar.from_str(r2.text, default_names=species).structure
                got = {sp.symbol for sp in st.composition}
                if species and got != set(species):
                    raise ValueError(f"species {sorted(got)} != AFLUX species {sorted(species)}")
                store[auid] = {"structure": st.as_dict()}
        except Exception as e:
            store[auid] = {"error": f"{type(e).__name__}: {e}"}
        if n % 50 == 0 or n == len(todo):
            save_store("aflow", store)
            print(f"    {n}/{len(todo)}")
        time.sleep(0.3)


def _oqmd_structure(rec: dict) -> Structure:
    # unit_cell: 3x3 list; sites: ["Fe @ 0 0 0", ...] with fractional coordinates
    species, frac = [], []
    for s in rec["sites"]:
        el, _, xyz = s.partition("@")
        species.append(el.strip())
        frac.append([float(v) for v in xyz.split()])
    return Structure(Lattice(np.array(rec["unit_cell"], dtype=float)), species, frac)


def fetch_oqmd(ids: list[str], store: dict) -> None:
    todo = [i for i in ids if i not in store]
    print(f"  OQMD: {len(todo)} entries to fetch")
    consecutive_fail = 0
    for n, mid in enumerate(todo, 1):
        if consecutive_fail >= 8:
            # oqmd.org answers nothing for minutes at a time (2026-09-20: every probe timed out);
            # stop burning hours on it and leave the rest "not attempted" for a later resume.
            print(f"    OQMD unreachable ({consecutive_fail} consecutive failures) -- stopping this source")
            break
        entry_id = mid.removeprefix("oqmd-")
        err = None
        for attempt in range(2):
            try:
                r = requests.get("https://oqmd.org/oqmdapi/formationenergy",
                                 params={"entry_id": entry_id, "limit": 1, "fields": "entry_id,unit_cell,sites"},
                                 timeout=45, headers=HEADERS)
                r.raise_for_status()
                data = r.json().get("data") or []
                data = [d for d in data if str(d.get("entry_id")) == entry_id]
                if not data:
                    err = "entry_id not returned"
                    break
                store[mid] = {"structure": _oqmd_structure(data[0]).as_dict()}
                err = None
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep(5)
        if err:
            store[mid] = {"error": err}
            consecutive_fail += 1
        else:
            consecutive_fail = 0
        if n % 25 == 0 or n == len(todo):
            save_store("oqmd", store)
            print(f"    {n}/{len(todo)}")
        time.sleep(0.5)


def fetch_nomad(ids: list[str], store: dict) -> None:
    todo = [i for i in ids if i not in store]
    print(f"  NOMAD: {len(todo)} entries to fetch")
    # results.properties.structures is absent for some (older-parser) entries; fall back to the
    # last system of the last run section, which is what that field is derived from anyway.
    body = {"required": {"results": {"properties": {"structures": {"structure_original": {
        "lattice_vectors": "*", "cartesian_site_positions": "*", "species_at_sites": "*"}}}},
        "run[-1]": {"system[-1]": {"atoms": {"lattice_vectors": "*", "positions": "*", "labels": "*"}}}}}
    for n, mid in enumerate(todo, 1):
        entry_id = mid.removeprefix("nomad-")
        try:
            r = requests.post(f"https://nomad-lab.eu/prod/v1/api/v1/entries/{entry_id}/archive/query",
                              json=body, timeout=90, headers=HEADERS)
            r.raise_for_status()
            arch = r.json()["data"]["archive"]
            so = (arch.get("results", {}).get("properties", {}).get("structures") or {}).get("structure_original")
            if so:
                lat, pos, species = so["lattice_vectors"], so["cartesian_site_positions"], so["species_at_sites"]
            else:
                atoms = arch["run"][-1]["system"][-1]["atoms"]
                lat, pos, species = atoms["lattice_vectors"], atoms["positions"], atoms["labels"]
            lat = np.array(lat, dtype=float) * 1e10   # NOMAD stores SI metres -> Angstrom
            pos = np.array(pos, dtype=float)
            # Three entries (found 2026-09-20) carry FRACTIONAL coordinates in the "cartesian"
            # positions field (values in [0, 1] that would be 0.5 m if taken as metres, giving
            # fractional coordinates ~1e9 that hang pymatgen's neighbour search). Atoms cannot sit
            # 0.5 m apart in an Angstrom-scale cell, so raw values inside [-1.5, 1.5] are
            # unambiguously fractional; everything else is metres.
            frac_like = np.abs(pos).max() <= 1.5
            if frac_like:
                struct = Structure(Lattice(lat), species, pos, coords_are_cartesian=False)
            else:
                struct = Structure(Lattice(lat), species, pos * 1e10, coords_are_cartesian=True)
            if np.abs(struct.frac_coords).max() > 10:
                raise ValueError(f"fractional coordinates up to {np.abs(struct.frac_coords).max():.3g}: "
                                 f"positions inconsistent with lattice")
            store[mid] = {"structure": struct.as_dict()}
        except Exception as e:
            store[mid] = {"error": f"{type(e).__name__}: {e}"}
        if n % 25 == 0 or n == len(todo):
            save_store("nomad", store)
            print(f"    {n}/{len(todo)}")
        time.sleep(0.3)


FETCHERS = {"mp": fetch_mp, "jarvis": fetch_jarvis, "aflow": fetch_aflow, "oqmd": fetch_oqmd, "nomad": fetch_nomad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(FETCHERS), choices=list(FETCHERS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N ids per source (testing)")
    ap.add_argument("--retry-errors", action="store_true", help="re-request ids whose stored entry is an error")
    args = ap.parse_args()

    records = json.load(open(ROOT / "data" / "processed" / "harmonized_v2.json"))
    for name in args.sources:
        ids = [r["material_id"] for r in records if r["source_database"] == SOURCE_KEY[name]]
        if args.limit:
            ids = ids[:args.limit]
        store = load_store(name)
        if args.retry_errors:
            store = {k: v for k, v in store.items() if "structure" in v}
        elements = {r["material_id"]: set(r.get("elements") or []) for r in records
                    if r["source_database"] == SOURCE_KEY[name]}
        wrong = [k for k, v in store.items() if "structure" in v and elements.get(k)
                 and {s["species"][0]["element"] for s in v["structure"]["sites"]} != elements[k]]
        if wrong:
            print(f"{name}: {len(wrong)} stored structures whose species disagree with the record's "
                  f"elements -- dropped and refetched")
            for k in wrong:
                del store[k]
        print(f"{name}: {len(ids)} records, {len(store)} already in store")
        FETCHERS[name](ids, store)
        report(name, store, ids)


if __name__ == "__main__":
    main()
