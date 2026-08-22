"""Acquire OQMD data into the harmonized schema, via direct REST calls.

IMPORTANT -- why this does NOT use the `qmpy_rester` library (which the previous attempt used
and which is what OQMD's own docs recommend):

qmpy_rester's get_oqmd_phases() contains a hardcoded interactive prompt at rester.py:77:
        ans = input('Proceed? [Y/n]:')
which fires whenever verbose is left at its default. In any non-interactive script this raises
EOFError and the call dies -- almost certainly the reason the previous attempt at this source
produced zero records. Passing verbose=False suppresses the prompt, but the library was still
observed to hang past 90s on a 3-record query, whereas the same query via direct REST returns
in 0.78s. So: skip the library, call the documented REST endpoint directly.

OQMD REST reports 1,407,395 formation-energy entries available, but DO NOT try to page through
them with limit/offset: OQMD's pagination degrades catastrophically at depth (measured:
offset=0 -> 8.3s, offset=1000 -> 7.7s, offset=10000 -> ReadTimeout after 90s). That is a
server-side OFFSET performance cliff, not a client bug.

Instead this queries by `element_set=<A-B>` (measured 20.4s/query, works reliably) over only
the chemical systems present in our magnetic dataset -- which is all OQMD is needed for anyway
(cross-validating formation energies of materials we already have). Note `composition=<formula>`
was also tried and times out; use element_set.

Caveat on magnetism: the /oqmdapi/formationenergy endpoint does NOT expose a magnetic moment
or magnetic-ordering field (verified against the returned record schema). OQMD's role in this
project per the proposal is "cross-validation of formation energies / stability", not magnetic
labels -- so this acquires composition + stability fields for cross-checking MP/AFLOW values,
and deliberately leaves ordering/total_magnetization as None rather than fabricating them.

Usage:
    python acquire_oqmd.py --limit 5000
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "oqmd_harmonized.json"
BASE = "https://oqmd.org/oqmdapi/formationenergy"
PAGE_SIZE = 500


def fetch_element_set(element_set: str, limit: int = 200, retries: int = 2) -> list[dict]:
    """Query one chemical system. Returns [] on persistent failure rather than aborting the
    whole run -- OQMD's public server is flaky and one bad system shouldn't lose the batch."""
    params = {"filter": f"element_set={element_set}", "limit": limit}
    for attempt in range(retries):
        try:
            r = requests.get(BASE, params=params, timeout=90)
            r.raise_for_status()
            return r.json().get("data") or []
        except Exception as e:
            if attempt == retries - 1:
                print(f"  {element_set}: giving up ({type(e).__name__})")
                return []
            time.sleep(5)
    return []


def _elements_from_formula(formula: str | None) -> list[str] | None:
    if not formula:
        return None
    try:
        from pymatgen.core import Composition
        return [str(el) for el in Composition(formula).elements]
    except Exception:
        return None


def _space_group_number(symbol: str | None) -> int | None:
    """OQMD returns bare Hermann-Mauguin symbols (e.g. 'P21/m'); pymatgen's SpaceGroup lookup
    wants the underscore screw-axis notation ('P2_1/m') for some of them -- verified against the
    36 symbols actually seen in this dataset (35/36 resolved as-is, 'P21/m' needed the
    underscore fix). Falls back to None (not fabricated) for anything neither form resolves."""
    if not symbol:
        return None
    from pymatgen.symmetry.groups import SpaceGroup
    for candidate in (symbol, re.sub(r"(\d)(?=/|$)", r"\1_", symbol)):
        try:
            return SpaceGroup(candidate).int_number
        except ValueError:
            continue
    return None


def to_harmonized(r: dict) -> dict:
    formula = r.get("composition")
    space_group = r.get("spacegroup")
    return {
        "material_id": f"oqmd-{r.get('entry_id')}",
        "source_database": "OQMD",
        "formula": formula,
        "chemical_system": None,
        "elements": _elements_from_formula(formula),
        "num_elements": r.get("ntypes"),
        "space_group": space_group,
        "space_group_number": _space_group_number(space_group),
        "crystal_system": None,
        "nsites": r.get("natoms"),
        "density": None,
        "volume": r.get("volume"),
        "band_gap": r.get("band_gap"),
        "formation_energy": r.get("delta_e"),
        "energy_above_hull": r.get("stability"),
        "ordering": None,           # not exposed by this endpoint -- see module docstring
        "total_magnetization": None,  # not exposed by this endpoint
        "is_magnetic": None,
        "is_metal": (r.get("band_gap") == 0.0) if r.get("band_gap") is not None else None,
        "structure": None,
        "magmom_per_site": None,
        "prototype": r.get("prototype"),
    }


def target_chemical_systems(max_systems: int) -> list[str]:
    """Chemical systems present in our harmonized magnetic dataset, most common first."""
    from collections import Counter
    with open(ROOT / "data" / "processed" / "harmonized_v2.json") as f:
        records = json.load(f)
    counter = Counter()
    for r in records:
        elements = r.get("elements")
        if elements:
            counter["-".join(sorted(elements))] += 1
    return [sys for sys, _ in counter.most_common(max_systems)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-systems", type=int, default=200,
                        help="how many chemical systems to cross-validate (~20s each)")
    args = parser.parse_args()

    systems = target_chemical_systems(args.max_systems)
    print(f"Querying {len(systems)} chemical systems from our magnetic dataset...")

    collected = []
    # Resume support: OQMD is slow enough that partial progress is worth keeping.
    if OUT.exists():
        with open(OUT) as f:
            collected = json.load(f)
        done = {r.get("_element_set") for r in collected}
        systems = [s for s in systems if s not in done]
        print(f"  resuming: {len(collected)} records already saved, {len(systems)} systems left")

    for i, sysname in enumerate(systems, 1):
        data = fetch_element_set(sysname)
        for r in data:
            h = to_harmonized(r)
            h["_element_set"] = sysname
            collected.append(h)
        print(f"[{i}/{len(systems)}] {sysname}: {len(data)} records (total {len(collected)})")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w") as f:  # checkpoint every system
            json.dump(collected, f)
        time.sleep(0.5)  # be polite to a public academic API

    print(f"Wrote {len(collected)} OQMD records to {OUT}")


if __name__ == "__main__":
    main()
