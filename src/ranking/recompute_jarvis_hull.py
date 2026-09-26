"""Recompute energy-above-hull for the JARVIS-DFT candidates from JARVIS's own formation energies.

Why: the `ehull` field of the JARVIS dft_3d snapshot used here (jdft_3d-9-24-2025) is not a usable
hull distance -- its median over all 93,902 entries is 1.72 eV/atom, TbMnSi (a known stable
compound, formation energy -0.45 eV/atom) carries ehull = 2.45, while BaFeHg4 with a POSITIVE
formation energy (+0.36 eV/atom) carries ehull = 0. Found 2026-09-20 while making the hull a hard
screening filter: every JARVIS record that had passed the "hull <= 0.1" ranking key was one of
those inconsistent zeros. The harmonized dataset copied this field verbatim as
`energy_above_hull`, so the "92.6% hull coverage (MP + JARVIS)" of the screening funnel was really
MP-only as far as trustworthy values go.

Fix: build the convex hull per chemical system from `formation_energy_peratom` (OptB88vdW, all
JARVIS bulk entries whose elements are a subset of the candidate's system; elements sit at 0 by
construction) with pymatgen's PhaseDiagram, and write the result for screen_candidates.py to use
in place of the raw field. This is a JARVIS-internal hull (same functional as the entries), so it
is the correct quantity; it is not MP's hull.

Usage:
    python recompute_jarvis_hull.py            -> data/processed/jarvis_ehull_recomputed.csv
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "jarvis_cache" / "jdft_3d-9-24-2025.json.zip"
OUT = ROOT / "data" / "processed" / "jarvis_ehull_recomputed.csv"


def main():
    with zipfile.ZipFile(RAW) as z:
        raw = json.loads(z.read(z.namelist()[0]))
    entries = []
    by_elem = defaultdict(list)
    for r in raw:
        fe = r.get("formation_energy_peratom")
        if r.get("typ") != "bulk" or not isinstance(fe, (int, float)):
            continue
        comp = Composition(r["formula"])
        e = PDEntry(comp, fe * comp.num_atoms, name=r["jid"])
        entries.append(e)
        for el in comp.elements:
            by_elem[el].append(len(entries) - 1)
    print(f"{len(entries)} JARVIS bulk entries with formation energies")

    # candidates: every JARVIS record that reached the screened set (+ the harmonized ones used
    # anywhere downstream); recompute for all JARVIS records in the harmonized dataset
    recs = json.load(open(ROOT / "data" / "processed" / "harmonized_v2.json"))
    targets = [r for r in recs if r.get("source_database") == "JARVIS"]
    print(f"{len(targets)} JARVIS records in harmonized_v2.json")
    jid_to_entry = {e.name: e for e in entries}

    pd_cache = {}
    rows = []
    for r in targets:
        e = jid_to_entry.get(r["material_id"])
        if e is None:
            rows.append({"material_id": r["material_id"], "ehull_recomputed": None, "note": "no formation energy"})
            continue
        system = frozenset(e.composition.elements)
        if system not in pd_cache:
            idx = set.intersection(*(set(by_elem[el]) for el in system))
            sub = [entries[i] for i in idx if set(entries[i].composition.elements) <= system]
            # elemental references must exist for the hull to be defined
            have = {el for s in sub for el in s.composition.elements if s.composition.is_element}
            if have != system:
                for el in system - have:
                    sub.append(PDEntry(Composition(str(el)), 0.0, name=f"ref-{el}"))
            pd_cache[system] = PhaseDiagram(sub)
        pdg = pd_cache[system]
        rows.append({"material_id": r["material_id"], "formula": r["formula"],
                     "formation_energy_peratom": e.energy / e.composition.num_atoms,
                     "ehull_raw_field": r.get("energy_above_hull"),
                     "ehull_recomputed": round(float(pdg.get_e_above_hull(e)), 5), "note": ""})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    ok = df["ehull_recomputed"].dropna()
    print(f"wrote {OUT.relative_to(ROOT)}: {len(ok)} recomputed; <=0.1 eV/atom: {(ok <= 0.1).sum()}; "
          f"median {ok.median():.3f} (raw field median {df['ehull_raw_field'].median():.3f})")
    for jid in ("JVASP-14014", "JVASP-68948", "JVASP-93203", "JVASP-94227"):
        print(df[df.material_id == jid].to_string(index=False, header=False))


if __name__ == "__main__":
    main()
