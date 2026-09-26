"""Provenance check for the DFT-validated candidates: is each Materials Project entry backed by
an experimentally reported structure (ICSD ids, MP's `theoretical == False`) or is it a
hypothetical/prototype-derived structure that has never been synthesised?

Why: a "DMI candidate" that only exists in a database is a much weaker result than one with an
experimental crystal structure. The screening funnel never looked at this, so the paper could
not say which of its FM-confirmed candidates are real compounds. MP exposes exactly this in the
summary endpoint (`theoretical`, `database_IDs.icsd`) -- pulled here with mp_api (MP_API_KEY from
the environment, never printed) and written to outputs/candidate_provenance.csv together with the
current DFT verdict from outputs/dft_ordering_validation.csv.

Usage:
    MP_API_KEY=... python candidate_provenance.py
"""
import csv
import os
from pathlib import Path

import pandas as pd
from mp_api.client import MPRester

ROOT = Path(__file__).resolve().parents[2]


def main():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit("MP_API_KEY not set (see .mp_api_key.local)")
    val = pd.read_csv(ROOT / "outputs" / "dft_ordering_validation.csv")
    ids = sorted(val["candidate_id"].unique())
    verdict = {}
    for cid, g in val.groupby("candidate_id"):
        gs = g[g["is_dft_ground_state"] == True]
        verdict[cid] = dict(
            dft_ground_state=gs["ordering"].iloc[0] if len(gs) else "",
            complete=bool(g["candidate_complete"].iloc[0]),
            predicted=g["predicted_ordering"].iloc[0],
            formula=g["formula"].iloc[0],
        )

    with MPRester(api_key, mute_progress_bars=True) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=ids,
            fields=["material_id", "formula_pretty", "theoretical", "database_IDs", "energy_above_hull",
                    "is_stable", "symmetry", "ordering", "total_magnetization", "deprecated", "origins"])
        prov = {}
        for d in docs:
            dbids = d.database_IDs or {}
            icsd = dbids.get("icsd") or []
            prov[str(d.material_id)] = dict(
                mp_formula=d.formula_pretty,
                theoretical=d.theoretical,
                n_icsd=len(icsd),
                icsd_ids=";".join(str(x) for x in icsd),
                other_db_ids=";".join(f"{k}:{len(v)}" for k, v in dbids.items() if k != "icsd" and v),
                mp_energy_above_hull=d.energy_above_hull,
                mp_is_stable=d.is_stable,
                spacegroup=d.symmetry.symbol if d.symmetry else "",
                mp_ordering=str(d.ordering) if d.ordering is not None else "",
                mp_total_magnetization=d.total_magnetization,
                deprecated=d.deprecated,
            )

    rows = []
    for cid in ids:
        p = prov.get(cid, {"theoretical": "", "n_icsd": "", "icsd_ids": "", "other_db_ids": "", "mp_formula": "NOT RETURNED",
                           "mp_energy_above_hull": "", "mp_is_stable": "", "spacegroup": "", "mp_ordering": "",
                           "mp_total_magnetization": "", "deprecated": ""})
        rows.append({"candidate_id": cid, **verdict[cid], **p,
                     "experimental_structure": (p["theoretical"] is False) if p["theoretical"] != "" else ""})
    out = ROOT / "outputs" / "candidate_provenance.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    df = pd.DataFrame(rows)
    print(df[["candidate_id", "formula", "dft_ground_state", "complete", "theoretical", "n_icsd", "mp_energy_above_hull", "mp_ordering", "deprecated"]].to_string(index=False))
    print(f"\n{(df['experimental_structure'] == True).sum()}/{len(df)} candidates have an experimental (ICSD-backed) structure; wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
