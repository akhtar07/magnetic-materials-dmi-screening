"""Composition-based feature engineering for the LightGBM baseline (Phase 2), following the
enriched-elemental-vector approach of Verma, Jami & Bhattacharya (2025, arXiv:2507.01913):
per-element atomic properties aggregated as mean/min/max/range/std across the composition,
plus group/period stats and transition-metal/rare-earth/metal fractions.

Self-contained (uses pymatgen's periodic table data directly) -- does NOT depend on the old
repo's magnetic_ai package, which lives in the read-only /data/Faiz_deep_work copy and isn't
on this project's Python path.

Usage:
    python build_features.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import Element

ROOT = Path(__file__).resolve().parents[2]

TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}
RARE_EARTHS = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc",
}


def element_props(symbol: str) -> dict:
    el = Element(symbol)
    return {
        "Z": el.Z,
        "mass": float(el.atomic_mass),
        "X": float(el.X) if el.X else np.nan,  # Pauling electronegativity
        "radius": float(el.atomic_radius) if el.atomic_radius else np.nan,
        "group": el.group,
        "row": el.row,
    }


def featurize(elements: list[str]) -> dict:
    props = [element_props(e) for e in elements if e]
    if not props:
        return {}
    Z = np.array([p["Z"] for p in props], dtype=float)
    mass = np.array([p["mass"] for p in props], dtype=float)
    X = np.array([p["X"] for p in props], dtype=float)
    radius = np.array([p["radius"] for p in props], dtype=float)
    group = np.array([p["group"] for p in props], dtype=float)
    row = np.array([p["row"] for p in props], dtype=float)

    def stats(arr, name):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {f"mean_{name}": np.nan, f"min_{name}": np.nan, f"max_{name}": np.nan,
                    f"range_{name}": np.nan, f"std_{name}": np.nan}
        return {f"mean_{name}": arr.mean(), f"min_{name}": arr.min(), f"max_{name}": arr.max(),
                f"range_{name}": arr.max() - arr.min(), f"std_{name}": arr.std()}

    feats = {}
    feats.update(stats(Z, "atomic_number"))
    feats.update(stats(mass, "atomic_mass"))
    feats.update(stats(X, "electronegativity"))
    feats.update(stats(radius, "atomic_radius"))
    feats["mean_group"] = np.nanmean(group)
    feats["mean_period"] = np.nanmean(row)
    feats["fraction_transition_metals"] = sum(e in TRANSITION_METALS for e in elements) / len(elements)
    feats["fraction_rare_earth"] = sum(e in RARE_EARTHS for e in elements) / len(elements)
    feats["num_elements"] = len(elements)
    return feats


def main():
    with open(ROOT / "data" / "processed" / "harmonized_v2.json") as f:
        records = json.load(f)

    labeled = [r for r in records if r.get("ordering") and r["ordering"] != "Unknown" and r.get("elements")]
    print(f"{len(labeled)} labeled records (excluding null/'Unknown' ordering)")

    rows = []
    for r in labeled:
        feats = featurize(r["elements"])
        if not feats:
            continue
        feats["material_id"] = r["material_id"]
        feats["source_database"] = r["source_database"]
        feats["density"] = r.get("density")
        feats["ordering"] = r["ordering"]
        rows.append(feats)

    df = pd.DataFrame(rows)
    out_path = ROOT / "data" / "features" / "baseline_features_v1.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} feature rows to {out_path}")
    print("Class balance:\n", df["ordering"].value_counts())


if __name__ == "__main__":
    main()
