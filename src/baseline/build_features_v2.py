"""Structure-aware feature engineering (v2), building on build_features.py (v1, composition-only).

Motivation: the v1 retrain after the MAGNDATA rescrape (79 -> 1,919 records) left accuracy/balanced
accuracy essentially flat despite 24x more ground-truth data (see docs/project_status_report.pdf,
Phase 1 section) -- because v1's features are purely composition-derived and never touch the
`structure` / `magmom_per_site` fields the new MAGNDATA records actually carry. This script adds
two tiers of genuinely structural features on top of v1's composition features:

  Tier 1 (broad, ~65-100% coverage across ALL sources, not just MAGNDATA): space_group_number,
  a crystal-system bucket derived from it, nsites, volume. These are already present in the
  harmonized schema for MP/JARVIS/AFLOW too (confirmed by direct inspection -- see below) but were
  never used by v1. Free, broad-coverage signal that plausibly matters: crystallographic symmetry
  constrains which magnetic point groups (and hence orderings) are compatible with a structure.

  Tier 2 (narrow, MAGNDATA-only, ~3% coverage -- the only source with full atomic `structure`):
  local-geometry features around magnetic-species sites (nearest magnetic-magnetic neighbor
  distance, coordination number), motivated by Goodenough-Kanamori superexchange rules (bond
  length/geometry between magnetic centers governs FM vs. AFM coupling sign). Deliberately built
  from site POSITIONS and SPECIES only, never from magmom_per_site -- the `ordering` label for
  MAGNDATA rows is itself derived from magmom vectors (see acquire_magndata.py's
  classify_ordering()), so any feature computed from magmom would leak the label-generating
  function for those rows. Coordinates/species carry real independent structural signal without
  that leakage risk.

  For rows without a `structure` (97% of the dataset), Tier 2 columns are left as NaN.
  LightGBM handles missing values natively (learns a default split direction), so this does not
  require dropping or imputing -- the model can use Tier 2 features where available and fall back
  to Tier 1 + v1 composition features elsewhere.

Usage:
    python build_features_v2.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
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
MAGNETIC_ELEMENTS = TRANSITION_METALS | RARE_EARTHS

# Standard space-group-number ranges -> crystal system (International Tables convention).
# MAGNDATA's space_group_number is the parent (BNS) space group number, same 1-230 range as
# MP/JARVIS/AFLOW's, so this bucketing applies uniformly across all sources.
def crystal_system_from_sg(sg: int | None) -> str | None:
    if sg is None:
        return None
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    if 195 <= sg <= 230:
        return "cubic"
    return None


def element_props(symbol: str) -> dict:
    el = Element(symbol)
    return {
        "Z": el.Z,
        "mass": float(el.atomic_mass),
        "X": float(el.X) if el.X else np.nan,
        "radius": float(el.atomic_radius) if el.atomic_radius else np.nan,
        "group": el.group,
        "row": el.row,
    }


def composition_features(elements: list[str]) -> dict:
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


def local_structure_features(structure_dict: dict) -> dict:
    """Tier 2: local-geometry features around magnetic-species sites, from site positions/species
    only (never magmom -- see module docstring on leakage)."""
    empty = {
        "n_magnetic_sites": np.nan, "fraction_magnetic_sites": np.nan,
        "mean_magmag_nn_distance": np.nan, "min_magmag_nn_distance": np.nan,
        "std_magmag_nn_distance": np.nan, "mean_magnetic_coordination": np.nan,
    }
    try:
        structure = Structure.from_dict(structure_dict)
    except Exception:
        return empty

    mag_idx = set()
    for i, site in enumerate(structure):
        try:
            symbol = site.specie.symbol
        except AttributeError:
            # disordered site (multiple species) -- treat as magnetic if any component is
            symbol = None
            if any(el.symbol in MAGNETIC_ELEMENTS for el in site.species.elements):
                mag_idx.add(i)
            continue
        if symbol in MAGNETIC_ELEMENTS:
            mag_idx.add(i)

    if not mag_idx:
        return empty

    try:
        neighbors = structure.get_all_neighbors(r=6.0)
    except Exception:
        return empty

    min_dists, coord_numbers = [], []
    for i in mag_idx:
        mag_neighbor_dists = [n.nn_distance for n in neighbors[i] if n.index in mag_idx]
        if mag_neighbor_dists:
            min_dists.append(min(mag_neighbor_dists))
        coord_numbers.append(sum(1 for n in neighbors[i] if n.nn_distance < 3.2))

    return {
        "n_magnetic_sites": len(mag_idx),
        "fraction_magnetic_sites": len(mag_idx) / len(structure),
        "mean_magmag_nn_distance": float(np.mean(min_dists)) if min_dists else np.nan,
        "min_magmag_nn_distance": float(np.min(min_dists)) if min_dists else np.nan,
        "std_magmag_nn_distance": float(np.std(min_dists)) if len(min_dists) > 1 else 0.0,
        "mean_magnetic_coordination": float(np.mean(coord_numbers)) if coord_numbers else np.nan,
    }


def main():
    with open(ROOT / "data" / "processed" / "harmonized_v2.json") as f:
        records = json.load(f)

    labeled = [r for r in records if r.get("ordering") and r["ordering"] != "Unknown" and r.get("elements")]
    print(f"{len(labeled)} labeled records (excluding null/'Unknown' ordering)")

    rows = []
    n_with_structure = 0
    for r in labeled:
        feats = composition_features(r["elements"])
        if not feats:
            continue
        feats["material_id"] = r["material_id"]
        feats["source_database"] = r["source_database"]
        feats["density"] = r.get("density")

        # Tier 1: broad structural fields, already in the schema for most sources.
        feats["nsites"] = r.get("nsites")
        feats["volume"] = r.get("volume")
        sg = r.get("space_group_number")
        feats["space_group_number"] = sg
        feats["crystal_system_bucket"] = crystal_system_from_sg(sg)

        # Tier 2: local structure geometry, MAGNDATA-only (NaN elsewhere).
        if r.get("structure") is not None:
            feats.update(local_structure_features(r["structure"]))
            n_with_structure += 1
        else:
            feats.update({
                "n_magnetic_sites": np.nan, "fraction_magnetic_sites": np.nan,
                "mean_magmag_nn_distance": np.nan, "min_magmag_nn_distance": np.nan,
                "std_magmag_nn_distance": np.nan, "mean_magnetic_coordination": np.nan,
            })

        feats["ordering"] = r["ordering"]
        rows.append(feats)

    df = pd.DataFrame(rows)
    out_path = ROOT / "data" / "features" / "baseline_features_v2.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} feature rows to {out_path} ({n_with_structure} with Tier 2 structural features)")
    print("Class balance:\n", df["ordering"].value_counts())
    print("Tier 1 coverage: space_group_number "
          f"{df['space_group_number'].notna().mean():.1%}, nsites {df['nsites'].notna().mean():.1%}, "
          f"volume {df['volume'].notna().mean():.1%}")


if __name__ == "__main__":
    main()
