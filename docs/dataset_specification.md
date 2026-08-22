# Dataset Specification — v2

## Project

Machine Learning–Guided Screening of Magnetic Materials for Thermally Robust Non-Volatile Data Storage

## What changed from v1

v1 (inherited from the original repo) defined the target schema but never documented which sources actually
populated it, or with what provenance. v2 is written against what is **actually acquired**, checked in against
`data/raw/`, and traceable back to a specific source database and record.

## Sources currently harmonized

| Source | Records | Fields present | Notes |
|---|---|---|---|
| Materials Project | 37,783 | full schema below | via `mp-api`, reused from prior run |
| JARVIS-DFT | 19,116 | full schema below | reused from prior run |
| AFLOW | 2,639 | full schema below | reused from prior run |
| MAGNDATA | 79 | full schema + `structure`, `species`, per-site `magmom` | scraped from Bilbao; only 102 of an unknown-but-much-larger total were scraped, 79 parsed cleanly — **this needs a much larger scrape, not just relabeling**, before it's a trustworthy bias-correction training set (see Fahmy 2026, which cross-checked 7,843 MP entries) |

## Sources not yet acquired (open work)

OQMD, C2DB, 2DMatPedia, NOMAD — see roadmap Section 5 for per-source diagnosis priority. **No 2D-specific
data exists yet** (C2DB, 2DMatPedia are the only 2D sources in the proposal and both are unacquired), so the
"2D and 3D" framing is currently 3D-only in practice until these land.

## Harmonized schema (v2)

Superset of the v1 schema (`material_id`, `formula_pretty`→`formula`, `elements`, `nelements`→`num_elements`,
`symmetry`→`space_group`/`space_group_number`/`crystal_system`, `density`, `volume`, `band_gap`, `ordering`,
`total_magnetization`) plus fields that were already present in the acquired data and are useful downstream:

| Field | Required | Reason |
|---|---|---|
| `material_id` | Yes | Unique ID, prefixed by source (`mp-`, `jvasp-`, `aflow-`, `magndata-`) |
| `source_database` | Yes | Provenance — which of the 8 databases this record came from |
| `formula` | Yes | Human-readable chemical formula |
| `elements` / `num_elements` | Yes | Composition |
| `chemical_system` | No | e.g. `"Eu-Fe-Hf-K-O"` |
| `space_group` / `space_group_number` / `crystal_system` | No | Symmetry |
| `nsites` | No | Atom count in the cell |
| `density` / `volume` | Yes | Physical properties |
| `band_gap` | Yes | Electronic descriptor |
| `formation_energy` / `energy_above_hull` | No | Stability descriptors (used for OQMD cross-validation once acquired) |
| `ordering` | Yes | Magnetic ordering label (FM/AFM/NM/FiM/...) — **the field the Phase 1 bias classifier corrects** |
| `total_magnetization` | Yes | Magnetic strength descriptor |
| `is_magnetic` / `is_metal` | No | Convenience booleans |
| `structure` | No | Full pymatgen `Structure` dict — present for MAGNDATA, missing for the tabular-only sources; required input for Phase 3 (MLIP fine-tuning) and Phase 4 (spin dynamics), so this needs backfilling for MP/JARVIS/AFLOW records that get shortlisted |
| `magmom_per_site` | No | Per-site magnetic moments — present for MAGNDATA only; needed for spin-dynamics initialization |

## Provenance and conflict tracking

Every record keeps `source_database`. When the same material appears in more than one source, both are kept
as separate records tagged with their own `source_database` rather than silently merged — dedup/conflict
resolution is a downstream step (Phase 0 in the roadmap), not baked into acquisition.
