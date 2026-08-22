# Machine-Learning-Guided Screening of Magnetic Materials for DMI-Hosting Candidates

This repository accompanies the paper **"DFT-Validated Evidence of Ferromagnetic Mislabeling Among
Dzyaloshinskii–Moriya-Candidate Materials in the Materials Project Database"** (`paper/main.tex`,
compiled PDF at `paper/main.pdf`).

## What this project does

A five-phase pipeline that screens public materials databases for compounds likely to host a
Dzyaloshinskii–Moriya interaction (DMI) — the mechanism behind chiral spin textures and skyrmions
relevant to non-volatile memory — and checks the screening assumptions against real DFT and
atomistic spin-dynamics simulation:

1. **`src/harmonize/`, `src/acquisition/`** — merge 7 public sources (Materials Project, JARVIS-DFT,
   AFLOW, MAGNDATA, 2DMatPedia, OQMD, NOMAD) into one 64,635-record schema.
2. **`src/baseline/`** — a LightGBM magnetic-ordering (FM/AFM/FiM/NM) classifier, trained on
   MAGNDATA ground truth, using composition + structural features.
3. **`src/ranking/`** — filters the harmonized database for DMI-compatible candidates
   (net moment, noncentrosymmetric space group, heavy spin-orbit-coupling element, hull stability),
   backfills structures, and ranks by MACE-relaxation consistency.
4. **`src/mlip/`** — fine-tunes a MACE foundation-model potential on the harmonized dataset for
   fast, magnetism-aware relaxation.
5. **`src/dft/`, `src/spin_dynamics/`** — real VASP magnetic-ordering validation (collinear
   energy mapping) on shortlisted candidates, plus DFT-derived exchange (\(J\)) and DMI (\(D\))
   constants driving finite-temperature LAMMPS-SPIN dynamics for one case study
   (Li₄Fe₂TeWO₁₂, mp-754090).

## Headline result

Of 24 Materials-Project-labeled "ground-truth" ferromagnets pulled from the top of the DMI-screening
shortlist, direct DFT energy-mapping finds **17 are actually antiferromagnetic ground states**, 4 are
confirmed FM, and 3 are degenerate within electronic-convergence precision. This independently
corroborates, at higher fidelity within the DMI-relevant structural family, the systematic MP
ferromagnetic-bias finding recently reported from ML classifiers trained on experimentally solved
magnetic structures (Fahmy, arXiv:2509.05909 — included at `reference/2509.05909v2.pdf`). See
`paper/main.pdf` for the full argument, and `outputs/dft_ordering_validation.csv` for the raw
per-candidate results.

## Repository layout

```
paper/          Paper source (main.tex), compiled PDF, and figures
reference/      The related-work PDF this paper's DFT results are compared against
src/            Pipeline code, phases 0-4 (see above)
outputs/        Curated result tables (classifier metrics, screening funnel, DFT validation csv/json)
docs/           Full project report, per-candidate DFT validation writeups, dataset spec
environments/   Conda environment specs for each pipeline stage
```

Large raw/intermediate artifacts (DFT staging directories, MACE fine-tuning configuration shards,
LAMMPS trajectory dumps — several GB total) are excluded from version control; they are regenerable
from the scripts in `src/` given the harmonized dataset and the documented run parameters in
`docs/` and the `.lammps`/INCAR-derived settings noted inline in the code and paper.

## Reproducing a stage

Each pipeline stage has its own conda environment spec under `environments/`. A Materials Project
API key is required for acquisition and structure backfill (`MP_API_KEY` environment variable;
never commit this key). VASP and LAMMPS-SPIN (with the SPIN package compiled in) are required for
the DFT validation and spin-dynamics stages respectively and are not distributed here.

## Known limitations (documented in the paper)

- The 24-candidate DFT validation batch is drawn from the top of one DMI-filtered shortlist
  (1,647 candidates pass the funnel), not a random or stratified sample — the mislabeling rate
  applies to this structural family, not the full database.
- Two further candidates (mp-1173143, mp-682554) are incomplete and excluded from the analysis.
- The DMI constant for mp-754090 resolves one projection of a 3-component vector (space group
  \(P1\), no symmetry to fix a preferred canting plane); a full characterization would probe
  additional planes.
- The spin-dynamics magnetization curve's absolute scale is suppressed by multi-domain cancellation
  in the periodic cell; its shape (a knee near 150-200 K), not its scale, is the physical signal.
