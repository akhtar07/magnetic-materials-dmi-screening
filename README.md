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

Of 26 Materials-Project-labeled "ground-truth" ferromagnets pulled from the top of the DMI-screening
shortlist, all 26 are resolved by DFT energy-mapping with a per-calculation spin-state audit and
same-cell (fixed-lattice) FM/AFM "tight" pairs for the marginal cases: **21 are actually
antiferromagnetic or ferrimagnetic ground states**, 3 are confirmed FM, and 2 are degenerate within
0.10 meV/atom. This independently
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
LAMMPS trajectory dumps and the raw `src/spin_dynamics/logs_*/` thermo logs of the M(T) and
finite-size-scaling sweeps — several GB total) are excluded from version control; the
analysis-ready reductions of the spin-dynamics runs are committed under `outputs/`
(`*_mvt*.csv`, `*_fss.csv`), and the rest is regenerable
from the scripts in `src/` given the harmonized dataset and the documented run parameters in
`docs/` and the `.lammps`/INCAR-derived settings noted inline in the code and paper.

## Reproducing a stage

Each pipeline stage has its own conda environment spec under `environments/`. A Materials Project
API key is required for acquisition and structure backfill (`MP_API_KEY` environment variable;
never commit this key). VASP and LAMMPS-SPIN (with the SPIN package compiled in) are required for
the DFT validation and spin-dynamics stages respectively and are not distributed here.

## Known limitations (documented in the paper)

- The 26-candidate DFT validation batch is drawn from the top of one DMI-filtered shortlist
  (1,864 candidates pass the funnel after the 2026-09-20 re-screen, 1,195 with the hull as a hard
  filter), not a random or stratified sample — the mislabeling rate
  applies to this structural family, not the full database.
- The verdict rule is asymmetric: one audited competing configuration below FM by more than 0.10 meV/atom
  overturns the label definitively, whereas "confirmed" needs every staged configuration to pass the
  spin-state audit (`outputs/dft_spin_state_audit.csv`). Where a loose (ISIF=3, each ordering at its own
  lattice) pair and a tight (same-cell, ISIF=2) pair both exist, the tight margin is used; in the sixteen
  configurations computed both ways the two protocols differ by up to 1.3 meV/atom, and of the six
  loose-only overturns with |margin| < 2 meV/atom re-decided with tight pairs (Batch D, 2026-09-21) all
  six kept their sign (mp-1173143, mp-551086, mp-1197024, mp-1343817, mp-22972, mp-682554). The
  mp-682554 pair is `provisional` (status column of `outputs/dft_tight_validation.csv`): the farm job was
  cancelled on 21 Sep with both 130-atom relaxations 2-3 ionic steps short of the 0.01 eV/Å force
  criterion (max|F| 0.024/0.036 eV/Å, total energies stationary to 1e-6 eV over the last three steps);
  `analyze_tight_pairs.py` accepts such a pair when forces are < 0.05 eV/Å and the energy drift is
  < 0.05 meV/atom. The same cancellation dropped the mp-682554 5_afm production run (never started; the
  candidate is already overturned by 3_afm) and the mp-850225 c-stacked pair (`afm_cstack/fm_in_cstack`,
  staged by `src/dft/stage_mp850225_cstack.py` to separate the in-plane and out-of-plane exchange, dE = 12 J_2;
  `not run`). Per-candidate verdicts: `outputs/dft_verdicts.csv` (tight pairs:
  `outputs/dft_tight_validation.csv`); figures regenerate from `src/figures/make_paper_figures.py`.
- The DMI constant for mp-754090 resolves one projection of a 3-component vector (space group
  \(P1\), no symmetry to fix a preferred canting plane); a full characterization would probe
  additional planes.
- The spin-dynamics case study (mp-754090) maps the FM/AFM DFT pair onto a single J on the
  twelve-neighbour close-packed Fe sublattice (six in-plane bonds at 5.11-5.16 Å, six inter-layer at
  5.41-5.43 Å): the AFM configuration flips 8 inter-sublattice bonds per cell, so ΔE = 16 J and
  J = 44.54/16 = 2.78 meV per bond from the same-cell tight pair (loose pair 3.01 meV, an 8% protocol
  spread); D_y = 0.097 meV per bond (D/J = 3.5%). Two earlier drafts were wrong: ΔE/4 = 12.06 meV
  (double counting) and then a 5.25 Å cutoff that kept only the six coplanar in-plane bonds (J = 5.57
  meV on a stack of uncoupled 2D layers, which has no long-range order and gave |m|(10 K) = 0.64 in
  Monte Carlo). The v4 model (`src/spin_dynamics/build_mp754090_model.py`, run-0 verified against the
  DFT pair) is exchange-only: LAMMPS `spin/dmi` cannot reproduce the DFT canting probe for this
  near-centrosymmetric shell (its energy is D (v×y)·Σe_b with Σe_b ≈ 0), so the DMI is bounded by a
  control sweep (≤0.013 change in |m|). Langevin sweeps use α = 1 and 20 ps (at α = 0.05 neither 4 nor
  20 ps equilibrated; `outputs/mp754090_mvt_v3_*` (6-bond model) and `mp1079285_mvt_4ps/a005_20ps` are the archived
  unequilibrated/superseded sweeps); |m| falls from 0.966 at 10 K to 0.343 at 100 K and 0.068 at
  125 K, bracketed by the classical fcc (≈103 K) and mean-field (129 K) estimates and reproduced by an
  independent Metropolis MC (`mc_check_mp754090.py`, `outputs/mp754090_mc_check_*.txt`). A Binder-cumulant
  finite-size scaling over four aspect-ratio-preserved cells (N = 256, 864, 2048, 4000;
  `src/spin_dynamics/run_fss.sh`, `analyze_fss.py`, `outputs/mp754090_fss.csv`) sharpens this to
  T_c = 101.2 ± 1.3 K (pairwise crossings 99.5/101.3/102.7 K, monotonic in N; χ peaks at 105 K at all
  four sizes) — the classical-fcc value, 22% below mean field (which therefore overestimates it by 28%). The in-plane,
  inter-layer and same-sublattice bonds cannot be separated by one pair and share the single J; the Fe
  moment is treated as a classical unit vector. mp-1079285 (three-shell model, J_b = 14.61, J_2b = 3.31,
  J_perp = 2.80 meV, |D| = 0.098 meV calibrated to the spiral pairs; `outputs/mp1079285_mvt.csv`)
  orders between 125 and 150 K in the 2048-spin cell; the same FSS analysis
  (`outputs/mp1079285_fss.csv`, 80 ps and 10-14 seeds per point at 135-150 K) gives crossings at
  140.9/141.4/135.6 K, T_c = 139.3 ± 2.6 K = 0.68 T_c^MF (T_c^MF = 203 K). The crossings are not
  monotonic in N as mp-754090's are: the exchange is strongly anisotropic (J_b/J_perp = 5.2, FM on every
  shell), so the U4 curves separate gradually instead of pivoting, and the 155/160 K points carry only
  four seeds each. Re-deriving both crossings from the last quarter of each run instead of the last half
  leaves mp-754090 at 101.1 K but moves mp-1079285 to 135.2 K (all of it from the 2048x4000 pair);
  analyzing only the six 80 ps runs gives 139.6 K. The quoted ± are lower bounds on the uncertainty. J_b and J_2b there come from different cells/k-meshes
  (1x2x1 with 3x3x3 vs 1x4x1 with 3x2x3); the same-cell +-+- pair (afm_2b_in_4b, provisional: cut off at
  ionic step 24 with max|F| 0.015 eV/Å) gives J_b = 12.62 meV (14% lower), hence J_2b = 4.31, zJ = 50.6
  meV, T_c^MF = 196 K (`build_mp1079285_model.py --jb-pair 2b_in_4b --suffix _samecell` ->
  `outputs/mp1079285_samecell_spin_model.json`, sweep `outputs/mp1079285_samecell_mvt.csv`); the two
  values bracket J_b and |D|/J_b = 0.67-0.78%.
- DMI vectors were extracted for two of the three confirmed-FM compounds (mp-754090 canting probe,
  mp-1079285 spiral pairs). For mp-850225 (FeP6(WO8)3, one Fe per rhombohedral cell, Bravais Fe
  sublattice) a two-sublattice canting cannot see D and a 102-atom noncollinear spiral pair (~5,800
  core-hours) was designed but not run; mp-1228547 and mp-1216981 were not probed (degenerate /
  overturned, so no FM ground state to host a DMI in).
- 24 of the 26 staged candidates are MP-theoretical structures (23 with no ICSD entry); only
  mp-1197024 and mp-22972 are experimentally reported. Confirmed-FM verdicts are statements about
  hypothetical structures (`outputs/candidate_provenance.csv`).
- The JARVIS `ehull` field of the dft_3d snapshot is not a hull distance (median 1.7 eV/atom); the
  screening now uses hull energies recomputed from JARVIS's own formation energies
  (`src/ranking/recompute_jarvis_hull.py`). OQMD structures (816 records) could not be fetched
  (endpoint unreachable), so those records carry no local-geometry features.
