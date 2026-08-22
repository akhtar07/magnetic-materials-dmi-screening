# DFT magnetic-ordering validation: mp-861559 (TaFeO4)

## Purpose

The screening pipeline (`src/ranking/screen_candidates.py`) labels candidates with an
`ordering_final` / `ordering_source` / `confidence` triple. For entries sourced from Materials
Project, `ordering_source=ground_truth, confidence=1.0` means "take MP's recorded ordering as
correct." This project had no DFT infrastructure to check that assumption until now. `mp-861559`
(TaFeO4) was picked as the first real validation case: run actual VASP calculations for the FM
configuration plus the symmetry-distinct AFM configurations MP's own ordering workflow would
consider, and see whether the lowest-energy one matches the label.

Method, staging, and job submission: `src/dft/prepare_magnetic_ordering_dft.py`,
`write_job_scripts.py`, `remote_assemble_and_submit.sh` (run on paramsanganak). Settings are
`pymatgen` `MPRelaxSet` — i.e. MP's own GGA+U/ENCUT/k-point convention — so results are directly
comparable to MP's own calculations. Results parsed and reconciled by
`src/dft/analyze_magnetic_ordering_results.py` → `outputs/dft_ordering_validation.csv`.

## Results

All 4 calculations converged cleanly (OUTCAR reports `reached required accuracy`, empty
`slurm.err`):

| calc | ordering | TOTEN (eV) | eV/atom | note |
|---|---|---|---|---|
| `1_afm` | AFM | −304.87416 | −8.468727 | **DFT ground state** |
| `2_afm` | AFM | −304.78236 | −8.466177 | |
| `3_afm` | AFM | −304.74876 | −8.465243 | |
| `0_fm`  | FM  | −304.60011 | −8.461114 | MP's recorded label |

All three symmetry-distinct AFM configurations tested come out lower than FM. The best AFM
configuration is **0.274 eV lower than FM** for this 36-atom (6 formula unit) cell — 0.0918 eV
even to the next-best AFM variant, so this is not enumeration noise.

Sanity check that the calculation setup itself is trustworthy: our FM run's converged
magnetization is 29.99998 μB, versus MP's own recorded `total_magnetization` of 30.0022251 μB for
this entry (`data/raw/master_dataset_v4.json`) — a near-exact match. So the local FM calculation
is reproducing MP's own result almost exactly; the discrepancy is not a setup/convention mismatch.

## Conclusion

MP's routine per-entry calculation pipeline does not run a magnetic-ordering enumeration — it
relaxes from a single initial magnetic-moment guess per structure. For `mp-861559`, that guess
converged to FM, and FM is what got stored as the "ground truth" ordering with confidence 1.0
in this project's harmonized dataset. It was never checked against AFM alternatives on MP's side.
This project's own DFT run — using MP's exact method, but *actually enumerating* the AFM
configurations `MagneticStructureEnumerator` finds — shows FM is not the ground state for this
compound; an AFM ordering is lower by ~0.0076 eV/atom.

**Implication for the pipeline:** `ordering_source=ground_truth, confidence=1.0` should be read as
"matches what MP happened to relax to," not "verified magnetic ground state." For a compound like
this with several competing AFM configurations close in energy, that is a meaningfully weaker
guarantee. This one data point isn't enough to say how often MP's stored ordering is wrong project
-wide, but it means the 30 `ground_truth`-labeled entries in `outputs/phase4_shortlist.csv` (and
the larger `ordering_source=ground_truth` population feeding the baseline models) carry real, not
just nominal, label uncertainty. Running the same AFM-enumeration check on a handful of the other
`ground_truth` entries would establish whether ~this kind of mislabeling rate is common before
trusting those labels at face value in the ranking/screening steps.

## Raw data

- Per-configuration results: `outputs/dft_ordering_validation.csv`
- Staged calculations: `data/dft_staging/mp-861559/{0_fm,1_afm,2_afm,3_afm}/`
- MP's own recorded values for this entry: `data/raw/master_dataset_v4.json` (`material_id: mp-861559`)
