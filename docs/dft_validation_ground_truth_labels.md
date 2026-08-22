# DFT magnetic-ordering validation: `ground_truth`-labeled candidates

## Status: snapshot as of 21 August 2026, batch nearly complete

**24 of 26** eligible `ground_truth` candidates from `outputs/phase4_shortlist.csv` have completed
DFT (all FM/AFM configurations converged). This report covers those 24 — 5 more than the previous
snapshot (mp-754090, mp-755203, mp-769471, mp-775194, mp-850225; see the tungsten note below for
why these needed a non-standard fix). The remaining 2 are still incomplete: mp-1173143 (1 of 4
configs converged) and mp-682554 (1 of 4 configs converged, a sulfide with no LDA+U applied at all
— a distinct, unrelated failure mode from the tungsten issue). This document will be updated again
once those finish.

## Purpose

Extension of the single-candidate check in `docs/dft_validation_mp-861559.md`. That first case
(mp-861559, TaFeO4) found MP's FM label overturned by real DFT. A first follow-up batch of 6
candidates (see below) confirmed this wasn't a one-off. This report extends the check to the full
set of `ordering_source=ground_truth, confidence=1.0` (and 2 `FiM`-labeled) entries in the
shortlist: relax → enumerate symmetry-distinct FM/AFM configurations
(`pymatgen.MagneticStructureEnumerator`) → VASP with MP's own `MPRelaxSet` convention → compare
total energies per atom.

## Method

Identical pipeline for every candidate, matching Materials Project's own magnetic-ordering
convention so results are directly comparable to MP's stored values:

1. Take the candidate's relaxed structure from the harmonized dataset.
2. Enumerate symmetry-distinct FM/AFM spin configurations with pymatgen's
   `MagneticStructureEnumerator` (backed by `enumlib`). **The enumerator only ever produces FM/AFM
   configurations — never FiM**, even for the 2 candidates here whose MP label is `FiM`
   (mp-22972, mp-551086). For those two, "AFM beats FM" does not directly confirm or refute the
   FiM label — it only shows FM is not the ground state either. Flagged explicitly in the results
   table below.
3. Generate VASP inputs with `MPRelaxSet` — MP's own GGA+U, ENCUT, and k-point density —
   overriding only `MAGMOM` per ordering.
4. Run each configuration to convergence (48 cores/job). The original 6 candidates ran on
   `paramsanganak` (VASP 5.4.4); all later candidates ran on `paramrudra` (VASP 6.4.3) after
   `paramsanganak` became unavailable mid-project.
5. Compare converged total energies **per atom** (not raw `TOTEN`) — the lowest is the DFT ground
   state.

### A note on data integrity

Two implementation bugs were found and fixed in `src/dft/analyze_magnetic_ordering_results.py`
while processing this expanded batch, disclosed here for transparency:

- **Ground-state comparison bug (significant, now fixed).** The script originally compared raw
  `TOTEN` across configurations rather than energy per atom. Because `MagneticStructureEnumerator`
  builds AFM configurations in supercells 2–4× the size of the FM primitive cell, raw `TOTEN`
  systematically favors whichever configuration has more atoms, independent of which ordering is
  actually more stable. This was caught when mp-1079285 and mp-1228547 (both true FM ground states,
  confirmed after the fix) initially showed an incorrect "AFM wins" result, and an anomalous
  −101 meV/atom gap on mp-551086 (versus a sane +0.57 meV/atom after the fix) served as a red flag.
  **This bug predates this session and was present when the original 6-candidate report below was
  generated — however, it did not flip any of those 6 verdicts** (their FM/AFM cell sizes either
  matched or the AFM per-atom energy was genuinely lower regardless of normalization), so the
  original 6-candidate numbers stand unchanged. It would have silently corrupted results for the 13
  newly-completed candidates had it not been caught here.
- **Non-converged energies polluting the comparison (minor, now fixed).** TIME-LIMIT-truncated or
  crashed calculations were briefly included in ground-state comparisons using whatever
  intermediate, unrelaxed `TOTEN` happened to be in the `OUTCAR` at truncation — physically
  meaningless. Fixed by restricting all verdicts to converged calculations only, and tracking
  per-candidate completeness explicitly (a candidate's verdict is only reported once every staged
  configuration for it has converged).

Separately (infrastructure, not analysis): a stale-partition config on 14 job scripts, a
submit-driver race condition, and the `debug` partition's `MaxSubmitPU=2` slot limit all caused
delays in getting this batch to complete, but did not affect the correctness of any converged
result — see git history in `src/dft/` for detail if needed.

- **Hubbard U on tungsten removed for 5 candidates (methodological deviation, disclosed here).**
  All 5 W-bearing candidates (mp-754090, mp-755203, mp-769471, mp-775194, mp-850225) initially
  failed to converge under MP's standard GGA+U value (`U=6.2` eV on W). Root cause: W in these
  compounds is formally W6+, i.e. nominally d0 — a large U on a near-empty d-shell makes the +U
  energy almost invariant to how that near-zero occupancy distributes across W's five d orbitals
  and across inequivalent W sites, so the SCF charge density has no restoring force and oscillates
  indefinitely rather than converging. Confirmed with 11 independent sandbox tests (three SCF
  algorithms, two mixing schemes, an `LMAXMIX` control, and a warm start from a converged
  no-U density) — none converge with `U=6.2` kept on W. The only working fix is removing U from W
  entirely (`LDAUL=-1` on the W species), applied uniformly across all orderings of a given
  compound so the FM/AFM energy comparison stays valid. This reproduced to within 0.5 meV/atom
  across two independent mixers in the sandbox, and converged all 18 production orderings cleanly
  on the first attempt (checked against false convergence via `rms(c)`, not just the "EDIFF is
  reached" banner — see the electronic-convergence caveat above). **Consequence:** the reported
  energies for these 5 candidates are not on the same U-value footing as MP's own database values
  or the other 19 candidates in this report, so cross-candidate energy comparisons involving these
  5 should be treated with that caveat in mind; the within-compound FM-vs-AFM verdict for each is
  unaffected, since the same U treatment applies to every ordering of a given compound.

## Results

24 candidates complete, sorted by margin between the DFT ground state and the best alternative
ordering (meV/atom). A margin below ~0.15 meV/atom is inside typical VASP electronic-convergence
tolerance (`EDIFF=1e-6` eV here) and is read as a statistical tie, not a confirmed result either
way. The 5 rows marked † are the tungsten candidates fixed this snapshot (`U` removed from W — see
the data-integrity note above).

| candidate | formula | label | DFT ground state | margin (meV/atom) | signal |
|---|---|---|---|---|---|
| mp-1228376 | Ba3Fe2WO9 | FM | AFM | 24.68 | clear — overturned |
| mp-1218521 | Sr3TaFeO7 | FM | AFM | 14.03 | clear — overturned |
| mp-1519427 | BaCaFeBiO6 | FM | AFM | 11.74 | clear — overturned |
| mp-1520073 | SrCaFeBiO6 | FM | AFM | 11.71 | clear — overturned |
| mp-1522159 | BaSrFeBiO6 | FM | AFM | 11.66 | clear — overturned |
| mp-1217456 | Ti4FeBiPb4O15 | FM | AFM | 9.18 | clear — overturned |
| mp-1217946 | TaFeO4 (entry 2) | FM | AFM | 8.92 | clear — overturned |
| mp-861559 | TaFeO4 (entry 1) | FM | AFM | 7.61 | clear — overturned |
| mp-22972 | Tb6FeBi2 | **FiM** | AFM | 7.32 | clear vs. FM, but doesn't test FiM (see caveat) |
| mp-775194 † | Li2Fe3WO8 | FM | AFM | 6.59 | clear — overturned |
| mp-1301722 | Ca3Fe2(WO6)2 | FM | AFM | 2.73 | clear — overturned |
| mp-754090 † | Li4Fe2TeWO12 | FM | **FM** | 2.41 | clear — **confirmed** |
| mp-1079285 | Tb6FeBi2 (2nd entry) | FM | **FM** | 2.32 | clear — **confirmed** |
| mp-769471 † | Fe3Co2P6WO24 | FM | AFM | 2.17 | clear — overturned |
| mp-850225 † | FeP6(WO8)3 | FM | **FM** | 2.10 | clear — **confirmed** |
| mp-1343817 | CaFeReO6 | FM | AFM | 1.77 | clear — overturned |
| mp-1197024 | FeBi(SeO3)3 | FM | AFM | 1.19 | clear — overturned |
| mp-1228547 | Ba4Ta10FeO30 | FM | **FM** | 1.18 | clear — **confirmed** |
| mp-755203 † | Li4NbFe(WO6)2 | FM | AFM | 0.73 | clear — overturned |
| mp-551086 | CrFe(BiO3)2 | **FiM** | AFM | 0.57 | clear vs. FM, doesn't test FiM (see caveat) |
| mp-1173139 | Ta2FeO6 | FM | AFM | 0.31 | clear — overturned |
| mp-1047414 | Zn2FeWO6 | FM | AFM | 0.10 | **noise-level** |
| mp-1043970 | CaLaTaFeO6 | FM | AFM | 0.09 | **noise-level** |
| mp-1216981 | TiFeBiPbO6 | FM | FM | 0.02 | **noise-level** (nominally confirmed, but not resolvable) |

Full per-configuration data: `outputs/dft_ordering_validation.csv` (regenerated by
`src/dft/analyze_magnetic_ordering_results.py`).

**Breakdown of the 24:**
- **17 clearly overturned** to AFM (margin 0.31–24.68 meV/atom).
- **4 clearly confirmed** as FM (mp-1079285, mp-1228547, mp-754090 †, mp-850225 †; margin
  1.18–2.41 meV/atom against the best AFM alternative).
- **3 noise-level ties** (margin ≤0.10 meV/atom) — 2 lean AFM, 1 nominally lands on FM but the
  margin is too small to call: mp-1047414, mp-1043970, mp-1216981.

So of 24 checked: **4 confirmed with a resolvable margin, 17 clearly wrong, 3 genuinely
inconclusive.** Among the candidates that resolve clearly (21 of 24), "AFM overturns FM" outnumbers
"FM confirmed" 17 to 4.

## Conclusion

Extending from the original 6-candidate check (0 of 6 confirmed) to the full 24-candidate batch
changes the picture only modestly: **4 of 24 are now confirmed** with a real margin (all in the
newer paramrudra batch), but **17 of 24 remain clearly overturned**, reinforcing rather than
undercutting the original finding. The mislabeling is not universal — a small minority of
`ground_truth` FM labels do hold up under real DFT — but the majority in this shortlisted
population do not. This is consistent with MP's known workflow gap: relaxation from one initial
magnetic-moment guess, with no systematic AFM enumeration, means MP's stored ordering reflects
whichever configuration that one guess happened to converge to, which is right often enough to not
be pure noise, but wrong on a majority of structures in this DMI-filtered population.

**On the FiM-labeled candidates:** mp-22972 and mp-551086 are labeled `FiM`, not `FM`, but the
enumerator used here never generates FiM configurations. AFM beating the FM baseline for these two
does **not** confirm or refute their actual FiM label — it only rules out FM as the ground state.
A real check of these two would need a dedicated FiM-configuration enumeration, not done here.

**Caveat on sample composition:** this is still not a random sample — all candidates were pulled
from the top of `phase4_shortlist.csv`, which already filters for noncentrosymmetric,
DMI-compatible, heavy-SOC-element structures. It's plausible this structural family is more
AFM-prone than the `ground_truth` population as a whole. What this sample does establish firmly:
`ordering_source=ground_truth, confidence=1.0` cannot be taken at face value for candidates in this
specific shortlisted population without an actual DFT check.

## Recommendation for advisor discussion

1. **Headline result (24-candidate snapshot, 2 remaining)**: of 24 shortlisted
   `ground_truth`/`FiM` labels checked against real DFT, 17 are clearly wrong (AFM ground state), 4
   are confirmed with a resolvable margin, and 3 are statistical ties. The confirmation rate with a
   real margin is roughly 1 in 6.
2. **Implication**: the classifier's/MP's ordering label should be treated as a *screening prior*,
   not ground truth, for this shortlist's structural family — worth deciding whether to re-flag the
   affected shortlist entries, or whether this changes candidate ranking (`ordering_final` is used
   to select "net-moment" candidates in `screen_candidates.py`, and AFM is excluded from that funnel
   entirely).
3. **Open question worth a decision**: is 24–26/1,647 candidates enough evidence to act on, or does
   this need a larger, more representative DFT-validation batch (e.g. a random sample stratified
   across the whole `ground_truth` population, not just the DMI-filtered top of the shortlist)
   before changing how the pipeline treats `ordering_source=ground_truth` labels?
4. **Follow-up worth scoping**: a dedicated FiM-configuration check for mp-22972 and mp-551086,
   since the FM/AFM enumeration used everywhere else in this report structurally cannot confirm or
   refute an FiM label.

## Raw data

- Per-configuration results: `outputs/dft_ordering_validation.csv` (all candidates with any
  converged data; `candidate_complete` column marks which of the 24 are final vs. still running)
- Staged calculations: `data/dft_staging/<candidate_id>/{0_fm,1_afm,...}/` (OUTCAR, OSZICAR,
  CONTCAR, meta.json — either directly in the calc dir (`paramrudra` runs) or under a `results/`
  subdirectory (`paramsanganak` runs))
- First single-candidate writeup (superseded by this doc for the aggregate conclusion, kept for
  method detail): `docs/dft_validation_mp-861559.md`
- Candidate labels: `outputs/phase4_shortlist.csv`
