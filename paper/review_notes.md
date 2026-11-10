# Referee-style review of `paper/main.tex`

*"DFT-Validated Evidence of Ferromagnetic Mislabeling Among DMI-Candidate Materials in the Materials Project Database"*

Reviewed 2026-11-09 against `main.tex` (line numbers refer to it), `outputs/dft_verdicts.csv`,
`outputs/dft_tight_validation.csv`, and the staged VASP inputs under `data/dft_staging*`.
Items marked **[checked]** were verified against the repo data; items marked **[question]** are things a
referee will ask that the repo could not settle.

**Overall assessment:** the DFT bookkeeping is careful and the arithmetic holds up (I re-derived every J, zJ,
T_c^MF and the funnel counts, and they are internally consistent). The problems are in **what the numbers are
claimed to mean**. As written, a computational-magnetism referee would recommend **major revision**. The
headline is weaker than the title suggests, two of the three "confirmed ferromagnets" have physics problems,
and the DMI number for mp-754090 is not the DMI.

---

## A. Major issues

### A1. For half the batch, MP's "FM label" is forced by the cell, not a mislabel [checked]
MP's FM-initialized workflow runs in the primitive cell. **A primitive cell with one magnetic ion cannot represent
any AFM order**, so MP's FM label in that case only means "magnetic", not that FM is the ground state. From
`data/dft_staging/*/0_fm/POSCAR`, **13 of the 26 have one magnetic ion per MP cell**:
*(Corrected 2026-11-09 against MP's stored cells: **11** have a single Fe site. mp-1047414 and mp-755203 hold 2 Fe in MP's cell. The 11 are 1079285, 1216981, 1217456, 1218521, 1228547, 1519427, 1520073, 1522159, 22972, 682554 and 850225. Of these, 8 lose their net moment, 2 stay FM and 1 is degenerate.)*
mp-1047414, 1079285, 1216981 (Ti d⁰), 1217456, 1218521, 1228547, 1519427, 1520073, 1522159, 22972, 682554,
755203, 850225.
(This count uses the cells the authors staged as MP's primitive cell. Confirm it against MP's own `nsites` for
each entry. Where MP's stored cell holds more than one magnetic ion, the objection weakens to "trapped in the FM
local minimum".)
The paper never says this. "Mislabeling" (title, l.33; l.448 "wrong more often than right") is the wrong word for
these cases. It is a known limitation that MP and Horton et al. (2019) document themselves. **Fix:** split the
26 into "single-magnetic-ion cell (AFM unrepresentable by MP)" and "multi-ion cell (AFM representable but not
found)". Report the overturn rate in each group. Retitle around "unresolved" or "untested" ordering instead of
"mislabeling".

### A2. The selection makes the AFM outcome expected [checked]
23/26 are oxides, and essentially all are Fe oxides: formal Fe²⁺ (d⁶; |m|≈3.74 μB in CaLaTaFeO₆, Zn₂FeWO₆,
Ta₂FeO₆, Ba₄Ta₁₀FeO₃₀) or Fe³⁺ (d⁵; |m|≈4.2–4.4 μB). Goodenough–Kanamori–Anderson (GKA) rules predict AFM
superexchange for both. The paper admits this (l.180–185) but still presents the 21/26 as a finding about MP. A referee will say
the result is "FM-initialized calculations on Fe oxides miss the AFM state", which is already well known (Horton
2019; Frey 2020). **Fix:** frame it as a quantification within this family. Give a binomial CI (21/24 resolved
gives roughly 70–96%, 95% Wilson interval). State the effective N: three (A,A′)FeBiO₆ variants with identical
margins (−11.7), two Ta₂FeO₆, two TaFeO₄ and two Tb₆FeBi₂ polymorphs leave about 20 independent chemistries.

### A3. FiM bookkeeping is internally inconsistent [checked]
- **mp-551086** is labelled **FiM by MP**, and its deciding DFT configuration is `5_fim` (a ferrimagnetic Fe/Cr
  cell). A FiM-labelled compound whose DFT ground state is FiM has been *confirmed*, not overturned. It is still
  counted in the 21 (l.178–180).
- **mp-775194** (FM → `8_fim`) keeps its net moment. The screening "treats FM and FiM identically" (l.118–119,
  SI Fig. S1 caption), so this overturn does not break the net-moment premise.
- l.170 and Table 1 say "21 have their **FM** label overturned", but two of the 26 labels are FiM.
- Discussion l.416–419 ("net-moment premise is wrong" for 21 of 26) is therefore an overstatement. At most
  **19** candidates actually lose their net moment.
**Fix:** use two outcome columns, "net moment lost (→AFM)" and "net moment kept (FM↔FiM)", and recount.

### A4. The mp-754090 "DMI" is a mix of D and symmetric off-diagonal exchange, and is not the chiral DMI [checked: only `canted_plus`/`canted_minus` exist]
With S₀ = ẑ and S₁ = ±x̂, E(−90°) − E(+90°) = 2 Σ_b J_b^{zx}. The off-diagonal element J^{zx} = Γ^{zx} + D^y
contains the **symmetric anisotropic exchange Γ** as well as D. Γ also changes sign between ±x̂, so it does
**not** cancel (l.648–669 assumes it does). Separating D needs the reversed probe (S₀ = x̂, S₁ = ±ẑ):
D^y ∝ (J^{zx} − J^{xz})/2. Without it, "D_y = 0.097 meV, D/J ≈ 3.5%" (abstract, l.274) may be mostly Γ.

Even the D part of that number is the *uniform* (weak-ferromagnetism-type) combination Σ_b D_b. The quantity that
stabilizes spirals and skyrmions is the **spiralization** Σ_b D_b ⊗ r_b, which is odd in the bond vector. The paper
finds that Σ_b ê_b ≈ 0.03 and that the Lifshitz-type `spin/dmi` term gives **0.000 meV** (l.736–742). That result
shows the probe measures the wrong DMI component for this paper's purpose. Fix both "3.5%" and "five times larger
than mp-1079285" (l.296), or drop them. That comparison also sets a projection against a magnitude.
(The Tb₆FeBi₂ 90° spiral, l.698–709, is fine: with one Fe per cell the Γ contributions cancel around the
spiral, and the spiral is the right probe.)

### A5. FeP₆(WO₈)₃ (mp-850225): unphysical exchange, and it is a ferrimagnet
- **J₁ = 18.4 meV and J₂ = 10.3 meV at Fe–Fe = 8.8–8.9 Å** (l.310–315). On this unit-spin footing
  (J_unit = J_S·S²), these are comparable to *nearest-neighbour* 180° Fe³⁺–O–Fe³⁺ superexchange in LaFeO₃
  (~30 meV). Here, though, the distance is about four times larger, there is no shared ligand, and the sign is
  ferromagnetic. Super-super-exchange through a phosphate–tungstate framework is one to two orders of magnitude
  weaker than this. They imply T_c^MF = 6(J₁+J₂)/3 ≈ 57 meV
  ≈ **660 K** for a dilute Fe phosphate-tungstate. The likely cause is U_W = 0 (l.613–624). The W⁵⁺ d¹ electron
  delocalizes and mediates a metallic, double-exchange-like coupling. **Request:** the FM DOS/band gap with and
  without U_W, and the margin with U_W restored (or with a moderate U_W) once SCF is stabilized.
- By the paper's own description (l.214–216, W moments antiparallel to Fe, 2.0 vs 4.4 μB/f.u.) the ground state
  is a **collinear ferrimagnet**, yet it is counted as "FM confirmed". The same rule counts mp-551086 and
  mp-775194 as overturns. Count it consistently: 2 FM + 1 FiM, or compute the W-parallel FM state and report it.

### A6. Li₄Fe₂TeWO₁₂ (mp-754090): charge balance, and "confirmed" rests on one AFM state [checked]
- With Li⁺, Te⁶⁺, W⁶⁺ and O²⁻, the formula needs **Fe⁴⁺** (or O holes). The DFT Fe moment (4.12 μB) sits between
  Fe³⁺ and Fe⁴⁺. A referee will ask whether this hypothetical compound (P1 cation ordering of a rock-salt) is
  chemically sensible before a T_c is computed for it. Report the band gap or metallicity and the O moments.
- `dft_verdicts.csv`: **n_calc = 2** for mp-754090, meaning FM plus a single AFM (Fe0-reversed in the primitive
  cell). On a 12-neighbour two-sublattice lattice no supercell stripe or layer orders were tested. "Confirmed FM"
  needs at least the in-plane-stripe and layer-AFM supercells. The single-J model built from this pair cannot
  test its own ground state (circular).
- The P1 symmetry, which is what makes it a "DMI candidate", is an artefact of MP's ordered small-cell cation
  arrangement. A real Li/Fe/Te/W rock-salt is likely cation-disordered, and centrosymmetric on average.

### A7. Tb₆FeBi₂: 4f in the core makes the T_c a property of a fictitious model
Tb³⁺ (≈ 9 μB, 6 of 9 atoms) is frozen in the core (l.443, l.675). In the R₆FeBi₂ family the rare-earth sublattice
dominates the magnetism, so a Fe-only model with |m_Fe| ≈ 1.6 μB cannot stand for the material. The abstract states
"139 ± 3 K" with no qualifier. The same objection applies to the verdicts for **both** Tb₆FeBi₂ entries, including
the claim that the experimental compound is "AFM in DFT" (l.258–261). Also:
- |m_Fe| changes from 1.68 μB (FM) to 1.50 μB (spirals) and 1.52–1.64 μB (AFM cells). A 10% longitudinal change
  means a rigid-moment Heisenberg mapping of an itinerant Fe moment is questionable. Discuss this.
- No U on Tb₆FeBi₂ **[checked: INCARs carry MAGMOM but no LDAU]**, which is correct for an intermetallic.

### A8. Symmetry filter: 4 of 26 cannot host bulk chiral DMI [checked: point groups computed with pymatgen]
Being noncentrosymmetric is necessary but not sufficient for bulk Lifshitz invariants. The classes **T_d (−43m),
C_3h (−6) and D_3h (−6m2)** have none. In the batch:
- BaCaFeBiO₆, SrCaFeBiO₆, BaSrFeBiO₆: **F−43m (T_d)**
- Tb₆FeBi₂ mp-22972, the only experimental intermetallic and the one the paper recommends for follow-up
  (l.265): **P−62m (D_3h)**

Also, l.130–131 "DMI is symmetry-forbidden under an inversion centre" is too strong. Bond DMI is forbidden only when
an inversion centre sits at the bond midpoint; centrosymmetric hematite has DMI and weak ferromagnetism. What
inversion forbids is the *net* Lifshitz invariant. **Fix:** filter on the point group, i.e. exclude
T_d/C_3h/D_3h, and correct the sentence.

### A9. Precision claims for T_c are overstated
- Abstract "101 ± 1 K" and "139 ± 3 K": the ± is the scatter of three Binder crossings only. The paper itself
  says the J spread shifts T_c by ~8 K (l.364) and by ~14% for J_b. SI Note 3 shows the 2048×4000 crossing for
  mp-1079285 moves 135.6 → 123.4 K under a window change, and the U₄ autocorrelation time is comparable to the run
  length. Quote about 100 ± 10 K and 140 ± 20 K (model), or move the precise numbers out of the abstract.
- Classical unit spins: for S = 5/2 (or S = 2 for Fe⁴⁺) the classical-vs-quantum T_c correction
  (S² vs S(S+1) normalization ≈ 30–40% at mean-field level) is never mentioned. It is larger than every quoted
  error bar.
- For mp-754090 the model is a single-J, 12-neighbour lattice whose T_c is a textbook number
  (k_BT_c ≈ 3.18J, l.362). The finite-size-scaling campaign reproduces a known result. That is fine as a
  validation, but it should not be presented as a finding. Langevin LLG at α = 1 is also a poor sampler for
  Binder crossings. Heat-bath/Wolff MC with histogram reweighting is standard and much cheaper.
- `chen1993` (Chen, Ferrenberg & Landau) is the **simple-cubic** study. Check the source of the fcc ratio
  (l.363, l.378) **[question]**.

### A10. The 0.10 meV/atom threshold is below the measured protocol noise
The loose-vs-tight difference is up to 1.3 meV/atom (l.600). EDIFF (l.166–167, l.608) is an SCF stopping
criterion, not an error bar on k-mesh, Pulay or cell-shape effects. "Exactly degenerate (0.001 meV/atom)" for the
90-atom Ba₄Ta₁₀FeO₃₀ cell (l.206–207) reports noise as a result. **Fix:** estimate the tight-protocol noise
empirically (one tight pair repeated with a denser k-mesh and ENCUT) and set the window from that. Expect the
sub-0.6 meV/atom overturns (Zn₂FeWO₆ −0.19, Ta₂FeO₆ −0.31/−0.41, mp-682554 −0.31, mp-22972 −0.46,
Li₄NbFe(WO₆)₂ −0.54) to become "marginal".
Also report margins **per magnetic ion**, not per atom. Per atom, dilute-Fe cells look degenerate even when the
per-Fe energy is meaningful, and vice versa. Discuss the U(Fe) sensitivity too (J ∝ t²/U; only U = 5.3 eV was
used).

---

## B. Moderate issues

- **B1. The W Hubbard-U statement is wrong in count and in physics (l.613–617) [checked].** The batch has
  **eight** W compounds, not five. The staged INCARs show that **three kept U_W = 6.2 eV and converged**: Zn₂FeWO₆
  (mp-1047414), Ba₃Fe₂WO₉ (mp-1228376) and Ca₃Fe₂(WO₆)₂ (mp-1301722). Five ran with U_W = 0 (LDAUL = −1):
  mp-754090, 755203, 769471, 775194 and 850225. The batch therefore mixes two W protocols, and "all
  tungsten-bearing candidates failed" is false. Those three U_W = 6.2 convergences, plus MP's own converged
  entries (whose relaxed cells are used here), also contradict the physical explanation. "A large U on an empty d shell gives no restoring force" is not
  correct physics. Say it empirically: "did not converge with our settings". Also note that 2 of the 3
  confirmed FMs (mp-754090, mp-850225) come from this modified recipe (U_W = 0, ISIF = 2 at MP's FM cell,
  ISMEAR = 0). The confirmations therefore are not under "MP's own recipe" (abstract framing).
- **B2. Inconsistent description of the batch selection.** "Drawn from the top of the shortlist" (l.244, l.432),
  "ranks 1–663, not the most stable" (l.142–144) and "top-30 shortlist is the pool" (l.550) do not agree as
  written. Say explicitly that the batch is the 26 MP entries in the top-30 of the composite ranking (the other 4
  were JARVIS), and explain why that ranking differs from the hull ranking in SI Fig. S2.
- **B3. The classifier and GNN section is disconnected from the main result and partly circular.** It is trained
  mostly on MP labels that the paper calls FM-biased. It does not select the 26 (they carry MP labels). Records are
  "not deduplicated across sources" (l.462–463), so the same compound in MP/AFLOW/OQMD/2DMatPedia can sit on both
  sides of the split. Ask for a formula/structure-overlap check. MAGNDATA balanced accuracy is 49–56% (l.123). A
  referee will ask why a 4-class model near that level is in a Nat. Commun. paper at all. Consider moving it to
  the SI.
- **B4. Ground states are only collinear.** For a paper about DMI candidates, the true ground state may be a
  spiral. The enumeration cannot see it, so "confirmed FM" means FM among the enumerated collinear states. Say so.
- **B5. mp-682554.** The 0.31 meV/atom (40 meV/cell) AFM preference between Fe sites 27.9 Å apart in a metallic
  Chevrel-like sulfide needs discussion (RKKY through Re/Mo?). It should not just be counted.
- **B6. Novelty vs Horton 2019 and Frey 2020.** That FM-initialized high-throughput runs miss AFM ground states is
  stated in the MP magnetic-ordering paper itself (l.81–84). The new content is the DMI-family slice and the
  DFT → spin-model chain. Lead with that.
- **B7. "MP rarely tests AFM" (l.47) is unquantified [question].** Check whether any of the 26 MP entries were
  computed with MP's magnetic-ordering workflow or are now r2SCAN entries. If so, "MP's own recipe (GGA+U)" is not
  what MP currently serves.

## C. Minor issues and presentation

- l.140 and l.554: `ordering_source = ground_truth, confidence = 1.0` are **your** schema fields, not something
  "recorded by MP". Reword.
- l.540: MACE was "fine-tuned on the harmonized dataset", but that dataset has no energies or forces. State the
  actual fine-tuning data.
- l.131: the heavy-SOC list omits Pb, Tl, Pd, lanthanides, Te/I and others. Justify the list or call it a choice.
- `fahmy2026`: cited as arXiv:2509.05909v2 (2026), but that is a Sept-2025 identifier. Verify the "~7,800"
  figure against the preprint **[question]**.
- ISMEAR = −5 (tetrahedron) during relaxations of metals (Tb₆FeBi₂, mp-682554) gives inaccurate forces. It is
  inherited from MPRelaxSet, but worth one sentence.
- Tight pairs are relaxed in the **AFM**-relaxed cell, a small bias toward AFM. One sentence.
- l.377–378: the mp-1079285 coordination is 2 + 2 + 6 = 10, not simple cubic. The "just below sc ratio" comparison
  is loose.
- Format: Nat. Commun. does not use revtex `prb`. The SI needs a full per-configuration table (cell, k-mesh,
  energies, moments, audit flags), i.e. the content of `dft_verdicts.csv` and `dft_tight_validation.csv`.
- Placeholders remain: funding, author contributions, competing interests (l.832–840).

---

## What I re-derived and found correct
- Funnel counts: 768+537+392+77+53+35+2 = 1,864; MP+JARVIS+OQMD = 1,340; 671 + 524 = 1,195.
- mp-754090: 16J = 44.54 → J = 2.78; zJ = 33.4; T_c^MF = 129 K; 3.18J → 103 K; D = 1.546/2/8 = 0.097.
- mp-1079285: J_b = 14.61, J_⊥ = 2.795, J_2b = 3.31, zJ = 52.6, T_c^MF = 203 K; same-cell variant 12.56 / 4.34 /
  50.5 / 196 K.
- mp-850225: J₂ = 10.27, J₁ = 18.43; 123.2 meV / 68 atoms = 1.81 meV/atom.
- Binder limits 2/3 and 4/9 for n = 3; 92/138 centro/non-centro space groups.

## Suggested priority for revision
1. Recount (A1, A3, A5) and retitle away from "mislabeling".
2. Reverse canting probe for mp-754090 (2 noncollinear single points), or drop its D (A4).
3. Supercell AFM orders for mp-754090; U_W sensitivity for mp-850225 (A5, A6).
4. Point-group filter fix (A8) and per-Fe margins with an empirical noise window (A10).
5. Soften T_c precision, add the classical-spin caveat, and qualify Tb₆FeBi₂ in the abstract (A7, A9).


---

## Revision status (2026-11-09)

**Applied in `main.tex`:**
- **Title:** reworded; "mislabeling" is removed throughout.
- **Outcome classes:** four classes replace overturned/confirmed.
  - Net moment lost: 18.
  - Kept as FiM: 3 (mp-775194, mp-769471, mp-850225; the last has Fe FM and W5+ antiparallel at U_W=0).
  - Consistent with label: 3 (mp-754090 and mp-1079285 FM; mp-551086 FiM).
  - Degenerate: 2.
  - Single-Fe count of 11 verified against MP's stored cells for all 26.
- **Where the counts live (update these when tb_k* / w850 return; all marked `% PENDING`):**
  - abstract; intro "18 ... and 3 more"; verdict paragraph and single-Fe paragraph (8/11, 10/15, 14/21 and Wilson CIs); Table 1; Fig. 2 caption
  - Discussion "18 of 26 (14 of 21)" and "fewer than a third"; provenance sentence on mp-22972
  - `src/figures/make_paper_figures.py` NET_KEPT_FIM / FIM_LABEL_CONSISTENT; the banner and panel 3 of the graphical abstract (regenerate both scripts)
- **Single-Fe split:** 8/11 vs 10/15, both with Wilson CIs. Effective N is 14/21 (A1, A2, A3).
- **Table 1:** rebuilt with a "Single Fe" column.
- **Inversion sentence:** corrected. Added the Lifshitz-invariant caveat with the 901/1,864 count and the 4 of 26 in the batch (A8).
- **Heavy-SOC list:** now justified as a stated choice (C).
- **Discussion:** collinear-only caveat (B4); classical S(S+1)/S² caveat; T_c quoted as ~100 K and ~140 K, with the J spread folded in (A9).
- **mp-682554:** discussed (B5). The nearest Fe–Fe in the converged 130-atom 3_afm cell is 6.8 Å (the a axis), not 27.9 Å. The "27.9 Å between the two Fe sites" claim was removed from Methods.
- **Novelty vs Horton 2019:** now stated (B6).
- **Methods:**
  - schema wording (C)
  - MACE fine-tune data (C)
  - leakage numbers (B3)
  - batch selection (B2)
  - AFM-cell bias, ISMEAR −5 on metals, U_Fe sensitivity (C, A10)
  - W-U paragraph rewritten empirically, 8 W compounds with 3 converged at U_W=6.2 (B1)
  - mp-754090 charge balance, metallicity, and the caveat that only one AFM ordering was tested (A6)
- **fcc T_c reference:** now ritchie1972. The mp-1079285 coordination is given as 10 (A9, C).

**Running on PARAM Rudra (`/scratch/nsmext/somnathb.iitk/rev2611`); paper text marked `% PENDING`:**
- `tb_k3`, `tb_k2`: Tb6FeBi2 k-convergence of the opposite mp-22972 / mp-1079285 verdicts.
- `dmi/*`: reversed and other-plane probes that separate D from Γ for mp-754090 (A4).
- `w850/fm`, `w850/fim`: mp-850225 at U_W=6.2 in MP's cell (A5).
- `noise/*`: k×1.5 + ENCUT 650 noise floor of the two smallest tight margins (A10).
- `sc754/*`: mp-754090 2a and 2c doubled cells, layer AFM plus an FM twin in each, statics on the tight FM CONTCAR (A6).

**Not done:**
- Point-group filter in the funnel (A8). The Lifshitz caveat is stated instead.
- Moving the classifier to the SI (B3).
- Journal format and placeholders.

### Revision results as they arrive (2026-11-09)
- `dmi/zx_rev`: E(x,+z) = -126.068359634, E(x,-z) = -126.069779295 eV, so ΔB = +1.420 meV against the
  published ΔA = E(z,+x)-E(z,-x) = -1.546 meV. Convention D·(s0×s1) = -(ΔA-ΔB)/32 and Γ = -(ΔA+ΔB)/32
  (8 bonds) give **D_y = 0.0927 meV** and Γ_zx = 0.0039 meV. The published 0.097 meV was 96 % D_y.
- `dmi/zy`: E(z,+y) = -126.069241755, E(z,-y) = -126.069353396, E(y,+z) = -126.068866913,
  E(y,-z) = -126.069565318 eV, so ΔA = +0.112 and ΔB = +0.698 meV. z×y = -x gives **D_x = -0.0183 meV**,
  and Γ_yz = -0.0253 meV.
- `w850`: both U_W=6.2 runs (W parallel / antiparallel, linear mixing AMIX 0.1) diverged after the
  NELMDL=-12 fixed-density start, with rms(c) > 100 by step 28-29. The watchdog killed them. Recorded as not converged.
- `tb_k3` chunk 1 was stopped by STOPCAR at 16-18 steps, not converged. static_chunk.sh mis-marked these
  runs DONE because VASP deletes STOPCAR. Fixed: OUTCAR "hard stop was set" now counts as stopped. Chunk 2 is job 117745.
- `dmi/xy`: E(x,+y) = -126.068126750, E(x,-y) = -126.070191543, E(y,+x) = -126.070076904,
  E(y,-x) = -126.068195010 eV, so ΔA = +2.065 and ΔB = -1.882 meV. **D_z = -0.1233 meV**, Γ_xy = -0.0057 meV.
  **Uniform D = (-0.018, 0.093, -0.123) meV, |D| = 0.155 meV**, dominated by D_z. The published y-only number
  missed the largest component.
- `tb_k3` chunk 2 (job 117745): hit the wall time (TIMEOUT 21:48) while writing WAVECAR, so the WAVECARs may be
  truncated and no resubmit happened. Not converged (rms 0.002-0.02). The FM runs at the denser mesh have
  **lost the Fe moment** (Fe -0.03/-0.03 in p1079285_fm, -0.39/-0.27 in p22972_fm), while the AFM runs keep
  ±1.6-1.65 μB (as in the tight runs, where FM had -1.66 μB on Fe). Current energies: p1079285 AFM -94.4325 vs
  collapsed FM -94.3001; p22972 AFM -94.4274 vs FM -94.3175 eV. The FM-vs-AFM comparison at k3 is not like-for-like yet.
- Restart (2026-11-09 ~22:05): the 3 WAVECARs written at the TIMEOUT were deleted; p1079285_afm and both FM runs start fresh.
  The FM runs have **NUPDOWN fixed** at the tight-FM total moment (1.03 for p1079285 2a, 0.86 for p22972). In chunk 1
  (fresh start) Fe was -2.1 μB, so the collapse came from the WAVECAR restart. Fixed-moment FM is an upper bound on the FM
  branch: if it lies below AFM, FM wins. STOP_MIN lowered to 40 (job 117776). noise/* is job 117777.
- Queue priority after the advisor: tb_k3 (critical path) > noise > sc754 > tb_k2. tb_k2 runs only if a slot is free
  or the k3 margin is < 0.3 meV/atom.
- The DMI vector is written into the paper (Results, Methods, Fig. 3c, abstract, graphical abstract);
  outputs/mp754090_dmi_vector.json holds the numbers.
- `noise/*` (job 117777, all four converged to EDIFF): tight margins with k ×1.5 and ENCUT 650 are mp-1043970 −0.0975
  (unchanged) and mp-1047414 −0.1842 (was −0.1852) meV/atom. Numerical noise ≤ 0.001 meV/atom. The 0.10 window is kept as
  a functional/U allowance; CaLaTaFeO6 stays degenerate (stated in Methods).
- Found and fixed: four `% PENDING` comments had been written onto the same line as paper text, which hid the abstract
  single-Fe sentence, "Degenerate: 2", the Fig. 2 caption marker sentence and the provenance "The funnel's symmetry ..."
  from the PDF.
- sc754 submitted as job 117798 (STOP_MIN 40).
- sc754 done, all four statics converged. 2a: FM −251.723156, AFM −251.583054 eV, +140.1 meV per 40-atom cell
  (3.50 meV/atom). 2c: FM −251.724219, AFM −251.571213 eV, +153.0 meV per cell (3.83 meV/atom). FM twins match
  2× primitive FM (−251.72254) within 2 meV. FM stays the ground state. The single-J model predicts 24J = 67 meV
  per cell, so the same-sublattice bonds are FM and ~12–14 meV (nearest-12 only, unrelaxed AFM). Written into
  Results, Methods and Discussion; the "weakest FM verdict" clause is gone; the model T_c is now called a low estimate.
- tb_k3 chunk 4 (job 117878): both FM WAVECAR restarts blew up at step 1 (E = −170 and −112 eV), while the AFM restart
  was healthy. Cancelled. FM runs restarted fresh with the same NUPDOWN, and static_chunk.sh now restarts from CHGCAR
  (LCHARG on, ICHARG=1, ISTART=0). Resubmitted as job 117892. The chunk-2 FM endpoints (−94.348, −94.370 eV, unconverged)
  are not used.
- 2026-11-10 02:30: p1079285_afm "DONE" at −94.30154 eV is nonmagnetic (Fe ±0.01 μB), so it is discarded (moved to
  nonmagnetic_20261110/). Moment history: chunk 2 (WAVECAR restart) reached −94.4325 with Fe ±1.65, matching
  p22972_afm (−94.42743, ±1.57, converged). The 22:43 fresh restart threw that away, and the next WAVECAR restart
  collapsed it. Fixed-NUPDOWN fresh FM runs report magnetization 1.34/1.19, not 1.03/0.86, so the constraint was
  not holding. The debug QOS (4 submitted / 2 running) is full with other users' jobs on the shared account, and
  the auto-resubmit failed (QOSMaxSubmitJobPerUserLimit). New plan: one uninterrupted 10-node terai job (117931,
  tbk3r.sh): p22972_fm on 4 nodes, p1079285_fm and p1079285_afm on 3 each, NUPDOWN released, no restarts.
  Verdict reference: p22972_afm −94.42743 eV.
- terai job 117931 started 03:59, then OOM-killed in 10 s: no --mem, so the terai default per-CPU memory applied. Added --mem=180G and resubmitted as 117995.

## 2026-11-10 — Tb6FeBi2 k-convergence result and paper update

- Job 117995 (terai, 10 nodes, uninterrupted) converged the three remaining k3 statics; p22972_afm from the chunked
  tb_k3 run (moments checked per chunk). Uniform ~60 A k-length (15x8x5 mp-22972, 4x15x8 mp-1079285), tight CONTCARs:
  - mp-22972: FM -94.448272, AFM(3_afm) -94.427430 eV -> FM by **+1.158 meV/atom** (tight at 6x2x1 was -0.46).
  - mp-1079285 2a: FM -94.456329, AFM -94.432671 eV -> FM by **+1.314 meV/atom** (tight 1.249).
  - Fe 1.57-1.69 muB in all four; Tb ~0.05 muB antiparallel to Fe in mp-22972 (consistent with its FiM label).
- mp-22972 moves from "lost" to "consistent". New counts: lost 17 / kept FiM 3 / consistent 4 / degenerate 2;
  single-Fe 7/1/2/1; tight 7/1/4/2. Wilson: 17/26=0.65 (0.46-0.81), 7/11 (0.35-0.85), 10/15 (0.42-0.85),
  13/21 compositions (0.41-0.79).
- Symmetry: tight relaxations of both entries are P-62m (<=0.0065 A); in P-62m the chain-bond D = 0
  (chain on the 3-fold axis, bond midpoint on sigma_h). The published |D| = 0.098 meV came from the loose ISIF=3
  geometry (Pm residual 0.029 A max, 0.1% cell strain): removed as a physical claim, kept in Methods as a
  loose-relaxation artifact. Fig. 3b D-arrow and Fig. 3c |D| marker removed; abstract/GA ratio "0.7%" removed.
- J_perp at k3 would be ~2.94 meV (+5%, inside the 14% spread); spin model not rerun.
- mp-682554 (metallic, -0.31, k-length 20-28 A) was NOT k-tested; stated in Methods.
- Provenance subsection retitled; the ICSD-backed Tb6FeBi2 keeps its moment but is in -6m2 (no Lifshitz invariant).
- 2026-11-11: intro now counts four FM-Fe-sublattice candidates (three compounds; the two Tb6FeBi2 entries are one).
  Classifier hold-out table recounted against the current verdicts (predictions unchanged; compare_models.py now reads
  outputs/dft_verdicts.csv): 9/20, 12/20, 6/20, CGCNN 9-12/20 non-FM; 2/4, 2/4, 2/4, 0-1/4 FM; base-rate ratio 1.5-2.7.
