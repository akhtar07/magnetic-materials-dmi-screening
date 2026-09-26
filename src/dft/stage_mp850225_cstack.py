"""Stage the layer-stacked (along the hexagonal c axis) tight FM/AFM pair for mp-850225
(FeP6(WO8)3, R3), appended to the live 2026-09-21 farm (job 72658) on its idle nodes.

Why: the Fe sublattice is a rhombohedral Bravais lattice (one Fe per primitive cell, site symmetry
3.) with two shells: six in-plane neighbours at 8.772 A (hexagonal-layer bonds, primitive vectors
a_i - a_j) and six out-of-plane neighbours at 8.912 A (+-a_1, +-a_2, +-a_3, linking adjacent
layers). The existing tight pair (afm_2_afm / fm_in_2_afm, Batch B) lives in the 2-Fe cell with
supercell matrix [[0,-1,1],[1,0,0],[-1,1,1]], whose sublattice parity is (n_2 + n_3) mod 2: it flips
4 of the 6 in-plane and 4 of the 6 out-of-plane bonds per cell, so with E = -sum_<ij> J s_i.s_j
(each bond counted once)
    dE_2afm = 2 (4 J_1 + 4 J_2) = 8 (J_1 + J_2)         -> J_1 + J_2 = 229.64 / 8 = 28.7 meV
and J_1 (in-plane) and J_2 (out-of-plane) cannot be separated. The cell staged here,
    A_1 = a_1 - a_2,  A_2 = a_2 - a_3,  A_3 = 2 a_1     (det 2, 68 atoms),
has sublattice parity (n_1 + n_2 + n_3) mod 2, i.e. alternate hexagonal layers along c: every
in-plane bond (n_1 + n_2 + n_3 = 0) stays parallel and all six out-of-plane bonds flip,
    dE_cstack = 2 * 6 J_2 = 12 J_2,
which gives J_2 directly and then J_1 from the Batch-B pair. Both members use the relaxed FM
primitive cell (0_fm/CONTCAR) with positions-only relaxation and the Batch B/D tight settings
(EDIFF 1e-6, EDIFFG -0.01, ISIF 2). The FM member also checks cell/k-mesh consistency against
fm_in_2_afm (same 68 atoms, per-atom energies should agree to ~0.1 meV/atom).

Why this matters for the DMI probe: with one Fe per primitive cell a two-sublattice canting cannot
see D (the +t and -t bonds contribute D.(S0xS1) and -D.(S0xS1)); only a spiral does, and a spin
model for the spiral needs J_1 and J_2 separately.

Usage:
    python stage_mp850225_cstack.py   -> data/dft_staging_tight/mp-850225/{afm,fm_in}_cstack
                                         and src/dft/farm_queue_append6.txt
"""
import sys
from pathlib import Path

import numpy as np
from pymatgen.io.vasp import Incar, Kpoints, Poscar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_rerun_batches import TIGHT, STAGING, TIGHT_TAGS, base_incar, best_file, write_calc  # noqa: E402

CAND = "mp-850225"
SUPERCELL = [[1, -1, 0], [0, 1, -1], [2, 0, 0]]


def main():
    fm = STAGING / CAND / "0_fm"
    prim = Poscar.from_file(best_file(fm, "CONTCAR")).structure
    assert sum(1 for s in prim if s.specie.symbol == "Fe") == 1
    inc = base_incar(CAND)
    inc.update(TIGHT_TAGS)
    pot = (fm / "potcar_symbols.txt").read_text()

    sup = prim.copy()
    sup.make_supercell(SUPERCELL)
    fe = [i for i, s in enumerate(sup) if s.specie.symbol == "Fe"]
    assert len(fe) == 2 and len(sup) == 68
    # the two Fe differ by one primitive vector (out-of-plane bond, 8.912 A): distinct layers
    d01 = sup.get_distance(fe[0], fe[1])
    assert abs(d01 - 8.912) < 0.02, d01
    # bond bookkeeping on the primitive lattice: parity of n1+n2+n3 must flip every +-a_i bond and
    # no in-plane bond
    inv = np.linalg.inv(np.array(SUPERCELL, dtype=float))
    for t, shell in ((np.array(v), s) for v, s in
                     [((1, 0, 0), "out"), ((0, 1, 0), "out"), ((0, 0, 1), "out"),
                      ((1, -1, 0), "in"), ((0, 1, -1), "in"), ((1, 0, -1), "in")]):
        frac = t @ inv                     # translation in supercell fractional coordinates
        same_sublattice = np.allclose(frac - np.round(frac), 0, atol=1e-8)
        assert same_sublattice == (shell == "in"), (t, shell, frac)
    lat = prim.lattice
    d_in = np.linalg.norm(lat.get_cartesian_coords([1, -1, 0]))
    d_out = np.linalg.norm(lat.get_cartesian_coords([1, 0, 0]))
    assert abs(d_in - 8.772) < 0.02 and abs(d_out - 8.912) < 0.02, (d_in, d_out)

    afm = [0.0] * len(sup)
    afm[fe[0]], afm[fe[1]] = 5.0, -5.0
    fmm = [abs(m) for m in afm]
    lines = []
    # 2x2x2 as the Batch-B pair (8.8 x 8.9 x 12.5 A cell); this cell is 8.8 x 8.8 x 17.8 A
    kp = Kpoints.gamma_automatic((2, 2, 2))
    for name, mm, ordering in (("afm_cstack", afm, "AFM"), ("fm_in_cstack", fmm, "FM")):
        d = TIGHT / CAND / name
        if d.exists():
            sys.exit(f"refusing to overwrite {d}")
        i = Incar(inc)
        i["MAGMOM"] = " ".join(f"{m}" for m in mm)
        write_calc(d, sup, i, kp, pot, {
            "candidate_id": CAND, "ordering": ordering, "magmoms": mm,
            "cell_from": f"0_fm/CONTCAR x {SUPERCELL}", "formula": sup.composition.reduced_formula,
            "staged_batch": "E-2026-09-21", "fe_sites": fe, "sublattice_parity": "(n1+n2+n3) mod 2",
            "flipped_bonds": "0 of 6 in-plane (8.772 A), 6 of 6 out-of-plane (8.912 A) per 2-Fe cell",
            "note": "layer-stacked tight pair: EDIFF=1e-6, EDIFFG=-0.01, ISIF=2 in the relaxed FM cell; "
                    "dE(AFM-FM) = 12 J_2, bonds counted once"})
        lines.append(f"dft_staging_tight/{CAND}/{name}")
    print(f"  {CAND}: staged c-stacked pair ({len(sup)} atoms, Fe-Fe {d01:.3f} A, k 2x2x2)")
    q = Path(__file__).resolve().parent / "farm_queue_append6.txt"
    q.write_text("\n".join(lines) + "\n")
    print(f"  queue lines -> {q}")


if __name__ == "__main__":
    main()
