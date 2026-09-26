"""Stage the chain next-nearest-neighbour tight pair for mp-1079285 (Tb6FeBi2), appended to the
live 2026-09-21 farm (job 72658).

Why: the Fe chain along b has its second neighbour at 2b = 8.396 A, the SAME distance as the six
interchain neighbours (8.396-8.407 A along +-a, +-c, +-(a+c)). A radial exchange function (LAMMPS
spin/exchange) cannot separate the two, so the spin model needs J_2b explicitly: either measured or
set to zero by assumption. Neither the 3_afm pair (chain +-+-: 2b bonds parallel) nor the 2a/2c
pairs (2b bonds parallel) nor the spirals (2b bonds antiparallel in BOTH chiralities, so J_2b cancels
in E+ - E-) resolves it. The 1x4x1 cell with the chain pattern + + - - does:
    b bonds : 4 per cell, 2 flipped -> 2 * 2 J_b
    2b bonds: 4 per cell, 4 flipped -> 2 * 4 J_2b
    dE(AFM - FM) = 4 J_b + 8 J_2b   (E = -sum_<ij> J s_i.s_j, each bond counted once)
with J_b from the 3_afm tight pair. Same tight settings as Batch B/D (EDIFF 1e-6, EDIFFG -0.01,
ISIF 2, positions-only) in the relaxed FM cell; k-mesh 3x2x3 on the 1x4x1 cell (the 3_afm pair uses
3x3x3 on 1x2x1, the spirals 3x2x3 on 1x4x1).

Addendum (2026-09-21 17:05, staged by hand on the cluster, mirrored to data/): afm_2b_in_4b, the
+ - + - chain pattern in the SAME 1x4x1 cell / 3x2x3 mesh (INCAR identical to afm_4b except MAGMOM
24*0.0 5 -5 5 -5 8*0.0), partner fm_in_4b: dE = 8 J_b. It gives J_b in the same cell as J_2b, so the
subtraction dE_4b - 4 J_b does not mix two k-meshes (3x3x3 on 1x2x1 vs 3x2x3 on 1x4x1).

Usage:
    python stage_mp1079285_chain_nnn.py   -> data/dft_staging_tight/mp-1079285/{afm,fm_in}_4b
                                             and src/dft/farm_queue_append4.txt
"""
import sys
from pathlib import Path

import numpy as np
from pymatgen.io.vasp import Incar, Kpoints, Poscar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_rerun_batches import TIGHT, STAGING, TIGHT_TAGS, base_incar, best_file, write_calc  # noqa: E402

CAND = "mp-1079285"


def main():
    fm = STAGING / CAND / "0_fm"
    prim = Poscar.from_file(best_file(fm, "CONTCAR")).structure
    assert sum(1 for s in prim if s.specie.symbol == "Fe") == 1
    inc = base_incar(CAND)
    inc.update(TIGHT_TAGS)
    pot = (fm / "potcar_symbols.txt").read_text()

    sup = prim.copy()
    sup.make_supercell([[1, 0, 0], [0, 4, 0], [0, 0, 1]])
    fe = [i for i, s in enumerate(sup) if s.specie.symbol == "Fe"]
    assert len(fe) == 4
    # chain order along b (fractional b of the 1x4x1 cell rises in steps of 0.25)
    fe = sorted(fe, key=lambda i: sup[i].frac_coords[1])
    fb = np.array([sup[i].frac_coords[1] for i in fe])
    assert np.allclose(np.diff(fb), 0.25, atol=1e-3), fb
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        assert abs(sup.get_distance(fe[i], fe[j]) - 4.198) < 0.01
    for i, j in ((0, 2), (1, 3)):
        assert abs(sup.get_distance(fe[i], fe[j]) - 8.396) < 0.01

    afm = [0.0] * len(sup)
    for k, i in enumerate(fe):
        afm[i] = 5.0 if k < 2 else -5.0          # + + - - along b
    fmm = [abs(m) for m in afm]
    lines = []
    for name, mm, ordering in (("afm_4b", afm, "AFM"), ("fm_in_4b", fmm, "FM")):
        d = TIGHT / CAND / name
        if d.exists():
            sys.exit(f"refusing to overwrite {d}")
        i = Incar(inc)
        i["MAGMOM"] = " ".join(f"{m}" for m in mm)
        write_calc(d, sup, i, Kpoints.gamma_automatic((3, 2, 3)), pot, {
            "candidate_id": CAND, "ordering": ordering, "magmoms": mm, "cell_from": "0_fm/CONTCAR x 4b",
            "formula": sup.composition.reduced_formula, "staged_batch": "D-2026-09-21",
            "fe_chain_order": fe, "chain_pattern": "+ + - -",
            "flipped_bonds": "2 of 4 b bonds, 4 of 4 2b bonds",
            "note": "chain NNN tight pair: EDIFF=1e-6, EDIFFG=-0.01, ISIF=2 in the relaxed FM cell; "
                    "dE(AFM-FM) = 4 J_b + 8 J_2b, bonds counted once"})
        lines.append(f"dft_staging_tight/{CAND}/{name} std")
    print(f"  {CAND}: staged 4b pair ({len(sup)} atoms, k 3x2x3)")
    q = Path(__file__).resolve().parent / "farm_queue_append4.txt"
    q.write_text("\n".join(lines) + "\n")
    print(f"  queue lines -> {q}")


if __name__ == "__main__":
    main()
