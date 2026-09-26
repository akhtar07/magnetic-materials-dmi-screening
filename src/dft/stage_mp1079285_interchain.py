"""Stage interchain FM/AFM tight pairs for mp-1079285 (Tb6FeBi2), appended to the 2026-09-20 farm.

Why: every collinear AFM configuration enumerated for mp-1079285 (1_afm, 2_afm, 3_afm) is the SAME
chain-antiparallel ordering in a different 2x-along-b cell setting (verified 2026-09-20: in each,
the two Fe differ by one b translation), so the tight pair in the 3_afm cell resolves only the
intra-chain J_b (Fe-Fe 4.198 A). The interchain shell -- six neighbours at 8.36-8.37 A along +-a,
+-c and +-(a+c) -- is never flipped, and a spin model with J_b alone is a set of decoupled
Heisenberg chains (no order at any T > 0), which makes a finite-temperature sweep meaningless.

Two 2-Fe supercells of the relaxed FM cell (0_fm/CONTCAR), positions-only relaxation, same tight
settings as Batch B (EDIFF 1e-6, EDIFFG -0.01, ISIF 2):
  2a cell, Fe(0) up / Fe(a) down : flips the +-a and +-(a+c) bonds -> dE = 4 (J_a + J_ac)
  2c cell, Fe(0) up / Fe(c) down : flips the +-c and +-(a+c) bonds -> dE = 4 (J_c + J_ac)
(each bond counted once, E = -sum_<ij> J s_i.s_j; 2 bonds of each flipped type per 2-Fe cell).
Two equations for three couplings: the spin model takes the mean interchain J = (dE_a + dE_c)/16
under the stated assumption J_a ~ J_c ~ J_ac (all three distances agree to 0.01 A).

Usage:
    python stage_mp1079285_interchain.py     -> data/dft_staging_tight/mp-1079285/{afm,fm_in}_{2a,2c}
                                                and src/dft/farm_queue_append.txt
"""
import json
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
    lines = []
    for tag, scale, kmesh in (("2a", [[2, 0, 0], [0, 1, 0], [0, 0, 1]], (2, 6, 3)),
                              ("2c", [[1, 0, 0], [0, 1, 0], [0, 0, 2]], (3, 6, 2))):
        sup = prim.copy()
        sup.make_supercell(scale)
        fe = [i for i, s in enumerate(sup) if s.specie.symbol == "Fe"]
        assert len(fe) == 2
        d = sup.get_distance(fe[0], fe[1])
        assert 8.3 < d < 8.5, d  # the two Fe of the doubled cell are interchain neighbours
        afm = [0.0] * len(sup)
        afm[fe[0]], afm[fe[1]] = 5.0, -5.0
        fmm = [abs(m) for m in afm]
        for name, mm, ordering in ((f"afm_{tag}", afm, "AFM"), (f"fm_in_{tag}", fmm, "FM")):
            i = Incar(inc)
            i["MAGMOM"] = " ".join(f"{m}" for m in mm)
            write_calc(TIGHT / CAND / name, sup, i, Kpoints.gamma_automatic(kmesh), pot, {
                "candidate_id": CAND, "ordering": ordering, "magmoms": mm, "cell_from": f"0_fm/CONTCAR x {tag}",
                "formula": sup.composition.reduced_formula, "staged_batch": "D-2026-09-20",
                "flipped_bonds": {"2a": "+-a, +-(a+c)", "2c": "+-c, +-(a+c)"}[tag],
                "note": "interchain tight pair: EDIFF=1e-6, EDIFFG=-0.01, ISIF=2 in the relaxed FM cell; "
                        "dE(AFM-FM) = 4 (J_x + J_ac), bonds counted once"})
            lines.append(f"dft_staging_tight/{CAND}/{name} std")
        print(f"  {CAND}: staged {tag} pair ({len(sup)} atoms, Fe-Fe {d:.3f} A, k {kmesh})")
    q = Path(__file__).resolve().parent / "farm_queue_append.txt"
    q.write_text("\n".join(lines) + "\n")
    print(f"  queue lines -> {q.relative_to(Path.cwd()) if q.is_relative_to(Path.cwd()) else q}")


if __name__ == "__main__":
    main()
