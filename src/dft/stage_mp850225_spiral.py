"""Stage the spin-spiral DMI probe for mp-850225 (FeP6(WO8)3, R3) in its 102-atom hexagonal cell.

Why a spiral and why this cell: the Fe sublattice is a rhombohedral Bravais lattice (one Fe per
primitive cell), so a two-sublattice canting sees no DMI (the +t and -t bonds cancel; paper
Sec. Limitations). The hexagonal cell A_1 = a_1 - a_2, A_2 = a_2 - a_3, C = a_1 + a_2 + a_3
(det 3) holds one Fe per hexagonal layer, three layers along c. Every out-of-plane neighbour
(+a_i, 8.912 A) of an Fe in layer k lies in layer k+1, every in-plane neighbour (a_i - a_j,
8.772 A) in its own layer. A q || c spiral with phase phi_k = s * k * 120 deg (s = +1 plus,
-1 minus), spins rotating in the ab plane (rotation normal n = c_hat), therefore gives
    in-plane bonds:     dphi = 0        -> no DMI signal, no change in exchange between +q/-q
    out-of-plane bonds: dphi = s * 120  -> S_i x S_j = s sin(120) c_hat for all 9 bonds per cell
(the k=2 -> k=0' bond across the cell boundary is also +120 mod 360), so with
E = sum_<ij> D_ij . (S_i x S_j), each bond counted once, unit spins,
    E(plus) - E(minus) = 2 * 9 * sin(120 deg) * D_z      ->   D_z = dE / (18 sin 120)
for the out-of-plane NN bond. The three bonds out of one Fe are related by the 3-fold axis, so
their in-plane D components sum to zero and the c-component is the only one a single-q probe in
this cell can see; a rotation normal in the ab plane gives zero by symmetry, hence one pair, not
two. Neglected: Fe-Fe bonds two layers apart (>= 15.5 A).

Settings follow the mp-1079285 spiral (I_CONSTRAINED_M=1, LAMBDA=10, ISYM=0, EDIFF=1e-7,
LSORBIT, NSW=0 statics) on the relaxed FM primitive structure (0_fm/CONTCAR, the cell the tight
pairs use) and the tight-pair Hubbard settings of fm_in_cstack (Fe U=5.3, no U on d0 W). This
compound is gapped, so the mixer is left at VASP defaults (memory: Kerker strength scales
inversely with the gap) instead of the metallic AMIX=0.2/BMIX=1e-4 used for Tb6FeBi2.

Usage:
    python stage_mp850225_spiral.py   -> data/dft_staging_noncollinear/mp-850225/spiral_c_{plus,minus}
"""
import re
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_rerun_batches import best_outcar, species_list, write_calc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CAND = "mp-850225"
FM = ROOT / "data" / "dft_staging" / CAND / "0_fm"
TIGHT_REF = ROOT / "data" / "dft_staging_tight" / CAND / "fm_in_cstack" / "INCAR"
OUT = ROOT / "data" / "dft_staging_noncollinear" / CAND
M_FE = 4.4  # |m_Fe| from the collinear runs (spin-state audit: 4.39-4.40 uB)


def hexagonal_cell() -> Structure:
    prim = Poscar.from_file(FM / "CONTCAR").structure
    assert sum(1 for s in prim if s.specie.symbol == "Fe") == 1
    sup = prim.copy()
    sup.make_supercell([[1, -1, 0], [0, 1, -1], [1, 1, 1]])
    # re-express in the standard orientation (A_1 along x, C along z) so SAXIS = 0 0 1 is c
    p = sup.lattice.parameters
    hexa = Structure(Lattice.from_parameters(*p), sup.species, sup.frac_coords)
    assert abs(hexa.lattice.alpha - 90) < 0.01 and abs(hexa.lattice.beta - 90) < 0.01
    assert abs(hexa.lattice.gamma - 120) < 0.01, hexa.lattice.parameters
    c_hat = hexa.lattice.matrix[2] / np.linalg.norm(hexa.lattice.matrix[2])
    assert np.allclose(c_hat, [0, 0, 1], atol=1e-6)
    return hexa


def fe_layers(st: Structure) -> list[int]:
    fe = [i for i, s in enumerate(st) if s.specie.symbol == "Fe"]
    assert len(fe) == 3
    z0 = st[fe[0]].frac_coords[2]
    fe.sort(key=lambda i: (st[i].frac_coords[2] - z0) % 1.0)
    dz = [((st[i].frac_coords[2] - z0) % 1.0) for i in fe]
    assert np.allclose(dz, [0, 1 / 3, 2 / 3], atol=1e-3), dz
    return fe


def check_bonds(st: Structure, fe: list[int]):
    """Every Fe-Fe neighbour within 9.0 A: in-plane (8.772) in the same layer, out-of-plane
    (8.912) one layer up or down, 6 + 6 per Fe."""
    layer = {i: k for k, i in enumerate(fe)}
    for i in fe:
        inplane = outp = 0
        for nb in st.get_neighbors(st[i], 9.0):
            if nb.specie.symbol != "Fe":
                continue
            dl = (layer[nb.index] - layer[i]) % 3
            if abs(nb.nn_distance - 8.772) < 0.01:
                assert dl == 0, (i, nb.index, nb.nn_distance)
                inplane += 1
            elif abs(nb.nn_distance - 8.912) < 0.01:
                assert dl in (1, 2), (i, nb.index, nb.nn_distance)
                outp += 1
            else:
                raise AssertionError(f"unexpected Fe-Fe distance {nb.nn_distance:.3f}")
        assert inplane == 6 and outp == 6, (inplane, outp)


def main():
    st = hexagonal_cell()
    fe = fe_layers(st)
    check_bonds(st, fe)

    ref = Incar.from_file(TIGHT_REF)
    nelect_prim = float(re.search(r"NELECT\s*=\s*([\d.]+)", best_outcar(FM).read_text()).group(1))
    nelect = nelect_prim * 3
    nbands = int(np.ceil(2 * (nelect / 2 + len(st)) / 48) * 48)
    species_order = species_list(st)

    # RWIGS: every pair sum at least 0.1 A below the corresponding minimum distance
    rwigs = {"Fe": 1.20, "P": 0.70, "W": 1.15, "O": 0.65}  # min Fe-O 2.008, P-O 1.495, W-O 1.952
    dm = st.distance_matrix
    for i in range(len(st)):
        for j in range(i + 1, len(st)):
            si, sj = st[i].specie.symbol, st[j].specie.symbol
            assert rwigs[si] + rwigs[sj] < dm[i, j] - 0.1, (si, sj, round(dm[i, j], 3))

    # gamma-centred mesh at the 0_fm density: 3x3x3 on the 8.9 A rhombohedral cell ->
    # 3x3 in-plane on the 8.77 A hexagonal cell is the same spacing, and C (22.0 A) gets 1
    kp = Kpoints.gamma_automatic((3, 3, 1))
    pot = (FM / "potcar_symbols.txt").read_text()
    e0, e1, n = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])

    for chir, sign in (("plus", +1), ("minus", -1)):
        vecs = np.zeros((len(st), 3))
        for k, i in enumerate(fe):
            phi = sign * k * 2 * np.pi / 3
            vecs[i] = np.cos(phi) * e0 + np.sin(phi) * e1
        magmom = " ".join(f"{M_FE*v[0]:.4f} {M_FE*v[1]:.4f} {M_FE*v[2]:.4f}" for v in vecs)
        mconstr = " ".join(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" for v in vecs)
        inc = Incar({
            "PREC": "Accurate", "ENCUT": 520.0, "EDIFF": 1e-7, "NELM": 300, "NELMIN": 6, "ALGO": "Normal",
            "ISMEAR": 0, "SIGMA": 0.05, "ISYM": 0, "LREAL": False, "LASPH": True,
            "IBRION": -1, "NSW": 0, "ISIF": 2, "LWAVE": False, "LCHARG": False, "LORBIT": 11,
            "LNONCOLLINEAR": True, "LSORBIT": True, "GGA_COMPAT": False, "SAXIS": "0 0 1",
            "NBANDS": nbands, "LMAXMIX": 4,
            "I_CONSTRAINED_M": 1, "LAMBDA": 10,
            "RWIGS": " ".join(str(rwigs[s]) for s in species_order),
        })
        for tag in ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LDAUPRINT"):
            inc[tag] = ref[tag]
        d = OUT / f"spiral_c_{chir}"
        write_calc(d, st, inc, kp, pot, {
            "candidate_id": CAND, "formula": st.composition.reduced_formula, "staged_batch": "F-2026-10-20",
            "supercell": "hexagonal [[1,-1,0],[0,1,-1],[1,1,1]] of 0_fm/CONTCAR (102 atoms, 3 Fe, one per layer)",
            "rotation_plane_normal": n.tolist(), "chirality": sign,
            "fe_site_indices_by_layer": fe,
            "spin_directions": {str(i): vecs[i].round(6).tolist() for i in fe},
            "nelect": nelect, "nbands": nbands,
            "extraction": "D_z = [E(plus) - E(minus)] / (2 * 9 * sin(120 deg)) eV per out-of-plane NN bond "
                          "(8.912 A), unit spins; E = sum_<ij> D_ij.(S_i x S_j), i in layer k -> j in layer k+1",
        }, raw_tags={"MAGMOM": magmom, "M_CONSTR": mconstr})
    print(f"staged {OUT}/spiral_c_{{plus,minus}}: {len(st)} atoms, NELECT={nelect:g}, NBANDS={nbands}, "
          f"k 3x3x1, Fe layers {fe}, c={st.lattice.c:.3f} A")


if __name__ == "__main__":
    main()
