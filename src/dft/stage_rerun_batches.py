"""Stage the 2026-09-20 rerun batches that fix / tighten the magnetic-ordering validation.

Three batches, all written locally (no POTCAR data), rsync'd to paramrudra and run as ONE
self-scheduling terai farm (farm_submit_rerun.sh):

  A. proper collinear configurations for the candidates whose earlier "AFM" calcs were invalid
     (audit_spin_states.py): enumerator configs with zeroed magnetic sites (mp-1173143,
     mp-1173139, mp-551086, mp-775194), d0-Ti flips (mp-1216981 -> handled in B), an FM run
     that collapsed to non-magnetic (mp-22972) and an SCF that diverged (mp-682554/2_afm).
     These go into data/dft_staging/<cand>/<k>_<ordering>/ as new calc dirs (k continues the
     existing numbering) and use the compound's OUTCAR-actual INCAR so every calc of a compound
     shares ENCUT/ISMEAR/SIGMA/ISIF/k-density/LDAU.
  B. tight same-cell FM-vs-AFM pairs (data/dft_staging_tight/<cand>/{afm_<calc>,fm_in_<calc>}):
     the earlier protocol relaxed FM and AFM in DIFFERENT cells with EDIFF = 5e-5 eV/atom and an
     energy-based EDIFFG, which is fine for a 10 meV/atom verdict but not for the < 1 meV/atom
     ones, and not for the exchange constants the spin-dynamics step needs. Here both spin
     states are relaxed (positions only, ISIF=2) in the SAME cell -- the relaxed cell of the
     lowest AFM config -- with EDIFF=1e-6, EDIFFG=-0.01, so E_FM - E_AFM is a clean J.
  C. DMI probe for mp-1079285 (Tb6FeBi2, Fe chains along b, 4.20 A): 4x supercell along b with a
     90-degree spin spiral of either chirality, constrained-moment noncollinear+SOC statics
     (data/dft_staging_noncollinear/mp-1079285/spiral_{n1,n2}_{plus,minus}). A 2x cell cannot
     measure D for a one-Fe-per-cell lattice: the bond and its periodic image contribute
     D.(S0xS1) + D.(S1xS0) = 0 exactly, so the 4x q = pi/2b spiral is the smallest cell that
     does. Two rotation-plane normals n1, n2 (both perpendicular to b, because the mirror plane
     of Pm through the bond midpoint forces D_b = 0 by Moriya's rules) give the full D vector:
         D.n = [E(+q) - E(-q)] / (2 * 4 bonds)   (unit spins, per NN bond, eV -> meV)

Usage:
    python stage_rerun_batches.py            # writes everything + prints the farm queue
"""
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar

from prepare_magnetic_ordering_dft import SameSpinComparator

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "data" / "dft_staging"
TIGHT = ROOT / "data" / "dft_staging_tight"
NONCOL = ROOT / "data" / "dft_staging_noncollinear"
REGEN = Path("/tmp/claude-1001/-data-faiz-MagneticMaterialsAI/7785225c-58e6-4ede-a384-4e02dd2054b3/scratchpad/regen")
AUDIT = ROOT / "outputs" / "dft_spin_state_audit.csv"
VALIDATION = ROOT / "outputs" / "dft_ordering_validation.csv"

QUEUE = []  # (est_minutes, relative dir, binary)


def outcar_incar(outcar: Path) -> Incar:
    """The INCAR block VASP echoes at the top of OUTCAR = the tags actually used (the on-disk
    INCAR of several calcs is stale, e.g. mp-754090 predates the tungsten-U fix)."""
    txt = outcar.read_text(errors="ignore")
    m = re.search(r"^ INCAR:\n(.*?)^ POTCAR:", txt, re.S | re.M)
    if not m:
        raise ValueError(f"no INCAR echo in {outcar}")
    return Incar.from_str(m.group(1))


def best_outcar(calc: Path) -> Path:
    return best_file(calc, "OUTCAR")


def best_file(calc: Path, name: str) -> Path:
    """Some calcs were rsync'd back into <calc>/results/, others straight into <calc>/."""
    o = calc / "results" / name
    return o if o.exists() else calc / name


def base_incar(cand: str) -> Incar:
    inc = outcar_incar(best_outcar(STAGING / cand / "0_fm"))
    inc.pop("MAGMOM", None)
    inc["LWAVE"] = False
    inc["LCHARG"] = False
    return inc


def ediff_per_atom(cand: str) -> float:
    fm = STAGING / cand / "0_fm"
    return float(outcar_incar(best_outcar(fm))["EDIFF"]) / len(Poscar.from_file(fm / "POSCAR").structure)


def species_list(struct: Structure) -> list[str]:
    out = []
    for s in struct:
        sym = s.specie.symbol
        if not out or out[-1] != sym:
            out.append(sym)
    return out


def magmom_string(magmoms) -> str:
    """Compact VASP MAGMOM (n*v) in the same site order as the POSCAR."""
    parts = []
    for m in magmoms:
        if parts and abs(parts[-1][1] - m) < 1e-9:
            parts[-1][0] += 1
        else:
            parts.append([1, float(m)])
    return " ".join(f"{n}*{v}" for n, v in parts)


def spin_structure(poscar: Path, magmoms) -> Structure:
    st = Poscar.from_file(poscar).structure
    st.add_site_property("magmom", list(magmoms))
    return CollinearMagneticStructureAnalyzer(st).get_structure_with_spin()


def write_calc(calc_dir: Path, structure: Structure, incar: Incar, kpoints: Kpoints, potcar_symbols: str, meta: dict,
               raw_tags: dict | None = None):
    """raw_tags are written verbatim (pymatgen re-formats list tags into n*v groups, which turns a
    flat 3N noncollinear MAGMOM into unreadable strings like '3*73*0.0')."""
    calc_dir.mkdir(parents=True, exist_ok=True)
    Poscar(structure).write_file(calc_dir / "POSCAR")
    for k in (raw_tags or {}):
        incar.pop(k, None)
    text = incar.get_str(sort_keys=True)
    for k, v in (raw_tags or {}).items():
        text += f"{k} = {v}\n"
    (calc_dir / "INCAR").write_text(text)
    kpoints.write_file(calc_dir / "KPOINTS")
    (calc_dir / "potcar_symbols.txt").write_text(potcar_symbols)
    (calc_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------------------------------
# Batch A: proper collinear configurations
# ----------------------------------------------------------------------------------------------
def existing_valid_spin_structures(cand: str, audit: dict):
    out = []
    for calc in sorted((STAGING / cand).glob("[0-9]*_*")):
        meta = json.loads((calc / "meta.json").read_text())
        if meta.get("superseded_by"):
            continue
        valid, _ = audit.get((cand, calc.name), (True, ""))
        if valid and best_outcar(calc).exists():
            out.append((calc.name, spin_structure(calc / "POSCAR", meta["magmoms"])))
    return out


def next_index(cand: str) -> int:
    idx = [int(p.name.split("_")[0]) for p in (STAGING / cand).glob("[0-9]*_*")]
    return max(idx) + 1


def stage_new_configs(cand: str, audit: dict, est_minutes: float):
    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False, comparator=SameSpinComparator())
    existing = existing_valid_spin_structures(cand, audit)
    base = base_incar(cand)
    ref_species = species_list(Poscar.from_file(STAGING / cand / "0_fm" / "POSCAR").structure)
    k = next_index(cand)
    n_new = 0
    for regen in sorted(REGEN.glob(f"{cand}/[0-9]*_*")):
        meta = json.loads((regen / "meta.json").read_text())
        if meta["ordering"] == "FM":
            continue
        st = spin_structure(regen / "POSCAR", meta["magmoms"])
        flipped = st.copy()
        flipped.remove_spin()
        flipped.add_site_property("magmom", [-m for m in meta["magmoms"]])
        flipped = CollinearMagneticStructureAnalyzer(flipped).get_structure_with_spin()
        dup = [name for name, ex in existing if matcher.fit(st, ex) or matcher.fit(flipped, ex)]
        if dup:
            print(f"  {cand}: regen {regen.name} duplicates existing {dup[0]} -- skipped")
            continue
        plain = Poscar.from_file(regen / "POSCAR").structure
        assert species_list(plain) == ref_species, (cand, regen.name, species_list(plain), ref_species)
        inc = Incar(base)
        inc["MAGMOM"] = magmom_string(meta["magmoms"])
        inc["EDIFF"] = ediff_per_atom(cand) * len(plain)  # same per-atom tolerance as the compound's other calcs
        calc_name = f"{k}_{meta['ordering'].lower()}"
        meta.update({"index": k, "staged_batch": "A-2026-09-20",
                     "note": "proper all-sites-populated configuration replacing the invalid enumerator output"})
        # POTCARs must match the compound's earlier calcs (the newer pymatgen MPRelaxSet picks
        # W_pv where the originals used W)
        write_calc(STAGING / cand / calc_name, plain, inc, Kpoints.from_file(regen / "KPOINTS"),
                   (STAGING / cand / "0_fm" / "potcar_symbols.txt").read_text(), meta)
        existing.append((calc_name, st))
        QUEUE.append((est_minutes, f"dft_staging/{cand}/{calc_name}", "std"))
        k += 1
        n_new += 1
    print(f"  {cand}: staged {n_new} new configuration(s)")


def stage_mp22972(audit):
    """FM state collapsed to NM in the 9-atom primitive (mp-22972/0_fm). Two anti-collapse reruns:
    4_fm = FM in the cell of the valid ground-state AFM calc (3_afm) -- same cell, same k-mesh, so
    it is also the cleanest FM reference for that cell; 5_fm = the primitive again. Both start
    from MAGMOM near the AFM-converged |m| (1.5 uB) instead of 5, with blocked-Davidson and a
    longer non-selfconsistent warm-up, which is what let the AFM runs keep their moments."""
    cand = "mp-22972"
    base = base_incar(cand)
    base.update({"ALGO": "Normal", "NELMDL": -12, "NELM": 200})
    k = next_index(cand)
    for src, tag in (("3_afm", "FM in the 3_afm cell"), ("0_fm", "primitive cell")):
        calc = STAGING / cand / src
        st = Poscar.from_file(calc / "POSCAR").structure
        mm = [2.0 if s.specie.symbol == "Fe" else 0.0 for s in st]
        inc = Incar(base)
        inc["MAGMOM"] = magmom_string(mm)
        inc["EDIFF"] = ediff_per_atom(cand) * len(st)
        meta = {"candidate_id": cand, "ordering": "FM", "index": k, "magmoms": mm,
                "formula": st.composition.reduced_formula, "staged_batch": "A-2026-09-20",
                "note": f"anti-collapse FM rerun ({tag}); 0_fm collapsed to |m|=0.02"}
        write_calc(STAGING / cand / f"{k}_fm", st, inc, Kpoints.from_file(calc / "KPOINTS"),
                   (calc / "potcar_symbols.txt").read_text(), meta)
        QUEUE.append((15 if src == "3_afm" else 10, f"dft_staging/{cand}/{k}_fm", "std"))
        k += 1
    print(f"  {cand}: staged 2 anti-collapse FM reruns")


def stage_mp682554():
    """mp-682554/2_afm never converged (rms(c) limit cycle with the default Kerker mixer on a
    130-atom metallic cell). Same configuration, conservative linear mixing, from the unrelaxed
    POSCAR. 2_afm is marked superseded so the analyzer stops counting it."""
    cand = "mp-682554"
    src = STAGING / cand / "2_afm"
    meta = json.loads((src / "meta.json").read_text())
    k = next_index(cand)
    inc = base_incar(cand)
    inc.update({"ALGO": "Normal", "NELM": 300, "NELMDL": -12, "AMIX": 0.2, "BMIX": 0.0001,
                "AMIX_MAG": 0.8, "BMIX_MAG": 0.0001, "MAGMOM": magmom_string(meta["magmoms"])})
    meta.update({"index": k, "staged_batch": "A-2026-09-20",
                 "note": "rerun of 2_afm (SCF diverged) with conservative linear mixing"})
    write_calc(STAGING / cand / f"{k}_afm", Poscar.from_file(src / "POSCAR").structure, inc,
               Kpoints.from_file(src / "KPOINTS"), (src / "potcar_symbols.txt").read_text(), meta)
    old = json.loads((src / "meta.json").read_text())
    old["superseded_by"] = f"{k}_afm"
    (src / "meta.json").write_text(json.dumps(old, indent=2))
    QUEUE.append((150, f"dft_staging/{cand}/{k}_afm", "std"))
    print(f"  {cand}: staged {k}_afm (supersedes 2_afm)")


# ----------------------------------------------------------------------------------------------
# Batch B: tight same-cell pairs
# ----------------------------------------------------------------------------------------------
TIGHT_TAGS = {"EDIFF": 1e-6, "EDIFFG": -0.01, "NELMIN": 6, "ALGO": "Normal", "ISIF": 2,
              "NSW": 99, "IBRION": 2, "NELM": 200}


def stage_tight_pair(cand: str, afm_calc: str, est_minutes: float, from_contcar=True):
    calc = STAGING / cand / afm_calc
    meta = json.loads((calc / "meta.json").read_text())
    src = best_file(calc, "CONTCAR") if from_contcar else calc / "POSCAR"
    st = Poscar.from_file(src).structure
    inc = base_incar(cand)
    inc.update(TIGHT_TAGS)
    kp = Kpoints.from_file(calc / "KPOINTS")
    pot = (calc / "potcar_symbols.txt").read_text()
    afm_mm = meta["magmoms"]
    fm_mm = [abs(m) for m in afm_mm]
    for name, mm, ordering in ((f"afm_{afm_calc}", afm_mm, meta["ordering"]), (f"fm_in_{afm_calc}", fm_mm, "FM")):
        i = Incar(inc)
        i["MAGMOM"] = magmom_string(mm)
        write_calc(TIGHT / cand / name, st, i, kp, pot,
                   {"candidate_id": cand, "ordering": ordering, "magmoms": mm, "cell_from": f"{afm_calc}/{src.name}",
                    "formula": st.composition.reduced_formula, "staged_batch": "B-2026-09-20",
                    "note": "tight same-cell pair: EDIFF=1e-6, EDIFFG=-0.01, ISIF=2 in the relaxed cell of " + afm_calc})
        QUEUE.append((est_minutes, f"dft_staging_tight/{cand}/{name}", "std"))
    print(f"  {cand}: staged tight pair in the {afm_calc} cell")


def stage_mp1216981_pairs():
    """No valid AFM sample exists (all three earlier 'AFM' calcs flipped the d0 Ti site). The
    regenerated Fe-Fe AFM supercells are run directly as tight same-cell pairs (positions-only
    relaxation of the MP-relaxed structure, which is already at MP settings)."""
    cand = "mp-1216981"
    base = base_incar(cand)
    base.update(TIGHT_TAGS)
    for regen in sorted(REGEN.glob(f"{cand}/[0-9]*_afm")):
        meta = json.loads((regen / "meta.json").read_text())
        st = Poscar.from_file(regen / "POSCAR").structure
        kp = Kpoints.from_file(regen / "KPOINTS")
        pot = (STAGING / cand / "0_fm" / "potcar_symbols.txt").read_text()
        for name, mm, ordering in ((f"afm_{regen.name}", meta["magmoms"], "AFM"),
                                   (f"fm_in_{regen.name}", [abs(m) for m in meta["magmoms"]], "FM")):
            i = Incar(base)
            i["MAGMOM"] = magmom_string(mm)
            write_calc(TIGHT / cand / name, st, i, kp, pot,
                       {"candidate_id": cand, "ordering": ordering, "magmoms": mm, "cell_from": f"regen/{regen.name}",
                        "formula": st.composition.reduced_formula, "staged_batch": "B-2026-09-20",
                        "note": "Fe-Fe AFM supercell (Ti excluded from the enumeration) and its FM twin, tight, ISIF=2"})
            QUEUE.append((6, f"dft_staging_tight/{cand}/{name}", "std"))
    print(f"  {cand}: staged 4 Fe-Fe AFM supercells + FM twins")


# ----------------------------------------------------------------------------------------------
# Batch C: spiral DMI probe for mp-1079285
# ----------------------------------------------------------------------------------------------
def stage_spiral_mp1079285():
    cand = "mp-1079285"
    fm = STAGING / cand / "0_fm"
    prim = Poscar.from_file(fm / "CONTCAR").structure
    assert sum(1 for s in prim if s.specie.symbol == "Fe") == 1
    # Fe chain along b (4.198 A); next shell 8.4 A along a and c. Verified in-session.
    sup = prim.copy()
    sup.make_supercell([[1, 0, 0], [0, 4, 0], [0, 0, 1]])
    # MPRelaxSet-style species grouping is preserved by make_supercell; sort Fe by y so the
    # spiral phase is assigned by position along the chain.
    fe_idx = [i for i, s in enumerate(sup) if s.specie.symbol == "Fe"]
    fe_idx.sort(key=lambda i: sup[i].frac_coords[1])
    assert len(fe_idx) == 4
    lat = sup.lattice.matrix
    b_hat = lat[1] / np.linalg.norm(lat[1])
    n1 = lat[0] - np.dot(lat[0], b_hat) * b_hat
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(b_hat, n1)
    normals = {"n1": n1, "n2": n2}
    # reference in-plane axis e0 and e1 = n x e0 so that e0 x e1 = n
    base_inc = outcar_incar(best_outcar(fm))
    base_inc.pop("MAGMOM", None)
    nelect = float(re.search(r"NELECT\s*=\s*([\d.]+)", best_outcar(fm).read_text()).group(1)) * 4
    nbands = int(np.ceil(2 * (nelect / 2 + len(sup)) / 48) * 48)
    kp = Kpoints.gamma_automatic((3, 2, 3))
    pot = (fm / "potcar_symbols.txt").read_text()
    # RWIGS: below every hetero-pair minimum distance (Fe-Tb 2.846, Tb-Bi 3.305, Tb-Tb/Bi-Bi checked below)
    rwigs = {"Tb": 1.40, "Fe": 1.25, "Bi": 1.50}
    dm = sup.distance_matrix
    for i in range(len(sup)):
        for j in range(i + 1, len(sup)):
            si, sj = sup[i].specie.symbol, sup[j].specie.symbol
            assert rwigs[si] + rwigs[sj] < dm[i, j] - 0.1, (si, sj, dm[i, j])
    species_order = species_list(sup)
    m_fe = 1.7  # |m_Fe| from the collinear FM run (audit reference 1.68 uB)
    for nname, n in normals.items():
        e0 = b_hat  # spins start along the chain; rotation plane contains b, normal n
        e1 = np.cross(n, e0)
        for chir, sign in (("plus", +1), ("minus", -1)):
            vecs = np.zeros((len(sup), 3))
            for j, i in enumerate(fe_idx):
                phi = sign * j * np.pi / 2
                vecs[i] = np.cos(phi) * e0 + np.sin(phi) * e1
            magmom = " ".join(f"{m_fe*v[0]:.4f} {m_fe*v[1]:.4f} {m_fe*v[2]:.4f}" for v in vecs)
            mconstr = " ".join(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" for v in vecs)
            inc = Incar(base_inc)
            inc.update({
                "PREC": "Accurate", "ENCUT": 520.0, "EDIFF": 1e-7, "NELM": 300, "ALGO": "Normal",
                "ISMEAR": 0, "SIGMA": 0.1, "ISYM": 0, "LREAL": False, "LASPH": True,
                "IBRION": -1, "NSW": 0, "ISIF": 2, "LWAVE": False, "LCHARG": False, "LORBIT": 11,
                "LNONCOLLINEAR": True, "LSORBIT": True, "GGA_COMPAT": False, "SAXIS": "0 0 1",
                "NBANDS": nbands, "AMIX": 0.2, "BMIX": 0.0001, "AMIX_MAG": 0.8, "BMIX_MAG": 0.0001,
                "I_CONSTRAINED_M": 1, "LAMBDA": 10,
                "RWIGS": " ".join(str(rwigs[s]) for s in species_order),
            })
            for tag in ("EDIFFG", "NELMDL", "POTIM", "NELMIN", "LMAXMIX"):
                inc.pop(tag, None)
            d = NONCOL / cand / f"spiral_{nname}_{chir}"
            write_calc(d, sup, inc, kp, pot, {
                "candidate_id": cand, "formula": sup.composition.reduced_formula, "staged_batch": "C-2026-09-20",
                "supercell": "1x4x1 of 0_fm/CONTCAR (Fe chain along b, 4.198 A)",
                "rotation_plane_normal": n.round(6).tolist(), "chirality": sign,
                "fe_site_indices_in_chain_order": fe_idx,
                "spin_directions": {str(i): vecs[i].round(6).tolist() for i in fe_idx},
                "extraction": "D.n = [E(plus) - E(minus)] / (2*4) eV per NN bond, unit spins; "
                              "E = sum_<ij> D_ij.(S_i x S_j), i->j along +b",
            }, raw_tags={"MAGMOM": magmom, "M_CONSTR": mconstr})
            QUEUE.append((240, f"dft_staging_noncollinear/{cand}/{d.name}", "ncl"))
    (NONCOL / cand / "dmi_setup_notes.md").write_text(SPIRAL_NOTES.format(nbands=nbands, n1=np.round(n1, 4), n2=np.round(n2, 4)))
    print(f"  {cand}: staged 4 spiral DMI runs ({len(sup)} atoms, NBANDS={nbands})")


SPIRAL_NOTES = """# mp-1079285 (Tb6FeBi2) -- spiral DMI probe, staged 2026-09-20

Fe forms chains along b (Fe-Fe 4.198 A); the next Fe shell is 8.4 A (along a and c). One Fe per
primitive cell, space group Pm (mirror plane perpendicular to b through every Fe AND through
every Fe-Fe bond midpoint) -> Moriya: D lies in the mirror plane, D_b = 0.

Why not the two-state 2x cell used for mp-754090: with one Fe per cell, Fe0->Fe1 and Fe1->Fe0'
(periodic image) are the same bond by translation, so D.(S0xS1) + D.(S1xS0) = 0 identically --
a 2x cell gives exactly zero DMI signal whatever D is. The smallest cell that sees D is 4x with a
q = pi/(2b) spiral: phases 0, 90, 180, 270 deg (plus) or 0, -90, -180, -270 deg (minus).

E(q) - E(-q) = 2 * sum_bonds D_ij . n * sin(phi_j - phi_i)  = 2 * 4 * D.n   (4 NN bonds per cell)
=> D.n = [E(plus) - E(minus)] / 8   (unit spins; symmetric exchange and anisotropy cancel)
Interchain bonds (r perpendicular to b) contribute sin(0)=0; the diagonal a+-b / c+-b bonds
(9.4 A) enter as D(a+b)-D(a-b) and are assumed negligible.

Two rotation-plane normals, both perpendicular to b:  n1 = {n1} (in-plane along a_perp),
n2 = {n2} (= b x n1). D = (D.n1) n1 + (D.n2) n2.

Settings: constrained-direction moments (I_CONSTRAINED_M=1, LAMBDA=10 -- held mp-754090's
directions to lambda*MW_perp ~ 1e-2, re-check here), RWIGS Tb 1.40 Fe 1.25 Bi 1.50 (all pair sums
>= 0.1 A below the corresponding minimum distance, asserted at staging), ISMEAR=0 SIGMA=0.1
(metal; identical in +q/-q so it cancels), linear-ish mixing AMIX=0.2 BMIX=1e-4 AMIX_MAG=0.8
(metallic cell), EDIFF=1e-7, NBANDS={nbands}, k 3x2x3 Gamma (4x along b from the 3x6x3 FM mesh),
no U (intermetallic, MP applies U only to oxides/fluorides; Tb_3 POTCAR, 4f in core).
"""


# ----------------------------------------------------------------------------------------------
def main():
    audit = {(r["candidate_id"], r["calc"]): (r["valid"] == "True", r["problems"])
             for r in csv.DictReader(open(AUDIT, newline=""))}
    for d in (TIGHT, NONCOL / "mp-1079285"):
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"{d} already populated -- refusing to overwrite; remove it first if re-staging")

    print("Batch A")
    stage_new_configs("mp-1173143", audit, 5)
    stage_new_configs("mp-1173139", audit, 5)
    stage_new_configs("mp-775194", audit, 15)
    stage_new_configs("mp-551086", audit, 4)
    stage_mp22972(audit)
    stage_mp682554()

    print("Batch B")
    stage_tight_pair("mp-1079285", "3_afm", 8)
    stage_tight_pair("mp-1228547", "3_afm", 40)
    stage_tight_pair("mp-850225", "2_afm", 40)
    stage_tight_pair("mp-754090", "1_afm", 12)
    stage_tight_pair("mp-755203", "2_afm", 20)
    stage_tight_pair("mp-1043970", "1_afm", 6)
    stage_tight_pair("mp-1047414", "3_afm", 5)
    stage_tight_pair("mp-1047414", "1_afm", 5)
    stage_tight_pair("mp-1173139", "1_afm", 8)
    stage_tight_pair("mp-551086", "1_afm", 4)
    stage_mp1216981_pairs()

    print("Batch C")
    stage_spiral_mp1079285()

    QUEUE.sort(key=lambda q: -q[0])
    (ROOT / "src" / "dft" / "farm_queue_rerun.txt").write_text(
        "".join(f"{d} {b}\n" for _, d, b in QUEUE))
    print(f"\n{len(QUEUE)} calcs, ~{sum(q[0] for q in QUEUE)/60:.1f} node-hours estimated; "
          f"queue written to src/dft/farm_queue_rerun.txt (longest first)")


if __name__ == "__main__":
    main()
