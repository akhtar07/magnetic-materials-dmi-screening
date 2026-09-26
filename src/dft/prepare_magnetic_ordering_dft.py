"""Generate real VASP inputs to validate a shortlisted candidate's predicted magnetic ordering
(screen_candidates.py) against actual DFT total energies -- the thing this project has not had
access to until now (no DFT infra existed in this repo; the fine-tuned MACE model cannot
distinguish magnetic configurations of the same atomic structure, since it was never trained
with magmom as an input feature -- verified by inspecting src/mlip/build_finetune_dataset.py).

Method: standard collinear magnetic-ordering energy mapping (the same approach behind Materials
Project's own magnetic-ordering enumeration workflow, Horton/Montoya/Persson npj Comput Mater
2019). For a candidate's DFT-relaxed structure, pymatgen's MagneticStructureEnumerator enumerates
symmetry-distinct FM/AFM spin configurations (via enumlib), and MPRelaxSet generates the exact
Materials-Project-standard POSCAR/INCAR/KPOINTS (GGA+U with MP's own U values, standard ENCUT/
k-point density) for each one, with MAGMOM overridden per-ordering to encode the actual spin
pattern. This makes the result directly comparable to the ground-truth MP calculation the
candidate was screened against, not an arbitrary DFT recipe.

Licensing note: POTCAR pseudopotential *data* is never touched here or transferred through this
repo -- only the required POTCAR *symbols* (e.g. "Fe_pv") are written to potcar_symbols.txt. The
actual POTCAR files must be concatenated on the remote cluster where the licensed pseudopotential
library lives (see assemble_and_submit.sh), not locally.

Usage:
    python prepare_magnetic_ordering_dft.py --candidate mp-861559 [--max-afm 3]
"""
import argparse
import json
import warnings
from pathlib import Path

from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer, MagneticStructureEnumerator
from pymatgen.analysis.structure_matcher import AbstractComparator, StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "data" / "dft_staging"


class SameSpinComparator(AbstractComparator):
    """Species equal iff same element and same *sign* of spin. (pymatgen's own SpinComparator
    matches a structure only to its time-reversed partner -- fit(a, a) is False -- so it cannot
    be used on its own to detect plain duplicates.)"""

    @staticmethod
    def _sign(sp):
        spin = getattr(sp, "spin", 0) or 0
        return (spin > 0) - (spin < 0)

    def are_equal(self, sp1, sp2) -> bool:
        return {(s.symbol, self._sign(s)) for s in sp1} == {(s.symbol, self._sign(s)) for s in sp2}

    def get_hash(self, composition):
        return composition.fractional_composition


def prepare(candidate_id: str, max_afm: int, staging: Path = STAGING, magnetic_species_override=()) -> Path:
    structs = json.loads((ROOT / "data" / "processed" / "candidate_structures.json").read_text())
    entry = structs.get(candidate_id)
    if entry is None or "structure" not in entry:
        raise SystemExit(f"{candidate_id}: no backfilled structure in candidate_structures.json")
    structure = Structure.from_dict(entry["structure"])

    # Moment carriers. Ti and V are only treated as magnetic when nothing else is: in the oxides
    # screened here they are d0 (Ti4+/V5+) and carry no moment, and letting the enumerator flip
    # them produced "AFM" configs that were really the FM state again (mp-1216981: Fe down vs
    # Ti4+ up, |m_Ti| = 0.06 after SCF -- caught by audit_spin_states.py, 2026-09-20).
    present = {str(site.specie) for site in structure}
    magnetic_species = sorted(present & (set(magnetic_species_override) if magnetic_species_override
                                         else {"Fe", "Mn", "Co", "Ni", "Cr"}))
    if not magnetic_species and not magnetic_species_override:
        magnetic_species = sorted(present & {"V", "Ti"})
    if not magnetic_species:
        raise SystemExit(f"{candidate_id}: no recognized 3d transition-metal species found")
    default_magmoms = {el: 5.0 for el in magnetic_species}

    print(f"{candidate_id}: {structure.composition.reduced_formula}, {len(structure)} sites, "
          f"magnetic species {magnetic_species}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enumerator = MagneticStructureEnumerator(
            structure, default_magmoms=default_magmoms, transformation_kwargs={"max_cell_size": 1},
        )

    # Keep the FM configuration plus up to max_afm symmetry-distinct antiparallel configurations;
    # drop the redundant "input" duplicate enumlib always appends.
    #
    # BUG FIXED 2026-09-20: the enumerator's `afm_by_motif_*` / `afm_by_<species>` strategies
    # emit configurations in which only ONE Wyckoff motif (or one species) is antiparallel and
    # every other magnetic site is set to ZERO moment. CollinearMagneticStructureAnalyzer still
    # labels those "AFM" (net moment zero), so the old filter accepted them -- and because they
    # come first in enumlib's output they crowded out the genuine all-sites-populated `afm`
    # configurations. Seven staged calcs (mp-1173143 x3, mp-551086 x2, mp-1173139, mp-775194)
    # were run that way; their "AFM" energies compare a partly non-magnetic state against FM and
    # are not exchange energies (see audit_spin_states.py). Now: every site of a magnetic species
    # must carry a non-zero moment, and ferrimagnetic (FiM) configurations -- which are the only
    # antiparallel arrangements possible for some multi-species cells -- are accepted too.
    # Also: the enumerator's output order is not stable and interleaves cell sizes, so choose
    # deliberately -- populated configs ranked AFM before FiM, smaller cell first, plain `afm`
    # origin first -- after removing time-reversal / spin-pattern duplicates that survive the
    # enumerator's own StructureMatcher pass.
    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False, comparator=SameSpinComparator())
    populated = []
    n_dropped_zero = 0
    for st, origin in zip(enumerator.ordered_structures, enumerator.ordered_structure_origins):
        if origin == "input":
            continue
        analyzer = CollinearMagneticStructureAnalyzer(st)
        ordering = analyzer.ordering.name  # "FM" / "AFM" / "FiM" / "NM"
        if ordering not in ("FM", "AFM", "FiM"):
            continue
        mag_sites = [abs(m) for m, site in zip(analyzer.magmoms, st)
                     if str(site.specie).split(",")[0] in magnetic_species]
        if any(m < 1e-3 for m in mag_sites):
            n_dropped_zero += 1
            continue
        populated.append((st, analyzer, ordering, origin))
    if n_dropped_zero:
        print(f"  dropped {n_dropped_zero} enumerated configuration(s) with un-populated magnetic sites")

    rank = {"FM": 0, "AFM": 1, "FiM": 2}
    populated.sort(key=lambda t: (rank[t[2]], len(t[0]), 0 if t[3] in ("fm", "afm") else 1))
    kept = []
    n_afm = 0
    for st, analyzer, ordering, origin in populated:
        mine = analyzer.get_structure_with_spin()
        flipped = mine.copy()
        flipped.remove_spin()
        flipped.add_site_property("magmom", [-m for m in analyzer.magmoms])
        flipped = CollinearMagneticStructureAnalyzer(flipped).get_structure_with_spin()
        if any(matcher.fit(mine, k) or matcher.fit(flipped, k) for k, *_ in kept):
            continue
        if ordering == "FM":
            if any(o == "FM" for *_, o in kept):
                continue
        else:
            if n_afm >= max_afm:
                continue
            n_afm += 1
        kept.append((mine, st, analyzer, ordering))
    kept = [(st, analyzer, ordering) for _, st, analyzer, ordering in kept]
    print(f"  keeping {sum(1 for *_, o in kept if o == 'FM')} FM + {n_afm} AFM/FiM configuration(s)")

    out_dir = staging / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (st, analyzer, ordering) in enumerate(kept):
        calc_dir = out_dir / f"{i}_{ordering.lower()}"
        calc_dir.mkdir(parents=True, exist_ok=True)

        # MPRelaxSet reads per-site MAGMOM straight off the structure (site.magmom property, or
        # spin-decorated Species) when user_incar_settings["MAGMOM"] is absent -- and st already
        # carries that from the enumerator/analyzer, so just make it explicit via site_properties
        # rather than fighting the (species-string-keyed, not per-site) user_incar_settings path.
        magmoms = list(analyzer.magmoms)
        st = st.copy()
        st.add_site_property("magmom", magmoms)
        mpset = MPRelaxSet(st, user_incar_settings={"ISYM": 0})

        # NOTE: MPRelaxSet internally regroups sites by spin-decorated species identity when it
        # builds self.structure (e.g. Fe,spin=+5 and Fe,spin=-5 become separate blocks), so the
        # on-disk site order can differ from `st`'s. POSCAR and INCAR MAGMOM are both generated
        # from that same reordered mpset.structure, so they stay mutually consistent -- but that
        # means the *original* `magmoms` list above is not what ends up on disk. Re-read the
        # actual written order back off mpset.structure for meta.json, so it reflects reality.
        written_magmoms = [float(getattr(site, "magmom", 0.0)) for site in mpset.structure]

        mpset.poscar.write_file(calc_dir / "POSCAR")
        mpset.incar.write_file(calc_dir / "INCAR")
        mpset.kpoints.write_file(calc_dir / "KPOINTS")
        (calc_dir / "potcar_symbols.txt").write_text("\n".join(mpset.potcar_symbols) + "\n")
        (calc_dir / "meta.json").write_text(json.dumps({
            "candidate_id": candidate_id, "ordering": ordering, "index": i,
            "magmoms": written_magmoms, "formula": st.composition.reduced_formula,
        }, indent=2))
        print(f"  wrote {calc_dir} ({ordering}, magmoms={written_magmoms})")

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--max-afm", type=int, default=3)
    ap.add_argument("--staging", type=Path, default=STAGING,
                    help="root dir to write <candidate>/<i>_<ordering>/ into (default data/dft_staging)")
    ap.add_argument("--magnetic-species", nargs="*", default=(),
                    help="override the moment-carrying species (default Fe/Mn/Co/Ni/Cr present in the cell)")
    args = ap.parse_args()
    prepare(args.candidate, args.max_afm, args.staging, args.magnetic_species)


if __name__ == "__main__":
    main()
