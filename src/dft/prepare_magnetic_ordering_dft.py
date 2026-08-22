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
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "data" / "dft_staging"


def prepare(candidate_id: str, max_afm: int) -> Path:
    structs = json.loads((ROOT / "data" / "processed" / "candidate_structures.json").read_text())
    entry = structs.get(candidate_id)
    if entry is None or "structure" not in entry:
        raise SystemExit(f"{candidate_id}: no backfilled structure in candidate_structures.json")
    structure = Structure.from_dict(entry["structure"])

    magnetic_species = sorted({
        str(site.specie) for site in structure
        if str(site.specie) in {"Fe", "Mn", "Co", "Ni", "Cr", "V", "Ti"}  # 3d TM, the usual carriers here
    })
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

    # Keep the FM configuration plus up to max_afm symmetry-distinct AFM configurations found;
    # drop the redundant "input" duplicate enumlib always appends.
    kept = []
    seen_fm = False
    n_afm = 0
    for st, origin in zip(enumerator.ordered_structures, enumerator.ordered_structure_origins):
        analyzer = CollinearMagneticStructureAnalyzer(st)
        ordering = analyzer.ordering.name  # "FM" / "AFM" / "FiM" / "NM"
        if origin == "input":
            continue
        if ordering == "FM":
            if seen_fm:
                continue
            seen_fm = True
        elif ordering == "AFM":
            if n_afm >= max_afm:
                continue
            n_afm += 1
        else:
            continue
        kept.append((st, analyzer, ordering))
    print(f"  keeping {sum(1 for *_, o in kept if o == 'FM')} FM + {n_afm} AFM configuration(s)")

    out_dir = STAGING / candidate_id
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
        print(f"  wrote {calc_dir.relative_to(ROOT)} ({ordering}, magmoms={written_magmoms})")

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--max-afm", type=int, default=3)
    args = ap.parse_args()
    prepare(args.candidate, args.max_afm)


if __name__ == "__main__":
    main()
