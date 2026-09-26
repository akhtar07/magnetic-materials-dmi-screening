"""Parse converged VASP magnetic-ordering calculations under data/dft_staging/<candidate>/ and
compare the DFT ground state against the ordering_final label used to screen the candidate
(outputs/phase4_shortlist.csv), i.e. close the loop left open by prepare_magnetic_ordering_dft.py.

Usage:
    python analyze_magnetic_ordering_results.py
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "data" / "dft_staging"
SHORTLIST = ROOT / "outputs" / "phase4_shortlist.csv"
OUT_CSV = ROOT / "outputs" / "dft_ordering_validation.csv"
SPIN_AUDIT = ROOT / "outputs" / "dft_spin_state_audit.csv"  # written by audit_spin_states.py

TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-\d.]+)\s*eV")


def final_toten(outcar: Path) -> float:
    matches = TOTEN_RE.findall(outcar.read_text())
    if not matches:
        raise ValueError(f"no TOTEN found in {outcar}")
    return float(matches[-1])


def converged(outcar: Path) -> bool:
    return "reached required accuracy" in outcar.read_text()


def load_spin_audit() -> dict:
    """(candidate_id, calc) -> (valid, problems). A calc whose magnetic ions collapsed to a
    different spin state, or whose moments re-ordered away from the intended pattern, is NOT a
    valid sample of that ordering and must not enter the verdict even though VASP converged it
    (see audit_spin_states.py). Missing audit file -> everything treated as valid, with a warning.
    """
    if not SPIN_AUDIT.exists():
        print(f"WARNING: {SPIN_AUDIT.relative_to(ROOT)} not found -- run audit_spin_states.py first; "
              f"treating every calc as spin-state-valid")
        return {}
    with open(SPIN_AUDIT, newline="") as f:
        return {(r["candidate_id"], r["calc"]): (r["valid"] == "True", r["problems"]) for r in csv.DictReader(f)}


def load_shortlist_labels() -> dict:
    with open(SHORTLIST, newline="") as f:
        return {row["material_id"]: row for row in csv.DictReader(f)}


def main():
    labels = load_shortlist_labels()
    spin_audit = load_spin_audit()
    rows = []

    for cand_dir in sorted(p for p in STAGING.iterdir() if p.is_dir()):
        candidate_id = cand_dir.name
        if not candidate_id.startswith("mp-"):
            # mixer/smearing test beds synced from the cluster (mixtest*, _mixtest755) are not candidates
            continue
        calcs = []
        # a staged calc dir carries meta.json; anything else (mp-682554/diag_1x1x2 NSW=0 mixer
        # diagnostics) is scaffolding and neither a sample nor a completeness gap
        staged_calc_dirs = sorted(p for p in cand_dir.iterdir() if p.is_dir() and (p / "meta.json").exists())
        for calc_dir in list(staged_calc_dirs):
            meta = json.loads((calc_dir / "meta.json").read_text())
            if meta.get("superseded_by"):
                # e.g. mp-682554/2_afm (SCF diverged) re-run as 4_afm with a different mixer:
                # the superseded attempt is neither a sample nor a "not run yet" gap
                print(f"  {candidate_id}/{calc_dir.name}: superseded by {meta['superseded_by']}, skipping")
                staged_calc_dirs.remove(calc_dir)
                continue
            outcar = calc_dir / "results" / "OUTCAR"
            if not outcar.exists():
                outcar = calc_dir / "OUTCAR"
            if not outcar.exists():
                print(f"  {candidate_id}/{calc_dir.name}: no OUTCAR, skipping (not run yet)")
                continue
            ok = converged(outcar)
            try:
                energy = final_toten(outcar)
            except ValueError:
                print(f"  {candidate_id}/{calc_dir.name}: OUTCAR has no TOTEN "
                      f"(crashed before first ionic step), skipping")
                continue
            n_atoms = len(meta["magmoms"])
            spin_valid, spin_problems = spin_audit.get((candidate_id, calc_dir.name), (True, ""))
            calcs.append({
                "candidate_id": candidate_id,
                "calc": calc_dir.name,
                "ordering": meta["ordering"],
                "formula": meta["formula"],
                "n_atoms": n_atoms,
                "toten_ev": energy,
                "energy_per_atom_ev": energy / n_atoms,
                "converged": ok,
                "spin_state_valid": spin_valid,
                "spin_state_problems": spin_problems,
            })
            if not ok:
                print(f"  WARNING: {candidate_id}/{calc_dir.name} did not report convergence")
            if not spin_valid:
                print(f"  WARNING: {candidate_id}/{calc_dir.name} excluded from verdict -- {spin_problems}")

        if not calcs:
            continue

        # Non-converged energies (e.g. a TIME-LIMIT-truncated relaxation) are not a valid basis
        # for a ground-state comparison -- an unrelaxed intermediate TOTEN can land arbitrarily
        # far from the true minimum. Only converged calcs count toward the verdict; a candidate
        # is "complete" only once every staged calc dir has converged.
        # ...and a converged calc that landed in the wrong spin state is equally unusable.
        converged_calcs = [c for c in calcs if c["converged"] and c["spin_state_valid"]]
        is_complete = len(converged_calcs) == len(staged_calc_dirs)
        label = labels.get(candidate_id, {})
        predicted_ordering = label.get("ordering_final", "?")
        predicted_source = label.get("ordering_source", "?")
        predicted_confidence = label.get("confidence", "?")

        print(f"\n=== {candidate_id} ({calcs[0]['formula']}) "
              f"[{len(converged_calcs)}/{len(staged_calc_dirs)} converged+valid] ===")
        for c in sorted(calcs, key=lambda c: c["toten_ev"]):
            print(f"  {c['calc']:8s} {c['ordering']:4s} TOTEN={c['toten_ev']:12.5f} eV "
                  f"({c['energy_per_atom_ev']:.6f} eV/atom) converged={c['converged']} spin_ok={c['spin_state_valid']}")

        ground_state = None
        agrees = None
        gap_vs_fm_ev_atom = None
        if len(converged_calcs) < 2:
            print(f"  INCOMPLETE: fewer than 2 converged calcs, no verdict yet")
        else:
            # Compare per-atom energy, not raw TOTEN: AFM configs are enumerated in a doubled/
            # quadrupled supercell relative to the FM primitive cell, so raw TOTEN systematically
            # favors whichever config has more atoms regardless of which ordering is actually
            # more stable.
            ground_state = min(converged_calcs, key=lambda c: c["energy_per_atom_ev"])
            agrees = ground_state["ordering"] == predicted_ordering
            print(f"  predicted ordering: {predicted_ordering} (source={predicted_source}, "
                  f"confidence={predicted_confidence})")
            print(f"  DFT ground state (among converged calcs):   {ground_state['ordering']}"
                  f"{'' if is_complete else '  ** PROVISIONAL: not all configs converged yet **'}")
            print(f"  AGREES: {agrees}")
            # The enumerator only ever produces FM/AFM configs (never FiM even when MP's own
            # label is FiM), so the one config comparable across every candidate is the FM
            # baseline (calc "0_fm") -- report the gap against that specifically, matching the
            # "ΔE vs FM" convention used in the original 6-candidate report, rather than against
            # whichever config happens to be the immediate runner-up by energy.
            fm_calc = next((c for c in converged_calcs if c["calc"] == "0_fm"), None)
            if fm_calc is not None:
                gap_vs_fm_ev_atom = fm_calc["energy_per_atom_ev"] - ground_state["energy_per_atom_ev"]
                print(f"  gap vs FM (0_fm) baseline: {gap_vs_fm_ev_atom * ground_state['n_atoms']:.5f} eV "
                      f"({gap_vs_fm_ev_atom:.6f} eV/atom = {gap_vs_fm_ev_atom * 1000:.4f} meV/atom)")
            runner_up = sorted(converged_calcs, key=lambda c: c["energy_per_atom_ev"])[1]
            gap_runner_up_atom = runner_up["energy_per_atom_ev"] - ground_state["energy_per_atom_ev"]
            print(f"  gap to next-lowest converged config ({runner_up['calc']}, {runner_up['ordering']}): "
                  f"{gap_runner_up_atom:.6f} eV/atom")

        for c in calcs:
            c["predicted_ordering"] = predicted_ordering
            c["gap_vs_fm_meV_per_atom"] = (
                round(gap_vs_fm_ev_atom * 1000, 4) if gap_vs_fm_ev_atom is not None else ""
            )
            c["predicted_source"] = predicted_source
            c["predicted_confidence"] = predicted_confidence
            c["is_dft_ground_state"] = ground_state is not None and c is ground_state
            c["agrees_with_prediction"] = agrees
            c["candidate_complete"] = is_complete
            c["n_staged"] = len(staged_calc_dirs)
            rows.append(c)

    if not rows:
        print("No completed DFT calculations found under data/dft_staging/.")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
