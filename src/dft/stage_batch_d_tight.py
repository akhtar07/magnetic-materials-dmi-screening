"""Batch D (2026-09-21): tight same-cell FM/AFM pairs for every "overturned" verdict that so far
rests only on the loose protocol with a margin inside the loose-vs-tight discrepancy band.

Comparing the eight configurations that were run under both protocols, the loose margin (FM in
its primitive cell vs. AFM in a supercell, both ISIF=3, MPRelaxSet k-meshes) differs from the
tight one (both members in the same relaxed AFM cell, ISIF=2, EDIFF=1e-6) by up to 1.3 meV/atom
in either direction (Ba4Ta10FeO30 +1.18 -> -0.00, FeP6(WO8)3 +2.09 -> +3.38). A loose-only
"AFM below FM" verdict with |margin| < ~2 meV/atom is therefore not trustworthy on its own; the
six candidates below are re-decided with a tight pair built in the cell of their lowest valid
non-FM configuration, exactly as Batch B did (stage_rerun_batches.stage_tight_pair).

Usage:
    python stage_batch_d_tight.py        # writes data/dft_staging_tight/<cand>/{afm_,fm_in_}<calc>
                                         # and src/dft/farm_queue_append3.txt (longest first)
"""
from pathlib import Path

from stage_rerun_batches import QUEUE, TIGHT, stage_tight_pair

ROOT = Path(__file__).resolve().parents[2]

# (candidate, lowest valid non-FM config, loose margin meV/atom, est. minutes on 48 cores)
PAIRS = [
    ("mp-682554", "3_afm", -0.29, 300),   # 130 atoms; loose 3_afm took ~2 h (ISIF=3)
    ("mp-1197024", "3_afm", -1.19, 120),  # 56 atoms
    ("mp-1173143", "7_afm", -0.41, 90),   # 54 atoms
    ("mp-551086", "5_fim", -0.85, 60),    # 20 atoms; the Batch-B 1_afm pair was spin-state invalid
    ("mp-1343817", "1_afm", -1.77, 60),   # 18 atoms
    ("mp-22972", "3_afm", -0.14, 60),     # 18 atoms; same-cell ISIF=3 already (4_fm), tightened here
]


def main():
    for cand, calc, _, _ in PAIRS:
        for name in (f"afm_{calc}", f"fm_in_{calc}"):
            if (TIGHT / cand / name).exists():
                raise SystemExit(f"{TIGHT / cand / name} exists -- refusing to overwrite")
    for cand, calc, _, est in PAIRS:
        stage_tight_pair(cand, calc, est)
    for cand, calc, _, _ in PAIRS:
        for name in (f"afm_{calc}", f"fm_in_{calc}"):
            meta = TIGHT / cand / name / "meta.json"
            meta.write_text(meta.read_text().replace('"B-2026-09-20"', '"D-2026-09-21"'))
    QUEUE.sort(key=lambda q: -q[0])
    out = ROOT / "src" / "dft" / "farm_queue_append3.txt"
    out.write_text("".join(f"{d} {b}\n" for _, d, b in QUEUE))
    print(f"{len(QUEUE)} calcs, ~{sum(q[0] for q in QUEUE)/60:.1f} node-hours estimated -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
