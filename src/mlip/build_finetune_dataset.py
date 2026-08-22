"""Build a MACE fine-tuning dataset (extxyz, energy+forces+stress) by filtering the MPtrj
dataset (github.com/materialsproject / huggingface.co/datasets/nimashoghi/mptrj -- the same
trajectory dataset MACE-MP-0 and MatterSim were themselves trained on) down to the magnetic
materials already identified in data/processed/harmonized_v2.json.

Earlier approach (pulling forces live via mp-api's /tasks endpoint) does NOT work: the live
API only serves reduced task documents without forces/stress (confirmed by testing, not
assumed -- every material came back with an AttributeError). MPtrj is the standard, correct
source for MLIP fine-tuning data in this ecosystem and is what this script uses instead.

Note on access: `datasets.load_dataset(..., streaming=True)` hung indefinitely with no network
activity (confirmed via `ss -tp`, zero open connections) -- a builder/lock issue in the
`datasets` library itself, not a network block (plain `requests` calls to the same HF
endpoints returned instantly). Workaround: download parquet shards directly via `requests`
and read with `pyarrow`, bypassing `datasets` entirely.

Usage:
    python download_mptrj_shard.py --shard 0          # downloads data/finetune/mptrj_raw/shard0.parquet
    python build_finetune_dataset.py --shard 0 --out data/finetune/mace_magnetic_shard0.xyz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    shard_path = ROOT / "data" / "finetune" / "mptrj_raw" / f"shard{args.shard}.parquet"
    if not shard_path.exists():
        raise SystemExit(f"{shard_path} not found -- run download_mptrj_shard.py --shard {args.shard} first")

    out_path = Path(args.out) if args.out else ROOT / "data" / "finetune" / f"mace_magnetic_shard{args.shard}.xyz"

    with open(ROOT / "data" / "processed" / "harmonized_v2.json") as f:
        records = json.load(f)
    magnetic_mp_ids = {r["material_id"] for r in records if r["source_database"] == "MaterialsProject"}
    print(f"Harmonized magnetic MP material set: {len(magnetic_mp_ids)} ids")

    table = pq.read_table(str(shard_path))
    print(f"Shard {args.shard}: {table.num_rows} rows total")

    mp_id_col = table.column("mp_id").to_pylist()
    match_mask = [mid in magnetic_mp_ids for mid in mp_id_col]
    matched = table.filter(match_mask).to_pylist()  # pyarrow's own conversion keeps clean nested Python lists
    print(f"Rows matching our magnetic subset: {len(matched)}")

    atoms_list = []
    for row in matched:
        numbers = np.array(row["numbers"], dtype=int)
        positions = np.array(row["positions"], dtype=float)
        cell = np.array(row["cell"], dtype=float)
        pbc = list(row["pbc"])
        forces = np.array(row["forces"], dtype=float)
        energy = float(row["energy"])
        stress = row["stress"]

        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
        atoms.info["REF_energy"] = energy
        atoms.arrays["REF_forces"] = forces
        if stress is not None:
            atoms.info["REF_stress"] = np.array(stress, dtype=float)
        atoms.info["mp_id"] = row["mp_id"]
        atoms_list.append(atoms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), atoms_list, format="extxyz")
    print(f"Wrote {len(atoms_list)} labeled structures to {out_path}")


if __name__ == "__main__":
    main()
