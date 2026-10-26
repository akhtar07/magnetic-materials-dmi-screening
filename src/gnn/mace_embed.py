"""Frozen MACE-MP-0 (medium) descriptors of every labelled structure: the pretrained foundation-model
representation of the classifier comparison. No ordering label or magnetic moment enters; the model was
trained on energies, forces and stresses of the MPtrj relaxation trajectories (Batatia et al. 2023).

    python src/gnn/mace_embed.py [--device cpu --shard i --nshards n]     # env: magnetic-mlip-mace
    python src/gnn/mace_embed.py --merge n                                 # join n shards

Per structure, the invariant per-atom descriptors of all interaction layers are pooled by mean and standard
deviation over atoms -> data/features/mace_mp0_embeddings.npz (ids, X).
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from mace.calculators import mace_mp

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path.home() / ".cache" / "mace" / "20231203mace128L1_epoch199model"  # MACE-MP-0 medium


OUT = ROOT / "data" / "features" / "mace_mp0_embeddings.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", type=int, default=0)
    a = ap.parse_args()
    if a.merge:
        parts = [np.load(OUT.with_suffix(f".part{i}.npz")) for i in range(a.merge)]
        np.savez_compressed(OUT, ids=np.concatenate([p["ids"] for p in parts]), X=np.concatenate([p["X"] for p in parts]))
        print("merged", sum(len(p["ids"]) for p in parts), "embeddings ->", OUT)
        return
    import torch
    torch.set_num_threads(4)
    G = pickle.load(open(ROOT / "data" / "processed" / "gnn_graphs.pkl", "rb"))["graphs"]
    items = list(G.items())[a.shard::a.nshards]
    calc = mace_mp(model=str(MODEL), device=a.device, default_dtype="float32")
    ids, X, t0 = [], [], time.time()
    for k, (mid, g) in enumerate(items):
        at = Atoms(numbers=g["z"].astype(int), scaled_positions=g["frac"], cell=g["lattice"], pbc=True)
        d = calc.get_descriptors(at, invariants_only=True)
        X.append(np.concatenate([d.mean(0), d.std(0)]).astype(np.float32))
        ids.append(mid)
        if k % 200 == 0:
            print(f"{k}/{len(items)} {time.time() - t0:.0f}s dim={X[-1].size}", flush=True)
    out = OUT.with_suffix(f".part{a.shard}.npz") if a.nshards > 1 else OUT
    np.savez_compressed(out, ids=np.array(ids), X=np.stack(X))
    print("wrote", len(ids), "embeddings")


if __name__ == "__main__":
    main()
