"""Crystal graphs for the GNN ordering classifiers, from the same labelled records and structures as the
LightGBM features (build_features_v2.py), and the shared train/test split every classifier is scored on.

    python src/gnn/build_graphs.py      # env: magnetic-acq (pymatgen, Python >= 3.10)

Outputs
  outputs/classifier_split.json        test ids = the 7,193 records of outputs/lightgbm_v2_test_predictions.csv
                                       (the stratified 80/20 split, random_state=42, of train_eval_lightgbm_v2.py);
                                       train ids = every other labelled record
  data/processed/gnn_graphs.pkl        per record: atomic numbers, fractional coordinates, lattice, and the
                                       k-nearest-neighbour edges (k=12 within 8 A, periodic) with their lengths
Only species, positions and lattice are used -- never magmom_per_site, from which MAGNDATA's labels are derived.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "baseline"))
from build_features_v2 import attach_structures  # noqa: E402

RCUT, KMAX = 8.0, 12


def graph(item):
    mid, sd = item
    try:
        s = Structure.from_dict(sd)
        ci, ni, img, d = s.get_neighbor_list(RCUT)
        keep = []
        for i in range(len(s)):  # k nearest neighbours of each centre
            sel = np.flatnonzero(ci == i)
            keep.append(sel[np.argsort(d[sel], kind="stable")[:KMAX]])
        keep = np.concatenate(keep) if keep else np.array([], int)
        return mid, dict(z=np.array(s.atomic_numbers, np.int16), frac=s.frac_coords.astype(np.float32),
                         lattice=s.lattice.matrix.astype(np.float32),
                         edge_index=np.stack([ni[keep], ci[keep]]).astype(np.int32),  # source -> centre
                         edge_len=d[keep].astype(np.float32))
    except Exception as e:  # noqa: BLE001
        return mid, {"error": repr(e)}


def main():
    feats = pd.read_csv(ROOT / "data" / "features" / "baseline_features_v2.csv",
                        usecols=["material_id", "source_database", "ordering"])
    test = pd.read_csv(ROOT / "outputs" / "lightgbm_v2_test_predictions.csv").material_id
    assert test.isin(feats.material_id).all() and len(test) == 7193
    split = dict(test=sorted(test), train=sorted(set(feats.material_id) - set(test)),
                 note="test = lightgbm_v2 test set (stratified 80/20, random_state=42); shared by every classifier")
    json.dump(split, open(ROOT / "outputs" / "classifier_split.json", "w"))

    records = json.load(open(ROOT / "data" / "processed" / "harmonized_v2.json"))
    attach_structures(records)
    want = set(feats.material_id)
    items = [(r["material_id"], r["structure"]) for r in records
             if r["material_id"] in want and r.get("structure") is not None]
    print(f"{len(items)} of {len(want)} labelled records carry a structure")
    out, bad = {}, {}
    with Pool(48) as pool:
        for mid, g in pool.imap_unordered(graph, items, chunksize=32):
            (bad if "error" in g else out)[mid] = g
    lab = feats.set_index("material_id")
    data = {"rcut": RCUT, "kmax": KMAX, "graphs": out,
            "label": {m: lab.at[m, "ordering"] for m in out}, "source": {m: lab.at[m, "source_database"] for m in out},
            "failed": bad}
    pickle.dump(data, open(ROOT / "data" / "processed" / "gnn_graphs.pkl", "wb"), protocol=4)
    n_test = sum(m in out for m in split["test"])
    print(f"graphs: {len(out)} ok, {len(bad)} failed; test ids with a graph: {n_test}/{len(split['test'])}")
    print("isolated-atom graphs (no edge within cutoff):", sum(g["edge_index"].shape[1] == 0 for g in out.values()))


if __name__ == "__main__":
    main()
