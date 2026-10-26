"""CGCNN-style crystal-graph classifier of magnetic ordering (FM / AFM / FiM / NM), trained from scratch on
the graphs of build_graphs.py with the shared split of outputs/classifier_split.json.

    python src/gnn/train_cgcnn.py --seed 0 [--holdout-dft]     # env: magnetic-mlip-mattersim (torch + PyG)

Architecture (Xie & Grossman, PRL 120, 145301 (2018)): learned element embedding (64) -> linear to 128;
4 gated graph convolutions v_i <- v_i + sum_j sigmoid(W_f z_ij) * softplus(W_s z_ij), z_ij = [v_i, v_j, e_ij],
with a 40-Gaussian expansion of the bond length (0-8 A) as e_ij, batch normalisation and residual updates;
mean pooling; 2-layer MLP head. Class-weighted cross-entropy (weights n/(4 n_c), the 'balanced' weighting of
the LightGBM baseline); AdamW, lr 1e-3, cosine schedule, 120 epochs; 10% of the training ids are held out
(stratified, seeded) and the epoch with the best validation balanced accuracy is kept.
"""
import argparse
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from common import CLASSES, ROOT, dft_candidates, evaluate, load_split, write_dft_predictions

NG, RCUT = 40, 8.0


def to_data(g, y):
    return Data(z=torch.as_tensor(g["z"].astype(np.int64)), edge_index=torch.as_tensor(g["edge_index"].astype(np.int64)),
                edge_len=torch.as_tensor(g["edge_len"]), y=torch.tensor([y]), num_nodes=len(g["z"]))


class Conv(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.lin = nn.Linear(2 * h + NG, 2 * h)
        self.bn1, self.bn2 = nn.BatchNorm1d(2 * h), nn.BatchNorm1d(h)

    def forward(self, v, ei, e):
        src, dst = ei
        f, s = self.bn1(self.lin(torch.cat([v[dst], v[src], e], 1))).chunk(2, 1)
        m = torch.zeros_like(v).index_add_(0, dst, torch.sigmoid(f) * F.softplus(s))
        return F.softplus(v + self.bn2(m))


class CGCNN(nn.Module):
    def __init__(self, h=128, n_conv=4, n_cls=4):
        super().__init__()
        self.emb = nn.Embedding(100, 64)
        self.inp = nn.Linear(64, h)
        self.convs = nn.ModuleList(Conv(h) for _ in range(n_conv))
        self.head = nn.Sequential(nn.Linear(h, h), nn.Softplus(), nn.Dropout(0.1), nn.Linear(h, n_cls))
        self.register_buffer("mu", torch.linspace(0, RCUT, NG))

    def forward(self, b):
        e = torch.exp(-((b.edge_len[:, None] - self.mu) ** 2) / (2 * (RCUT / NG) ** 2))
        v = self.inp(self.emb(b.z))
        for c in self.convs:
            v = c(v, b.edge_index, e)
        return self.head(global_mean_pool(v, b.batch))


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    return torch.cat([F.softmax(model(b.to(dev)), 1).cpu() for b in loader]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--holdout-dft", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda")
    G = pickle.load(open(ROOT / "data" / "processed" / "gnn_graphs.pkl", "rb"))
    gr, lab, src = G["graphs"], G["label"], G["source"]
    train_ids, test_ids = load_split(a.holdout_dft)
    train_ids = [m for m in train_ids if m in gr]
    test_ids = [m for m in test_ids if m in gr]
    y = {m: CLASSES.index(lab[m]) for m in gr}
    tr, va = train_test_split(train_ids, test_size=0.1, random_state=a.seed, stratify=[y[m] for m in train_ids])
    mk = lambda ids: [to_data(gr[m], y[m]) for m in ids]  # noqa: E731
    dl_tr = DataLoader(mk(tr), batch_size=128, shuffle=True)
    dl_va, dl_te = DataLoader(mk(va), batch_size=256), DataLoader(mk(test_ids), batch_size=256)
    n_c = np.bincount([y[m] for m in tr], minlength=4)
    w = torch.tensor(len(tr) / (4 * n_c), dtype=torch.float32, device=dev)

    model = CGCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(dl_tr))
    best, best_state, yv = -1, None, np.array([y[m] for m in va])
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for b in dl_tr:
            b = b.to(dev)
            loss = F.cross_entropy(model(b), b.y, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        bal = balanced_accuracy_score(yv, predict(model, dl_va, dev).argmax(1))
        if bal > best:
            best, best_ep = bal, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:3d} loss {loss.item():.3f} val bal {bal:.4f} (best {best:.4f} @ {best_ep}) {time.time() - t0:.0f}s",
                  flush=True)
    model.load_state_dict(best_state)
    p = predict(model, dl_te, dev)
    tag = f"cgcnn{'_holdout' if a.holdout_dft else ''}_s{a.seed}"
    s = evaluate(tag, test_ids, [src[m] for m in test_ids], [lab[m] for m in test_ids],
                 [CLASSES[k] for k in p.argmax(1)], p,
                 extra={"n_train": len(tr), "n_val": len(va), "best_epoch": best_ep, "val_balanced_accuracy": best,
                        "seed": a.seed, "holdout_dft": a.holdout_dft})
    print(f"{tag}: test acc {s['accuracy']:.4f} bal {s['balanced_accuracy']:.4f}")
    if a.holdout_dft:
        ids = [m for m in dft_candidates().candidate_id if m in gr]
        write_dft_predictions(tag, ids, predict(model, DataLoader(mk(ids), batch_size=64), dev))
    torch.save(model.state_dict(), ROOT / "data" / "processed" / f"{tag}.pt")


if __name__ == "__main__":
    main()
