"""LightGBM classifiers of the comparison, all on the shared split (outputs/classifier_split.json):

  lightgbm_v2       the paper's screening classifier, refit with the identical settings of
                    train_eval_lightgbm_v2.py (composition + symmetry + local magnetic geometry)
  mace_lgbm         LightGBM on the frozen MACE-MP-0 embeddings only (mace_embed.py)
  hybrid_lgbm       LightGBM on the v2 features concatenated with the MACE-MP-0 embeddings

    python src/gnn/train_heads.py [--holdout-dft]      # env: magnetic-acq

Scored on the test ids that carry a crystal graph (7,161 of 7,193; the 32 others are disordered MAGNDATA
structures) so every row of the comparison sees the same records; lightgbm_v2 is also scored on all 7,193.
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, ROOT, dft_candidates, evaluate, load_split, write_dft_predictions  # noqa: E402

CAT = ["space_group_number", "crystal_system_bucket"]


def fit_predict(Xtr, ytr, Xte, cat):
    m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, objective="multiclass",
                           class_weight="balanced", random_state=42, verbose=-1)
    m.fit(Xtr, ytr, categorical_feature=cat or "auto")
    return m, m.predict_proba(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-dft", action="store_true")
    a = ap.parse_args()
    hd = "_holdout" if a.holdout_dft else ""
    df = pd.read_csv(ROOT / "data" / "features" / "baseline_features_v2.csv").set_index("material_id")
    fcols = [c for c in df.columns if c not in ("source_database", "ordering")]
    for c in CAT:
        df[c] = df[c].astype("category")
    emb = np.load(ROOT / "data" / "features" / "mace_mp0_embeddings.npz")
    E = pd.DataFrame(emb["X"], index=emb["ids"], columns=[f"mace_{i}" for i in range(emb["X"].shape[1])])
    train, test = load_split(a.holdout_dft)
    test_g = [m for m in test if m in E.index]
    y = df.ordering.map(CLASSES.index)
    dft = [m for m in dft_candidates().candidate_id]
    runs = {"lightgbm_v2": (df[fcols], CAT, train),
            "mace_lgbm": (E, None, [m for m in train if m in E.index]),
            "hybrid_lgbm": (df.loc[E.index, fcols].join(E), CAT, [m for m in train if m in E.index])}
    for name, (X, cat, tr) in runs.items():
        model, p = fit_predict(X.loc[tr], y.loc[tr], X.loc[test_g], cat)
        s = evaluate(f"{name}{hd}_graphsubset", test_g, df.loc[test_g, "source_database"], df.loc[test_g, "ordering"],
                     [CLASSES[k] for k in p.argmax(1)], p, extra={"n_train": len(tr)})
        print(f"{name}{hd} (graph subset, n={len(test_g)}): acc {s['accuracy']:.4f} bal {s['balanced_accuracy']:.4f}"
              f" MAGNDATA bal {s['per_source']['MAGNDATA']['balanced_accuracy']:.4f}")
        if name == "lightgbm_v2" and not a.holdout_dft:
            p_all = model.predict_proba(X.loc[test])
            s = evaluate("lightgbm_v2_refit", test, df.loc[test, "source_database"], df.loc[test, "ordering"],
                         [CLASSES[k] for k in p_all.argmax(1)], p_all)
            print(f"  lightgbm_v2 refit on all 7,193: acc {s['accuracy']:.4f} bal {s['balanced_accuracy']:.4f}")
        if a.holdout_dft:
            ids = [m for m in dft if m in X.index]
            write_dft_predictions(f"{name}{hd}", ids, model.predict_proba(X.loc[ids]))


if __name__ == "__main__":
    main()
