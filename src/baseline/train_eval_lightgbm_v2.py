"""Train + evaluate LightGBM on magnetic ordering classification using the structure-aware
feature set (build_features_v2.py). See that script's docstring for what Tier 1 (broad
crystal-symmetry fields) and Tier 2 (MAGNDATA-only local-geometry fields) actually add over the
v1 composition-only baseline (train_eval_lightgbm.py).

Reports overall metrics AND a source-database breakdown, since Tier 2 features only apply to the
~3% of rows with a `structure` (all MAGNDATA) -- if there's a real effect, it should show up
disproportionately in MAGNDATA's per-source numbers, not necessarily in the aggregate.

Usage:
    python train_eval_lightgbm_v2.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
CATEGORICAL_COLS = ["space_group_number", "crystal_system_bucket"]


def main():
    df = pd.read_csv(ROOT / "data" / "features" / "baseline_features_v2.csv")
    print(f"Loaded {len(df)} rows")

    feature_cols = [c for c in df.columns if c not in ("material_id", "source_database", "ordering")]
    X = df[feature_cols].copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    y_raw = df["ordering"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print("Classes:", list(encoder.classes_))

    X_train, X_test, y_train, y_test, id_train, id_test, src_train, src_test = train_test_split(
        X, y, df["material_id"], df["source_database"], test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        objective="multiclass",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLS)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)

    print(f"\nOverall accuracy: {acc:.4f}")
    print(f"Overall balanced accuracy: {bal_acc:.4f}")
    print(f"\n(v1 composition-only baseline for comparison: accuracy 0.5674, balanced accuracy 0.6070)\n")
    print(classification_report(y_test, preds, target_names=encoder.classes_))

    print("\nPer-source-database accuracy on test set:")
    per_source = {}
    for src in sorted(src_test.unique()):
        mask = (src_test == src).values
        src_acc = accuracy_score(y_test[mask], preds[mask])
        src_bal_acc = balanced_accuracy_score(y_test[mask], preds[mask])
        per_source[src] = {"n": int(mask.sum()), "accuracy": src_acc, "balanced_accuracy": src_bal_acc}
        print(f"  {src:20s} n={mask.sum():5d}  accuracy={src_acc:.4f}  balanced_accuracy={src_bal_acc:.4f}")

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 15 feature importances:")
    print(importance.head(15))

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out_dir / "lightgbm_v2_feature_importance.csv")

    with open(out_dir / "lightgbm_v2_eval_summary.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "classes": list(encoder.classes_),
            "class_balance": y_raw.value_counts().to_dict(),
            "per_source": per_source,
            "v1_comparison": {"accuracy": 0.5674, "balanced_accuracy": 0.6070},
        }, f, indent=2)
    print(f"\nWrote feature importance + eval summary to {out_dir}")


if __name__ == "__main__":
    main()
