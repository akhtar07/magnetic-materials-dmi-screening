"""Train + evaluate LightGBM on magnetic ordering classification, in ONE script.

This is deliberately a single script, not train_X.py + evaluate_X.py as two separate files --
the old repo's version of this (scripts/train_lightgbm.py + scripts/evaluate_lightgbm.py)
silently pointed at two different datasets (master_dataset_v4.json vs the older, smaller
transition_metals_v1.json/transition_features_v1.csv) because nothing enforced them staying
in sync. Train and eval sharing one script/one train_test_split call structurally prevents
that class of bug from recurring.

Target: Verma, Jami & Bhattacharya (2025, arXiv:2507.01913) report 82.4% accuracy on magnetic
ordering classification with a similar enriched-elemental-vector + LightGBM approach on 5,741
MP compounds. This should be viewed as a rough benchmark, not an exact comparison: different
label set (their FM/AFM binary-ish framing vs our FM/FiM/NM/AFM 4-class), different data size
and source mix.

Usage:
    python train_eval_lightgbm.py
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


def main():
    df = pd.read_csv(ROOT / "data" / "features" / "baseline_features_v1.csv")
    print(f"Loaded {len(df)} rows")

    feature_cols = [c for c in df.columns if c not in ("material_id", "source_database", "ordering")]
    X = df[feature_cols]
    y_raw = df["ordering"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print("Classes:", list(encoder.classes_))

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df["material_id"], test_size=0.2, random_state=42, stratify=y
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
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"\nReference: Verma et al. (2025) report 82.4% accuracy on a similar but not "
          f"identical setup -- see module docstring for why this isn't an apples-to-apples number.\n")
    print(classification_report(y_test, preds, target_names=encoder.classes_))

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 feature importances:")
    print(importance.head(10))

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out_dir / "lightgbm_feature_importance.csv")

    with open(out_dir / "lightgbm_eval_summary.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "classes": list(encoder.classes_),
            "class_balance": y_raw.value_counts().to_dict(),
        }, f, indent=2)
    print(f"\nWrote feature importance + eval summary to {out_dir}")


if __name__ == "__main__":
    main()
