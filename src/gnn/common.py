"""Shared pieces of the classifier comparison: the split, the label encoding, and the evaluation writer that
puts every model's predictions and summary in outputs/ in the shape of the LightGBM v2 files."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
CLASSES = ["AFM", "FM", "FiM", "NM"]


def load_split(holdout_dft=False):
    """Train/test ids (outputs/classifier_split.json). With holdout_dft, the 26 DFT-validated candidates are
    removed from training so they can serve as an external test against the DFT verdicts."""
    split = json.load(open(OUT / "classifier_split.json"))
    train, test = list(split["train"]), list(split["test"])
    if holdout_dft:
        dft = set(pd.read_csv(OUT / "dft_verdicts.csv").candidate_id)
        train = [m for m in train if m not in dft]
    return train, test


def dft_candidates():
    return pd.read_csv(OUT / "dft_verdicts.csv")[["candidate_id", "verdict"]]


def evaluate(tag, ids, sources, y_true, y_pred, proba=None, extra=None):
    """Write outputs/<tag>_test_predictions.csv and outputs/<tag>_eval_summary.json; return the summary."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    df = pd.DataFrame({"material_id": ids, "source_database": sources, "y_true": y_true, "y_pred": y_pred})
    if proba is not None:
        for k, c in enumerate(CLASSES):
            df[f"p_{c}"] = np.asarray(proba)[:, k]
    df.to_csv(OUT / f"{tag}_test_predictions.csv", index=False)
    per_source = {}
    for s, g in df.groupby("source_database"):
        per_source[s] = {"n": len(g), "accuracy": accuracy_score(g.y_true, g.y_pred),
                         "balanced_accuracy": balanced_accuracy_score(g.y_true, g.y_pred)}
    summ = {"accuracy": accuracy_score(y_true, y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "n_test": len(df), "classes": CLASSES,
            "recall": dict(zip(CLASSES, recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0))),
            "per_source": per_source,
            "confusion_matrix": {"rows_true_cols_pred": CLASSES,
                                 "counts": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()}}
    summ.update(extra or {})
    json.dump(summ, open(OUT / f"{tag}_eval_summary.json", "w"), indent=2, default=float)
    return summ


def write_dft_predictions(tag, ids, proba):
    """Predicted class and probabilities for the DFT candidates (holdout runs)."""
    d = dft_candidates().set_index("candidate_id")
    df = pd.DataFrame(np.asarray(proba), columns=[f"p_{c}" for c in CLASSES], index=ids)
    df["y_pred"] = [CLASSES[k] for k in np.asarray(proba).argmax(1)]
    df = df.join(d, how="inner")
    df.index.name = "candidate_id"
    df.to_csv(OUT / f"{tag}_dft_candidates.csv")
    return df
