"""Aggregate the classifier comparison into outputs/classifier_comparison.{csv,json}.

Every model is scored on the 7,161 shared test ids that carry a crystal graph. CGCNN rows are the mean and
standard deviation over three seeds; the LightGBM heads are deterministic. The DFT columns come from the runs
with the 26 DFT-validated candidates held out of training: how many of the 21 overturned candidates each model
calls non-FM (AFM or FiM), and how many of the 3 confirmed ferromagnets it calls FM.

    python src/gnn/compare_models.py      # env: magnetic-acq
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, OUT  # noqa: E402

MODELS = [("LightGBM v2 (screening)", "lightgbm_v2{h}_graphsubset", [None]),
          ("LightGBM, MACE-MP-0 embedding", "mace_lgbm{h}_graphsubset", [None]),
          ("LightGBM, v2 + MACE-MP-0", "hybrid_lgbm{h}_graphsubset", [None]),
          ("CGCNN", "cgcnn{h}_s{s}", [0, 1, 2])]


def row(summ, holdout_tag):
    # base rate from the held-out run, i.e. the same model whose DFT-candidate predictions are counted
    p = pd.read_csv(OUT / f"{holdout_tag}_test_predictions.csv")
    mpfm = p[(p.source_database == "MaterialsProject") & (p.y_true == "FM")]
    r = {"mp_FM_called_nonFM": float(mpfm.y_pred.isin(["AFM", "FiM"]).mean()), "accuracy": summ["accuracy"], "balanced_accuracy": summ["balanced_accuracy"],
         "magndata_accuracy": summ["per_source"]["MAGNDATA"]["accuracy"],
         "magndata_balanced_accuracy": summ["per_source"]["MAGNDATA"]["balanced_accuracy"],
         "mp_balanced_accuracy": summ["per_source"]["MaterialsProject"]["balanced_accuracy"]}
    r.update({f"recall_{c}": summ["recall"][c] for c in CLASSES})
    return r, summ["n_test"]


def dft_counts(df):
    over = df[df.verdict == "overturned"]
    conf = df[df.verdict == "fm_confirmed"]
    return {"overturned_called_nonFM": int(over.y_pred.isin(["AFM", "FiM"]).sum()),
            "overturned_called_AFM": int((over.y_pred == "AFM").sum()), "n_overturned": len(over),
            "confirmed_called_FM": int((conf.y_pred == "FM").sum()), "n_confirmed": len(conf)}


def main():
    rows, js = [], {}
    for name, pat, seeds in MODELS:
        recs, dft, n = [], [], None
        for s in seeds:
            tag = pat.format(h="", s=s)
            r, n = row(json.load(open(OUT / f"{tag}_eval_summary.json")), pat.format(h="_holdout", s=s))
            recs.append(r)
            base = pat.format(h="_holdout", s=s).replace("_graphsubset", "")
            dft.append(dft_counts(pd.read_csv(OUT / f"{base}_dft_candidates.csv")))
        R, D = pd.DataFrame(recs), pd.DataFrame(dft)
        out = {"model": name, "n_seeds": len(seeds), "n_test": n}
        for c in R:
            out[c] = R[c].mean()
            out[c + "_sd"] = R[c].std(ddof=1) if len(R) > 1 else np.nan
        for c in D:
            out["dft_" + c] = D[c].tolist() if len(D) > 1 else int(D[c].iloc[0])
        rows.append(out)
        js[name] = out
    p = pd.read_csv(OUT / "lightgbm_v2_graphsubset_test_predictions.csv")
    js["_magndata_test_counts"] = p[p.source_database == "MAGNDATA"].y_true.value_counts().to_dict()
    print("MAGNDATA test counts:", js["_magndata_test_counts"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "classifier_comparison.csv", index=False)
    json.dump(js, open(OUT / "classifier_comparison.json", "w"), indent=2, default=lambda x: None if x != x else float(x))
    pd.set_option("display.width", 250)
    print(df[["model", "accuracy", "accuracy_sd", "balanced_accuracy", "balanced_accuracy_sd",
              "magndata_balanced_accuracy", "magndata_balanced_accuracy_sd", "recall_AFM", "recall_FM", "recall_FiM",
              "recall_NM", "mp_FM_called_nonFM", "dft_overturned_called_nonFM", "dft_overturned_called_AFM", "dft_confirmed_called_FM"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
