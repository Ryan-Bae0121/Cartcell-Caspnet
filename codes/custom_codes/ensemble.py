"""Ensemble of Phase2 + Phase3 (briefing section 2, Ensemble).

Not a new architecture: forward Phase2 and Phase3 independently (already done --
their OOF/test prediction tables are on disk from train_phase2.py/train_phase3.py),
then linearly mix the *outputs* only. No shared parameters, nothing trained here.

    classification: alpha * sigmoid(logit_phase2) + (1-alpha) * sigmoid(logit_phase3)
    regression:     alpha * yhat_phase2           + (1-alpha) * yhat_phase3

``cls_prob`` in the Phase2/Phase3 prediction tables already IS sigmoid(logit), so
the mix is just a weighted average of the two tables' cls_prob / reg_yhat columns.

Briefing default: alpha = 0.5 ("simple averaging -- limited sample count (~34,400)
makes a meta-learner risky to overfit; averaging is safer"). This script reports
that default AND a full alpha sweep (0.0..1.0) on OOF for the per-allele/LOMO
follow-up work to reference.

Usage:
  python ensemble.py
  ALPHA=0.6 python ensemble.py                 # override the default reported alpha
  PHASE2_DIR=... PHASE3_DIR=... python ensemble.py
"""

import json
import os

import numpy as np
import pandas as pd

from evaluate import evaluate, pooled_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ALPHA = 0.5
SWEEP_ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def _resolve(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def load_config():
    return {
        "phase2_dir": os.environ.get("PHASE2_DIR", "reports_phase2_affinity_from_scratch"),
        "phase3_dir": os.environ.get("PHASE3_DIR", "reports_phase3_capsnet"),
        "out_dir": os.environ.get("OUT_DIR", "reports_ensemble"),
        "alpha": float(os.environ.get("ALPHA", DEFAULT_ALPHA)),
    }


def _load_pair(phase2_dir, phase3_dir, filename):
    p2 = pd.read_csv(os.path.join(_resolve(phase2_dir), filename), sep="\t")
    p3 = pd.read_csv(os.path.join(_resolve(phase3_dir), filename), sep="\t")
    if len(p2) != len(p3):
        raise ValueError(f"{filename}: row count mismatch phase2={len(p2)} phase3={len(p3)}")
    if "HLA" in p2.columns and not (p2["HLA"].to_numpy() == p3["HLA"].to_numpy()).all():
        raise ValueError(f"{filename}: HLA column misaligned between phase2/phase3 -- "
                          f"can't ensemble row-wise")
    if "label" in p2.columns and not (p2["label"].to_numpy() == p3["label"].to_numpy()).all():
        raise ValueError(f"{filename}: label column misaligned between phase2/phase3")
    if "affinity" in p2.columns:
        diff = np.abs(p2["affinity"].to_numpy() - p3["affinity"].to_numpy())
        if diff.max() > 1e-3:
            raise ValueError(f"{filename}: affinity column misaligned "
                              f"(max abs diff {diff.max():.4g})")
    return p2, p3


def mix(p2, p3, alpha):
    out = p2[["HLA", "label", "affinity"]].copy() if "HLA" in p2.columns else pd.DataFrame()
    out["cls_prob"] = alpha * p2["cls_prob"].to_numpy() + (1 - alpha) * p3["cls_prob"].to_numpy()
    out["reg_yhat"] = alpha * p2["reg_yhat"].to_numpy() + (1 - alpha) * p3["reg_yhat"].to_numpy()
    return out


def sweep_alpha(p2, p3, alphas=SWEEP_ALPHAS):
    rows = []
    for a in alphas:
        m = pooled_metrics(mix(p2, p3, a))
        rows.append({"alpha": float(a), **m})
    return pd.DataFrame(rows)


def _best(df, col, how="max"):
    d = df.dropna(subset=[col])
    if d.empty:
        return None
    row = d.loc[d[col].idxmax()] if how == "max" else d.loc[d[col].idxmin()]
    return {"alpha": float(row["alpha"]), col: float(row[col])}


def main():
    cfg = load_config()
    out_dir = _resolve(cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    oof2, oof3 = _load_pair(cfg["phase2_dir"], cfg["phase3_dir"], "oof_predictions.tsv")
    test2, test3 = _load_pair(cfg["phase2_dir"], cfg["phase3_dir"], "test_predictions.tsv")
    print(f"OOF rows {len(oof2)} | test rows {len(test2)} | alignment OK")

    sweep = sweep_alpha(oof2, oof3)
    sweep.to_csv(os.path.join(out_dir, "alpha_sweep_oof.csv"), index=False)
    print("\n=== alpha sweep (OOF) ===")
    print(sweep.to_string(index=False))

    best_auroc = _best(sweep, "auroc", "max")
    best_r = _best(sweep, "pearson_r", "max")
    best_mse = _best(sweep, "mse", "min")
    print(f"\nbest AUROC: {best_auroc} | best Pearson r: {best_r} | best (lowest) MSE: {best_mse}")

    alpha = cfg["alpha"]
    oof_df = mix(oof2, oof3, alpha)
    test_df = mix(test2, test3, alpha)
    oof_df.to_csv(os.path.join(out_dir, "oof_predictions.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(out_dir, "test_predictions.tsv"), sep="\t", index=False)

    oof_eval = evaluate(oof_df)
    test_eval = evaluate(test_df)

    summary = {
        "config": cfg,
        "alpha_used": alpha,
        "n_oof": len(oof_df),
        "n_test": len(test_df),
        "alpha_sweep_best": {
            "auroc": best_auroc,
            "pearson_r": best_r,
            "mse": best_mse,
        },
        "oof": oof_eval,
        "test": test_eval,
    }
    spath = os.path.join(out_dir, "ensemble_summary.json")
    with open(spath, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n=== Ensemble @ alpha={alpha} | OOF pooled ===")
    print(json.dumps(oof_eval["pooled"], indent=2))
    print(f"=== Ensemble @ alpha={alpha} | TEST pooled ===")
    print(json.dumps(test_eval["pooled"], indent=2))
    print(f"\nsummary -> {spath}")


if __name__ == "__main__":
    main()
