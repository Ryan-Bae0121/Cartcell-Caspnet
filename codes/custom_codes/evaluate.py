"""Metrics for the cartcell rebuild (briefing sections 4-5).

Pooled metrics from a predictions table:
    AUROC, AUPRC        -- label      vs  cls_prob
    Pearson r, MSE      -- affinity   vs  reg_yhat

Plus a per-allele breakdown restricted to alleles with >= min_count samples
(briefing 4.2: "최소 30 샘플 이상 allele만 분석").

A predictions table is a DataFrame / TSV with columns:
    HLA, label, affinity, cls_prob, reg_yhat
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe(fn, *a):
    try:
        v = float(fn(*a))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def pooled_metrics(df):
    label = df["label"].to_numpy(dtype=float)
    affinity = df["affinity"].to_numpy(dtype=float)
    cls_prob = df["cls_prob"].to_numpy(dtype=float)
    reg_yhat = df["reg_yhat"].to_numpy(dtype=float)

    both_classes = 0 < label.sum() < len(label)
    mse = float(np.mean((affinity - reg_yhat) ** 2))
    return {
        "n": int(len(df)),
        "n_pos": int(label.sum()),
        "auroc": _safe(roc_auc_score, label, cls_prob) if both_classes else None,
        "auprc": _safe(average_precision_score, label, cls_prob) if both_classes else None,
        "pearson_r": _safe(lambda: pearsonr(affinity, reg_yhat)[0]),
        "mse": mse,
    }


def per_allele_metrics(df, min_count=30):
    out = {}
    for allele, g in df.groupby("HLA"):
        if len(g) < min_count:
            continue
        out[allele] = pooled_metrics(g)
    return out


def evaluate(df, min_count=30):
    return {
        "pooled": pooled_metrics(df),
        "per_allele": per_allele_metrics(df, min_count),
        "per_allele_summary": _per_allele_summary(per_allele_metrics(df, min_count)),
    }


def _per_allele_summary(pa):
    def _avg(key):
        vals = [m[key] for m in pa.values() if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n_alleles": len(pa),
        "mean_auroc": _avg("auroc"),
        "mean_auprc": _avg("auprc"),
        "mean_pearson_r": _avg("pearson_r"),
        "mean_mse": _avg("mse"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_tsv")
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.pred_tsv, sep="\t")
    res = evaluate(df, args.min_count)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
