"""Raw IEDB MHC-II CSV -> train/test TSV for the cartcell rebuild.

Input  : custom_dataset/Anthem_dataset/raw/merged_data_without.csv
Output : custom_dataset/Anthem_dataset/{train_data.txt,test_data.txt}
         custom_dataset/Anthem_dataset/mhc_ii_pseudo.csv

Steps (briefing sections 1.1-1.3):
  1. keep human rows
  2. aggressive inequality transform  -> ic50_numeric
  3. affinity = 1 - log(ic50_numeric)/log(50000), clipped to [0, 1]
  4. label = 1 if affinity >= threshold (0.8) else 0
  5. keep peptide length in [13, 25]
  6. uppercase peptide, drop non-standard amino acids
  7. normalise allele name, drop alleles with no pseudo-sequence entry
  8. de-duplicate (allele, peptide) by geometric mean of ic50_numeric
  9. stratified train/test split, write TSV
     header: HLA \t peptide \t affinity \t label \t ic50_numeric
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from mhc_pseudo import normalize_allele, build_pseudo_csv
from seq_encoding import STANDARD_AA

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))


def load_config(path=None):
    path = path or os.path.join(HERE, "config", "data.json")
    with open(path) as fh:
        return json.load(fh)


def inequality_to_numeric(meas, ineq):
    """Aggressive transform (briefing 1.2). ``ineq`` in {=, >, >=, <, <=, ''}."""
    ineq = (ineq or "").strip()
    if ineq in (">", ">="):
        return meas * 2.0
    if ineq in ("<", "<="):
        return meas / 2.0
    return meas  # '=' or blank -> exact


def to_affinity(ic50_numeric, log_base=50000):
    ic50 = max(float(ic50_numeric), 1e-8)
    return float(np.clip(1.0 - math.log(ic50) / math.log(log_base), 0.0, 1.0))


def run(cfg, verbose=True):
    raw_path = _resolve(cfg["raw_csv"])
    df = pd.read_csv(raw_path)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"[0] raw rows: {len(df)}")

    # 1. human only
    df = df[df["species"].str.contains("Homo sapiens", case=False, na=False)].copy()
    log(f"[1] human rows: {len(df)}")

    # 2. inequality -> ic50_numeric
    df["ic50_numeric"] = [
        inequality_to_numeric(m, i) for m, i in zip(df["meas"], df["inequality"])
    ]
    df = df[df["ic50_numeric"] > 0].copy()
    log(f"[2] positive ic50_numeric rows: {len(df)}")

    # 3-4. affinity + label
    lb = cfg["affinity_log_base"]
    df["affinity"] = df["ic50_numeric"].map(lambda x: to_affinity(x, lb))
    thr = cfg["affinity_threshold"]
    df["label"] = (df["affinity"] >= thr).astype(int)

    # 5. peptide length filter
    lo, hi = cfg["pep_min_len"], cfg["pep_max_len"]
    df["peptide"] = df["sequence"].astype(str).str.strip().str.upper()
    n_before = len(df)
    df = df[df["peptide"].str.len().between(lo, hi)].copy()
    log(f"[5] length in [{lo},{hi}]: {len(df)}  (dropped {n_before - len(df)}, "
        f"{100*(n_before-len(df))/max(n_before,1):.2f}%)")

    # 6. standard amino acids only
    ok = df["peptide"].map(lambda s: set(s) <= STANDARD_AA)
    log(f"[6] standard-AA peptides: {ok.sum()}  (dropped {(~ok).sum()})")
    df = df[ok].copy()

    # 7. normalise allele, keep those with a pseudo-sequence entry
    df["HLA"] = df["mhc"].map(lambda a: normalize_allele(a, cfg["collapse_dra"]))
    allele_keys = sorted(df["HLA"].unique())
    log(f"[7] alleles after normalisation: {len(allele_keys)} "
        f"(from {df['mhc'].nunique()} raw spellings)")
    table = build_pseudo_csv(allele_keys, _resolve(cfg["pseudo_csv"]))
    keep = df["HLA"].isin(table.keys)
    df = df[keep].copy()
    log(f"    rows with pseudo-seq: {len(df)}  "
        f"({table.n_placeholder}/{len(table)} alleles are placeholders)")

    # 8. de-duplicate (allele, peptide) by geometric mean of ic50_numeric
    n_before = len(df)
    g = (df.groupby(["HLA", "peptide"], as_index=False)
           .agg(ic50_numeric=("ic50_numeric",
                              lambda x: float(np.exp(np.log(x).mean())))))
    g["affinity"] = g["ic50_numeric"].map(lambda x: to_affinity(x, lb))
    g["label"] = (g["affinity"] >= thr).astype(int)
    log(f"[8] de-dup (allele,peptide): {len(g)}  (merged {n_before - len(g)})")

    # strong-binder ratio
    ratio = 100.0 * g["label"].mean()
    log(f"    strong-binder ratio: {ratio:.2f}%  "
        f"(implied pos_weight ~ {(1-g['label'].mean())/max(g['label'].mean(),1e-9):.1f})")

    # 9. stratified split
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(
        g, test_size=cfg["test_size"], random_state=cfg["split_seed"],
        stratify=g["label"],
    )
    cols = ["HLA", "peptide", "affinity", "label", "ic50_numeric"]
    tr_path, te_path = _resolve(cfg["train_txt"]), _resolve(cfg["test_txt"])
    tr[cols].to_csv(tr_path, sep="\t", index=False)
    te[cols].to_csv(te_path, sep="\t", index=False)
    log(f"[9] train: {len(tr)} -> {tr_path}")
    log(f"    test : {len(te)} -> {te_path}")
    log(f"    train strong-binder: {100*tr['label'].mean():.2f}%  "
        f"test strong-binder: {100*te['label'].mean():.2f}%")
    return tr, te


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(load_config(args.config), verbose=not args.quiet)
