"""Check an incoming dataset TSV against what DataProvider/the models actually require.

Hand this to whoever is preprocessing: run it on your own file before sending it,
and it either prints PASS or tells you exactly which rows are wrong. Everything
checked here is something that would otherwise either crash training or silently
change the labels.

  python validate_dataset.py my_data.txt              # check
  python validate_dataset.py my_data.txt --fix out.txt # write a repaired copy
  python validate_dataset.py a.txt b.txt --compare     # compare several files

Exit code is 0 on pass, 1 on any error, so it drops into a script or CI.

Checks (E = error, blocks training; W = warning, worth a look):
  E  5 columns, exact names, exact order (data_provider.py indexes by position:
     df.iat[i, 0] is HLA and df.iat[i, 1] is peptide, so a reordered or inserted
     column silently feeds the wrong string into the encoder)
  E  HLA present in mhc_ii_pseudo.csv (otherwise DataProvider raises KeyError);
     normalize_allele() is tried first and the suggested key is reported
  E  peptide length 13-25, standard 20 AA only, uppercase, no gaps/X/U
  E  affinity in [0, 1] and finite; label in {0, 1}
  E  no NaN in any required column
  W  affinity disagrees with 1 - log(ic50)/log(50000) beyond tolerance
  W  label disagrees with affinity >= 0.8
  W  duplicate (HLA, peptide) pairs
  W  no positives at all, or fewer than 2 (ranking metrics are undefined /
     degenerate -- see lomo.py's summary_min_pos)
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

from mhc_pseudo import PseudoSeqTable, normalize_allele
from seq_encoding import STANDARD_AA

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["HLA", "peptide", "affinity", "label", "ic50_numeric"]
PEP_MIN, PEP_MAX = 13, 25
LOG_BASE = 50000
AFF_THRESHOLD = 0.8
PSEUDO_CSV = os.path.normpath(os.path.join(
    HERE, "..", "..", "custom_dataset", "Anthem_dataset", "mhc_ii_pseudo.csv"))


def to_affinity(ic50, log_base=LOG_BASE):
    if not np.isfinite(ic50) or ic50 <= 0:
        return None
    return float(min(max(1.0 - math.log(ic50) / math.log(log_base), 0.0), 1.0))


class Report:
    def __init__(self, path):
        self.path = path
        self.errors = []
        self.warnings = []
        self.info = []

    def err(self, msg, rows=None):
        self.errors.append((msg, rows))

    def warn(self, msg, rows=None):
        self.warnings.append((msg, rows))

    @property
    def ok(self):
        return not self.errors

    def show(self, max_rows=5):
        name = os.path.basename(self.path)
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        for line in self.info:
            print(f"  {line}")
        for tag, items in (("ERROR", self.errors), ("WARN", self.warnings)):
            for msg, rows in items:
                print(f"  [{tag}] {msg}")
                if rows is not None and len(rows):
                    shown = [int(r) for r in rows[:max_rows]]
                    more = f" ... (+{len(rows) - len(shown)} more)" if len(rows) > len(shown) else ""
                    print(f"          rows (0-based): {shown}{more}")
        verdict = "PASS" if self.ok else f"FAIL ({len(self.errors)} error type(s))"
        if self.ok and self.warnings:
            verdict += f" with {len(self.warnings)} warning(s)"
        print(f"  -> {verdict}")


def validate(path, pseudo_keys, strict_header=True):
    rep = Report(path)
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as exc:
        rep.err(f"cannot read as TSV: {exc}")
        return rep, None

    rep.info.append(f"{len(df)} rows, {len(df.columns)} columns")

    # --- header / column order -------------------------------------------------
    cols = list(df.columns)
    if cols[:5] != COLUMNS:
        rep.err(f"first 5 columns must be exactly {COLUMNS} in this order, got {cols[:5]}. "
                "DataProvider reads HLA/peptide by position, so order matters; "
                "put any extra columns AFTER these five.")
        if set(COLUMNS) - set(cols):
            rep.err(f"missing column(s): {sorted(set(COLUMNS) - set(cols))}")
            return rep, None
        df = df[COLUMNS + [c for c in cols if c not in COLUMNS]]
    elif len(cols) > 5:
        rep.info.append(f"extra trailing columns (ignored by the loader): {cols[5:]}")

    if df.empty:
        rep.err("file has no data rows")
        return rep, None

    # --- numeric parsing -------------------------------------------------------
    num = {}
    for c in ("affinity", "label", "ic50_numeric"):
        num[c] = pd.to_numeric(df[c], errors="coerce")
        bad = np.where(num[c].isna() & (df[c].str.strip() != ""))[0]
        if len(bad):
            rep.err(f"{c}: {len(bad)} value(s) not numeric", bad)
        blank = np.where(df[c].str.strip() == "")[0]
        if len(blank):
            rep.err(f"{c}: {len(blank)} empty value(s)", blank)

    aff, lab, ic50 = num["affinity"], num["label"], num["ic50_numeric"]

    # --- HLA -------------------------------------------------------------------
    hla = df["HLA"].str.strip()
    unknown, suggestions = [], {}
    for i, raw in enumerate(hla):
        if raw in pseudo_keys:
            continue
        try:
            norm = normalize_allele(raw)
        except Exception:
            norm = None
        if norm in pseudo_keys:
            suggestions.setdefault(raw, norm)
        else:
            unknown.append(i)
    if suggestions:
        rep.err(f"{len(suggestions)} allele spelling(s) need normalising before training "
                f"(normalize_allele fixes them): "
                + ", ".join(f"{k!r} -> {v!r}" for k, v in list(suggestions.items())[:5]))
    if unknown:
        bad_names = sorted(set(hla.iloc[unknown]))
        rep.err(f"{len(unknown)} row(s) have an allele with NO pseudo-sequence -- "
                f"DataProvider will raise KeyError. Add them to mhc_ii_pseudo.csv or drop "
                f"the rows. Unmatched: {bad_names[:5]}", unknown)
    rep.info.append(f"{hla.nunique()} distinct alleles")

    # --- peptide ---------------------------------------------------------------
    pep = df["peptide"].str.strip()
    lens = pep.str.len()
    bad_len = np.where((lens < PEP_MIN) | (lens > PEP_MAX))[0]
    if len(bad_len):
        rep.err(f"{len(bad_len)} peptide(s) outside length {PEP_MIN}-{PEP_MAX} "
                f"(min {int(lens.min())}, max {int(lens.max())})", bad_len)
    bad_aa = np.where([bool(set(p) - STANDARD_AA) for p in pep])[0]
    if len(bad_aa):
        offending = sorted({ch for p in pep.iloc[bad_aa] for ch in set(p) - STANDARD_AA})
        rep.err(f"{len(bad_aa)} peptide(s) contain non-standard character(s) {offending} "
                f"(only the 20 standard AA, uppercase)", bad_aa)
    rep.info.append(f"peptide length: min {int(lens.min())}, median {int(lens.median())}, "
                    f"max {int(lens.max())}")

    # --- affinity / label ------------------------------------------------------
    bad_aff = np.where(aff.notna() & ((aff < 0) | (aff > 1)))[0]
    if len(bad_aff):
        rep.err(f"{len(bad_aff)} affinity value(s) outside [0, 1]", bad_aff)
    bad_lab = np.where(lab.notna() & ~lab.isin([0, 1]))[0]
    if len(bad_lab):
        rep.err(f"{len(bad_lab)} label value(s) not 0/1", bad_lab)

    recomputed = np.array([to_affinity(v) if np.isfinite(v) else None for v in ic50.fillna(np.nan)],
                          dtype=object)
    cmp_mask = np.array([r is not None for r in recomputed]) & aff.notna().to_numpy()
    if cmp_mask.any():
        diff = np.abs(np.array([r for r, m in zip(recomputed, cmp_mask) if m], dtype=float)
                      - aff.to_numpy()[cmp_mask])
        off = np.where(cmp_mask)[0][diff > 1e-3]
        if len(off):
            rep.warn(f"{len(off)} row(s) where affinity != 1 - log(ic50)/log({LOG_BASE}) "
                     f"(max deviation {diff.max():.4f}) -- different transform or stale column", off)

    lab_mask = aff.notna() & lab.notna()
    implied = (aff[lab_mask] >= AFF_THRESHOLD).astype(int)
    mism = np.where(lab_mask)[0][(implied.to_numpy() != lab[lab_mask].to_numpy())]
    if len(mism):
        rep.warn(f"{len(mism)} row(s) where label != (affinity >= {AFF_THRESHOLD}) -- "
                 f"a different positive threshold changes what the model is trained on", mism)

    n_pos = int((lab == 1).sum())
    rep.info.append(f"positives: {n_pos} / {len(df)} ({100 * n_pos / len(df):.2f}%)"
                    + (f" | implied pos_weight ~ {(len(df) - n_pos) / max(n_pos, 1):.1f}" if n_pos else ""))
    if n_pos == 0:
        rep.warn("no positive rows: AUROC/AUPRC are undefined for this file")
    elif n_pos < 2:
        rep.warn(f"only {n_pos} positive row(s): ranking metrics are degenerate "
                 "(AUROC is a function of that one peptide's rank)")

    # --- duplicates ------------------------------------------------------------
    dup = df.duplicated(subset=["HLA", "peptide"], keep=False)
    if dup.any():
        rep.warn(f"{int(dup.sum())} row(s) share an (HLA, peptide) pair "
                 f"({df[dup].groupby(['HLA', 'peptide']).ngroups} distinct pairs) -- "
                 "duplicates inflate whichever split they land in",
                 np.where(dup.to_numpy())[0])

    per_allele = df.assign(_l=lab).groupby(hla)["_l"].agg(["size", "sum"])
    thin = per_allele[per_allele["size"] < 50]
    if len(thin):
        rep.warn(f"{len(thin)} allele(s) with < 50 rows (below lomo.json min_test_samples=50, "
                 f"so they will not get a LOMO fold): {sorted(thin.index)[:5]}")
    return rep, df


def repair(df, pseudo_keys):
    """Best-effort fixes: column order, whitespace, allele normalisation, uppercase."""
    out = df.copy()
    out["HLA"] = out["HLA"].str.strip().map(
        lambda s: s if s in pseudo_keys else (normalize_allele(s) or s))
    out["peptide"] = out["peptide"].str.strip().str.upper()
    extra = [c for c in out.columns if c not in COLUMNS]
    return out[COLUMNS + extra]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pseudo-csv", default=PSEUDO_CSV)
    ap.add_argument("--fix", default=None,
                    help="write a repaired copy (single input only); does not invent data")
    ap.add_argument("--compare", action="store_true",
                    help="print a side-by-side summary of the inputs")
    args = ap.parse_args()

    pseudo_keys = PseudoSeqTable.load(args.pseudo_csv).keys
    print(f"pseudo table: {len(pseudo_keys)} allele keys from {args.pseudo_csv}")

    reports, frames = [], {}
    for p in args.paths:
        rep, df = validate(p, pseudo_keys)
        rep.show()
        reports.append(rep)
        if df is not None:
            frames[os.path.basename(p)] = df

    if args.compare and len(frames) > 1:
        rows = []
        for name, df in frames.items():
            lab = pd.to_numeric(df["label"], errors="coerce")
            lens = df["peptide"].str.strip().str.len()
            rows.append({
                "file": name, "rows": len(df), "alleles": df["HLA"].nunique(),
                "peptides": df["peptide"].nunique(),
                "pos": int((lab == 1).sum()),
                "pos_pct": round(100 * float((lab == 1).mean()), 2),
                "len_min": int(lens.min()), "len_med": int(lens.median()), "len_max": int(lens.max()),
            })
        print(f"\n{'=' * 70}\ncomparison\n{'=' * 70}")
        print(pd.DataFrame(rows).to_string(index=False))
        sets = {n: set(zip(d["HLA"], d["peptide"])) for n, d in frames.items()}
        names = list(sets)
        print("\nshared (HLA, peptide) pairs:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = sets[names[i]], sets[names[j]]
                inter = len(a & b)
                print(f"  {names[i]} vs {names[j]}: {inter} shared "
                      f"({100 * inter / max(min(len(a), len(b)), 1):.1f}% of the smaller)")

    if args.fix:
        if len(frames) != 1:
            raise SystemExit("--fix takes exactly one input file")
        df = repair(list(frames.values())[0], pseudo_keys)
        df.to_csv(args.fix, sep="\t", index=False)
        print(f"\nrepaired copy -> {args.fix}")
        rep, _ = validate(args.fix, pseudo_keys)
        rep.show()
        reports.append(rep)

    failed = [r.path for r in reports if not r.ok]
    print(f"\n{len(reports) - len(failed)}/{len(reports)} file(s) passed")
    if failed:
        print("failed: " + ", ".join(os.path.basename(p) for p in failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
