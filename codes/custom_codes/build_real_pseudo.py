"""Replace placeholder MHC-II pseudo-sequences with the real 34-mers.

Source: DeepMHCII's vendored copy of the NetMHCIIpan 2016 pseudo-sequence table
(``pseudosequence.2016.all.X.dat``, 5,640 alleles, tab-separated
``allele<TAB>34mer``). DTU's own distribution is unreachable from here.

Key formats differ and must be translated:
    ours                          theirs
    DRB1*03:01                 -> DRB1_0301
    DQA1*05:01-DQB1*02:01      -> HLA-DQA10501-DQB10201
    DPA1*01:03-DPB1*02:01      -> HLA-DPA10103-DPB10201

Two of our key shapes have NO counterpart and are deliberately NOT invented:

  * beta-only DP/DQ (``DPB1*04:01``, ``DQB1*03:01``, ...) -- the source table is
    keyed on alpha+beta pairs, and DPA1/DQA1 are polymorphic, so picking an
    alpha chain would fabricate the half of the binding groove the record never
    specified.
  * bare genes (``DRB3``, ``DRB4``, ``DRB5``) -- no allele given at all.

Those alleles are dropped instead. Re-run ``data_pipeline.py`` afterwards with
``require_real_pseudoseq: true`` so their rows leave the dataset rather than
being back-filled with placeholders.

  python build_real_pseudo.py --dat <downloaded.dat> [--dry-run]
"""

import argparse
import csv
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(
    HERE, "..", "..", "custom_dataset", "Anthem_dataset", "mhc_ii_pseudo.csv"))

PSEUDO_LEN = 34
# DRA is monomorphic, so every DRB entry shares the same alpha contribution.
# Used as an authenticity check on the downloaded table, per the source notes.
DRB_ALPHA_PREFIX = "QEFFIASGAAVDAIM"


def load_source(path):
    """allele -> 34-mer. Rejects the file if any row is malformed."""
    table = {}
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{ln}: expected 2 tab-separated fields")
            allele, seq = parts[0].strip(), parts[1].strip()
            if len(seq) != PSEUDO_LEN:
                raise ValueError(
                    f"{path}:{ln}: {allele} has {len(seq)} residues, expected {PSEUDO_LEN}")
            table[allele] = seq
    return table


def verify_source(table):
    """Sanity-check the download before we trust it with a retrain."""
    checks = {}
    drb1 = {k: v for k, v in table.items() if k.startswith("DRB1_")}
    checks["n_alleles"] = len(table)
    checks["n_DRB1"] = len(drb1)
    checks["all_34mer"] = all(len(v) == PSEUDO_LEN for v in table.values())
    checks["DRB1_alpha_invariant"] = (
        bool(drb1) and all(v.startswith(DRB_ALPHA_PREFIX) for v in drb1.values()))
    # DQ alpha is polymorphic -> the first 15 residues should NOT be constant
    dq = [v for k, v in table.items() if "DQA1" in k]
    checks["n_DQ"] = len(dq)
    checks["DQ_alpha_polymorphic"] = len({v[:15] for v in dq}) > 1
    return checks


def to_source_key(our_key):
    """Translate our ``allele_key`` to the source table's key, or None."""
    if re.fullmatch(r"DRB[0-9]+", our_key):
        return None                                   # bare gene, no allele
    parts = our_key.split("-")

    def chain(p):
        m = re.fullmatch(r"([A-Z0-9]+)\*(\d+):(\d+)", p)
        if not m:
            return None
        gene, fam, allele = m.groups()
        return gene, f"{fam}{allele}"

    chains = [chain(p) for p in parts]
    if any(c is None for c in chains):
        return None

    if len(chains) == 1:
        gene, digits = chains[0]
        if not gene.startswith("DRB"):
            return None                               # beta-only DP/DQ
        return f"{gene}_{digits}"

    (ga, da), (gb, db) = chains
    return f"HLA-{ga}{da}-{gb}{db}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat", required=True, help="downloaded pseudosequence .dat")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--train-txt",
                    default=os.path.normpath(os.path.join(
                        HERE, "..", "..", "custom_dataset", "Anthem_dataset", "train_data.txt")))
    ap.add_argument("--test-txt",
                    default=os.path.normpath(os.path.join(
                        HERE, "..", "..", "custom_dataset", "Anthem_dataset", "test_data.txt")))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = load_source(args.dat)
    checks = verify_source(src)
    print("source verification:")
    for k, v in checks.items():
        print(f"  {k:24s} {v}")
    if not (checks["all_34mer"] and checks["DRB1_alpha_invariant"]
            and checks["DQ_alpha_polymorphic"]):
        sys.exit("source table failed verification -- refusing to use it")

    # the allele keys our dataset actually needs
    keys = set()
    for p in (args.train_txt, args.test_txt):
        keys |= set(pd.read_csv(p, sep="\t")["HLA"].astype(str))
    keys = sorted(keys)

    rows, unmapped = [], []
    for k in keys:
        sk = to_source_key(k)
        seq = src.get(sk) if sk else None
        if seq is None:
            unmapped.append((k, sk))
        else:
            rows.append((k, seq, "real"))

    print(f"\nalleles needed: {len(keys)}")
    print(f"  mapped to real 34-mers: {len(rows)}")
    print(f"  unmapped (will be dropped): {len(unmapped)}")
    for k, sk in unmapped:
        why = "bare gene / beta-only -- no alpha+beta pair in source" if sk is None \
              else f"source key {sk} absent"
        print(f"    {k:26s} {why}")

    # how much data the drop costs
    tr = pd.read_csv(args.train_txt, sep="\t")
    te = pd.read_csv(args.test_txt, sep="\t")
    allr = pd.concat([tr, te])
    lost = allr.HLA.isin([k for k, _ in unmapped]).sum()
    print(f"\nrows affected: {lost}/{len(allr)} ({100 * lost / len(allr):.2f}%)")

    # a real table must not collapse distinct alleles onto one sequence by accident
    uniq = len({s for _, s, _ in rows})
    print(f"distinct 34-mers among mapped alleles: {uniq}/{len(rows)}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["allele_key", "pseudo_sequence", "source"])
        w.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} real rows, 0 placeholders)")
    print("next: re-run data_pipeline.py with require_real_pseudoseq=true, then retrain")


if __name__ == "__main__":
    main()
