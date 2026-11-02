"""Re-aggregate a finished LOMO run from its saved prediction TSVs -- no retraining.

Covers the three fixes that are pure post-processing:
  1. n_pos is reported, and ranking metrics are summarised only over alleles
     with >= --min-pos positives (degenerate ones are listed, not averaged).
  2. F1 comes from a train-pool threshold instead of a fixed 0.5.
  3. The ensemble weight is fitted per allele instead of hardcoded to 0.5.

(2) and (3) need the val-split predictions (``phase{2,3}_valpred_<allele>.tsv``),
which only runs after this change writes. For an older run that has just the
held-out ``phase{2,3}_pred_*.tsv``, this still fixes (1) and recomputes
everything else -- F1 is then reported as None and the ensemble falls back to
--alpha, both flagged per allele in the CSV (``alpha_tuned``) and in the summary.

The --sweep output is a DIAGNOSTIC: it scores alpha on the held-out rows, which
is exactly the leak the per-allele alpha avoids. Use it to see how much is on
the table, never to pick the number you report.

Usage:
  python reaggregate_lomo.py                                  # reports_lomo, min_pos=2
  python reaggregate_lomo.py --out-dir reports_lomo --min-pos 5
  python reaggregate_lomo.py --sweep
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from evaluate import pooled_metrics
from lomo import best_f1_threshold, blend, extra_metrics, load_config, summarize, tune_alpha

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    return pd.read_csv(path, sep="\t") if os.path.exists(path) else None


def collect(out_dir, alpha_fallback):
    rows = []
    for p2_path in sorted(glob.glob(os.path.join(out_dir, "phase2_pred_*.tsv"))):
        safe = os.path.basename(p2_path)[len("phase2_pred_"):-len(".tsv")]
        p3_path = os.path.join(out_dir, f"phase3_pred_{safe}.tsv")
        if not os.path.exists(p3_path):
            print(f"skip {safe}: no phase3 predictions")
            continue
        p2, p3 = _read(p2_path), _read(p3_path)
        if len(p2) != len(p3):
            print(f"skip {safe}: phase2/phase3 row counts differ ({len(p2)} vs {len(p3)})")
            continue
        p2_val = _read(os.path.join(out_dir, f"phase2_valpred_{safe}.tsv"))
        p3_val = _read(os.path.join(out_dir, f"phase3_valpred_{safe}.tsv"))

        a_cls, a_reg = tune_alpha(p2_val, p3_val)
        tuned = a_cls is not None and a_reg is not None
        if not tuned:
            a_cls = a_reg = alpha_fallback
        ens = blend(p2, p3, a_cls, a_reg)
        ens_val = blend(p2_val, p3_val, a_cls, a_reg) if tuned else None

        allele = str(p2["HLA"].iloc[0])
        row = {"allele": allele, "n_test": len(p2), "n_pos": int(p2["label"].sum()),
               "alpha_cls": a_cls, "alpha_reg": a_reg, "alpha_tuned": tuned}
        for name, pred, val in [("phase2", p2, p2_val), ("phase3", p3, p3_val),
                                ("ensemble", ens, ens_val)]:
            thr = best_f1_threshold(val)
            m = pooled_metrics(pred)
            m.update(extra_metrics(pred, threshold=thr))
            row[f"{name}_threshold"] = thr
            for k, v in m.items():
                if k in ("n", "n_pos"):
                    continue
                row[f"{name}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def sweep(out_dir, grid):
    """Held-out-scored alpha sweep. Diagnostic only -- see the module docstring."""
    recs = []
    for p2_path in sorted(glob.glob(os.path.join(out_dir, "phase2_pred_*.tsv"))):
        safe = os.path.basename(p2_path)[len("phase2_pred_"):-len(".tsv")]
        p3_path = os.path.join(out_dir, f"phase3_pred_{safe}.tsv")
        if not os.path.exists(p3_path):
            continue
        p2, p3 = _read(p2_path), _read(p3_path)
        if len(p2) != len(p3):
            continue
        for a in grid:
            m = pooled_metrics(blend(p2, p3, a, a))
            recs.append({"allele": str(p2["HLA"].iloc[0]), "n_pos": int(p2["label"].sum()),
                         "alpha": float(a), "auroc": m["auroc"], "auprc": m["auprc"],
                         "pearson_r": m["pearson_r"], "mse": m["mse"]})
    return pd.DataFrame(recs)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=cfg["out_dir"])
    ap.add_argument("--min-pos", type=int, default=cfg.get("summary_min_pos", 2),
                    help="minimum positives for an allele to enter the ranking-metric means")
    ap.add_argument("--alpha", type=float, default=cfg["ensemble_alpha"],
                    help="ensemble weight used only where val predictions are missing")
    ap.add_argument("--sweep", action="store_true", help="also write the diagnostic alpha sweep")
    ap.add_argument("--suffix", default="_reagg", help="suffix for the output filenames")
    args = ap.parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    res_df = collect(out_dir, args.alpha)
    if res_df.empty:
        raise SystemExit(f"no usable prediction TSV pairs in {out_dir}")

    cfg = dict(cfg, summary_min_pos=args.min_pos, ensemble_alpha=args.alpha)
    summary = summarize(res_df, cfg)

    csv_path = os.path.join(out_dir, f"lomo_per_allele{args.suffix}.csv")
    json_path = os.path.join(out_dir, f"lomo_summary{args.suffix}.json")
    res_df.sort_values("n_test", ascending=False).to_csv(csv_path, index=False)
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    cols = ["allele", "n_test", "n_pos", "phase2_auroc", "phase3_auroc", "ensemble_auroc",
            "phase2_pearson_r", "phase3_pearson_r", "ensemble_pearson_r", "alpha_tuned"]
    print(res_df[[c for c in cols if c in res_df]]
          .sort_values("n_pos").to_string(index=False))
    n_deg = len(summary["degenerate_alleles"])
    print(f"\n{n_deg} allele(s) below n_pos={args.min_pos}, excluded from ranking means:")
    for d in summary["degenerate_alleles"]:
        print(f"  {d['allele']}  n_test={d['n_test']}  n_pos={d['n_pos']}")
    if not res_df["alpha_tuned"].all():
        print(f"\nWARNING: {int((~res_df['alpha_tuned']).sum())} allele(s) had no val "
              f"predictions -> fixed alpha={args.alpha}, F1 = None. Rerun those alleles "
              "to fit alpha and the F1 threshold properly.")
    print("\n=== summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    print(f"\n-> {csv_path}\n-> {json_path}")

    if args.sweep:
        sw = sweep(out_dir, np.round(np.arange(0.0, 1.0001, 0.05), 4))
        rank = sw[sw["n_pos"] >= args.min_pos].groupby("alpha")[["auroc", "auprc"]].mean()
        reg = sw.groupby("alpha")[["pearson_r", "mse"]].mean()
        agg = rank.join(reg)
        sweep_path = os.path.join(out_dir, f"lomo_alpha_sweep{args.suffix}.csv")
        sw.to_csv(sweep_path, index=False)
        print(f"\n=== alpha sweep (DIAGNOSTIC, scored on held-out rows; "
              f"n_pos >= {args.min_pos} for auroc/auprc) ===")
        print(agg.to_string())
        print(f"-> {sweep_path}")


if __name__ == "__main__":
    main()
