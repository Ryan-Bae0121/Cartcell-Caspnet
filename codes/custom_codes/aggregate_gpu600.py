"""Aggregate the GPU 5-fold run (docs/project_status.md section 5.9).

Produces the one table section 5.9 promises: three models x three pre-registered
selection rules, on TEST, with per-fold values and a paired bootstrap CI.

Why this script exists rather than reusing the per-run summary JSONs: the
training scripts evaluate and pool only the PRIMARY (AUPRC) checkpoint. The two
secondary rules are saved as extra checkpoints (``*.pt.composite``,
``*.pt.rmax``) and never evaluated, so their test metrics have to be produced
here. Section 5.9 pre-registers reporting all three because the sign of the
headline regression gap depends on which rule picks the checkpoint.

Bootstrap design -- paired, matching section 5.3: one resample of TEST ROW
INDICES per replicate, reused for every model/rule, so a difference CI reflects
the model difference and not two independent resamples. Difference p-values are
two-sided from the bootstrap distribution, floored at 1/n_boot.

Per-fold rows use that fold's single model. The "ensemble" row averages
cls_prob / reg_yhat across the 5 folds, which is what section 5.2's test row
reported and what the bootstrap runs on.

  python aggregate_gpu600.py                       # all three models, 1000 reps
  python aggregate_gpu600.py --n-boot 200          # quick check
  python aggregate_gpu600.py --rules auprc         # primary only
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from data_provider import DataProvider
from evaluate import pooled_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device(os.environ.get("DEVICE") or
                      ("cuda" if torch.cuda.is_available() else "cpu"))

# model key -> (config file, checkpoint prefix, which build_from_config)
MODELS = {
    "phase2":     ("config/phase2_gpu600.json",    "phase2", "phase2"),
    "phase3":     ("config/phase3_gpu600.json",    "phase3", "phase3"),
    "norouting":  ("config/norouting_gpu600.json", "phase3", "phase3"),
}

# selection rule -> checkpoint suffix. "" is the primary checkpoint the training
# script wrote; the other two are the extra files train_one_fold saved.
RULES = {"auprc": "", "composite": ".composite", "rmax": ".rmax"}

METRICS = ["auroc", "auprc", "pearson_r", "mse"]

# Pairs the pre-registration cares about: the regression gap and the routing
# ablation. Each is (a, b) reported as a - b.
PAIRS = [("phase3", "phase2"), ("norouting", "phase3"), ("norouting", "phase2")]


def _resolve(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def _build(kind, cfg):
    if kind == "phase2":
        from models.phase2 import build_from_config
    else:
        from models.phase3 import build_from_config
    return build_from_config(cfg)


@torch.no_grad()
def predict_test(model, dp, batch_size):
    """Test predictions ordered by row index, for either architecture."""
    model.eval()
    idxs, probs, yhats = [], [], []
    for b in dp.iter_batches(np.arange(len(dp)), batch_size=batch_size):
        out = model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
        cls_logit, reg_yhat = out[0], out[1]
        probs.append(torch.sigmoid(cls_logit).cpu().numpy().ravel())
        yhats.append(reg_yhat.cpu().numpy().ravel())
        idxs.append(b["index"])
    i = np.concatenate(idxs)
    order = np.argsort(i)
    return np.concatenate(probs)[order], np.concatenate(yhats)[order]


def frame(dp, prob, yhat):
    return pd.DataFrame({"HLA": dp.alleles, "label": dp.labels,
                         "affinity": dp.affinities,
                         "cls_prob": prob, "reg_yhat": yhat})


def collect(model_key, rule, dp, n_folds, batch_size):
    """Per-fold test predictions for one model under one selection rule."""
    cfg_path, prefix, kind = MODELS[model_key]
    cfg = json.load(open(_resolve(cfg_path)))
    ckpt_dir = os.path.join(_resolve(cfg["out_dir"]), "checkpoints")
    suffix = RULES[rule]

    per_fold, missing, epochs = {}, [], {}
    for k in range(n_folds):
        path = os.path.join(ckpt_dir, f"{prefix}_fold{k}.pt{suffix}")
        if not os.path.exists(path):
            missing.append(path)
            continue
        model = _build(kind, cfg).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        per_fold[k] = predict_test(model, dp, batch_size)

        rep = os.path.join(_resolve(cfg["out_dir"]), f"fold_report_{k}.json")
        if os.path.exists(rep):
            r = json.load(open(rep))
            epochs[k] = (r.get("best_epoch") if rule == "auprc"
                         else r.get("alt_selection", {}).get(rule, {}).get("epoch"))
    return per_fold, missing, epochs


def ensemble(per_fold):
    probs = np.mean([p for p, _ in per_fold.values()], axis=0)
    yhats = np.mean([y for _, y in per_fold.values()], axis=0)
    return probs, yhats


def boot_metrics(df, rows):
    """Metrics on one bootstrap resample; None where undefined."""
    sub = df.iloc[rows]
    if not (0 < sub["label"].sum() < len(sub)):
        return {m: None for m in METRICS}
    m = pooled_metrics(sub)
    return {k: m.get(k) for k in METRICS}


def ci(vals, lo=2.5, hi=97.5):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return None, None, None
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)), int(v.size))


def two_sided_p(diffs, n_boot):
    d = np.asarray([x for x in diffs if x is not None], dtype=float)
    if d.size == 0:
        return None
    frac = min((d <= 0).mean(), (d >= 0).mean())
    return max(2.0 * frac, 1.0 / n_boot)


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--rules", nargs="*", default=list(RULES))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--out", default="reports_gpu600_aggregate.json")
    ap.add_argument("--md", default="reports_gpu600_aggregate.md")
    args = ap.parse_args()

    base_cfg = json.load(open(_resolve(MODELS["phase2"][0])))
    n_folds, bs = base_cfg["n_folds"], base_cfg["batch_size"]
    dp = DataProvider(_resolve(base_cfg["test_txt"]), _resolve(base_cfg["pseudo_csv"]))
    print(f"test rows {len(dp)} | n_folds {n_folds} | bs {bs} | device {DEVICE}")
    print(f"bootstrap {args.n_boot} reps, paired on identical test rows, seed {args.seed}")

    # ---------------------------------------------------------- predictions
    ens, fold_rows, miss_all, sel_epochs = {}, [], [], {}
    for mk in args.models:
        for rule in args.rules:
            per_fold, missing, epochs = collect(mk, rule, dp, n_folds, bs)
            miss_all += missing
            if not per_fold:
                print(f"  SKIP {mk}/{rule}: no checkpoints found")
                continue
            sel_epochs[(mk, rule)] = epochs
            for k, (p, y) in sorted(per_fold.items()):
                m = pooled_metrics(frame(dp, p, y))
                fold_rows.append({"model": mk, "rule": rule, "fold": k,
                                  "selected_epoch": epochs.get(k),
                                  **{x: m.get(x) for x in METRICS}})
            ens[(mk, rule)] = frame(dp, *ensemble(per_fold))
            print(f"  {mk:10s} {rule:9s} folds {sorted(per_fold)} "
                  f"| selected epochs {[epochs.get(k) for k in sorted(per_fold)]}")

    if miss_all:
        print(f"\n!! {len(miss_all)} checkpoint(s) missing -- those folds were skipped:")
        for p in miss_all[:10]:
            print("   ", p)

    if not ens:
        print("\nNothing to aggregate. Has the run finished writing checkpoints?")
        return

    # ------------------------------------------------------------ bootstrap
    rng = np.random.default_rng(args.seed)
    n = len(dp)
    point = {key: pooled_metrics(df) for key, df in ens.items()}
    draws = {key: {m: [] for m in METRICS} for key in ens}
    diffs = {(a, b, rule): {m: [] for m in METRICS}
             for a, b in PAIRS for rule in args.rules
             if (a, rule) in ens and (b, rule) in ens}

    for rep in range(args.n_boot):
        rows = rng.integers(0, n, size=n)          # one resample, shared by all
        vals = {key: boot_metrics(df, rows) for key, df in ens.items()}
        for key, v in vals.items():
            for m in METRICS:
                draws[key][m].append(v[m])
        for (a, b, rule) in diffs:
            va, vb = vals[(a, rule)], vals[(b, rule)]
            for m in METRICS:
                d = (None if va[m] is None or vb[m] is None else va[m] - vb[m])
                diffs[(a, b, rule)][m].append(d)
        if (rep + 1) % 200 == 0:
            print(f"  bootstrap {rep + 1}/{args.n_boot}")

    # -------------------------------------------------------------- assemble
    abs_rows = []
    for (mk, rule), m in point.items():
        row = {"model": mk, "rule": rule, "n": m["n"], "n_pos": m["n_pos"]}
        for x in METRICS:
            lo, hi, used = ci(draws[(mk, rule)][x])
            row[x] = m.get(x)
            row[f"{x}_ci"] = [lo, hi]
            row[f"{x}_boot_n"] = used
        abs_rows.append(row)

    diff_rows = []
    for (a, b, rule), per_m in diffs.items():
        row = {"a": a, "b": b, "rule": rule}
        for x in METRICS:
            pa, pb = point[(a, rule)].get(x), point[(b, rule)].get(x)
            lo, hi, _ = ci(per_m[x])
            row[x] = (None if pa is None or pb is None else pa - pb)
            row[f"{x}_ci"] = [lo, hi]
            row[f"{x}_p"] = two_sided_p(per_m[x], args.n_boot)
        diff_rows.append(row)

    # Pre-registered non-convergence rule: selected epoch >= 540 of 600.
    flags = []
    for (mk, rule), epochs in sel_epochs.items():
        late = {k: e for k, e in epochs.items() if e is not None and e >= 540}
        if late:
            flags.append({"model": mk, "rule": rule, "folds": late,
                          "note": "selected epoch >= 540/600 -> NON-CONVERGED, "
                                  "budget must be raised (section 5.9)"})

    result = {
        "provenance": {
            "device": str(DEVICE), "n_boot": args.n_boot, "seed": args.seed,
            "n_folds": n_folds, "batch_size": bs, "test_rows": len(dp),
            "run": "GPU 600-epoch 5-fold (section 5.9) -- do NOT mix with the "
                   "CPU runs in sections 5.2/5.3/5.8",
            "configs": {k: MODELS[k][0] for k in args.models},
            "missing_checkpoints": miss_all,
        },
        "absolute": abs_rows,
        "differences": diff_rows,
        "per_fold": fold_rows,
        "selected_epochs": {f"{mk}/{rule}": e for (mk, rule), e in sel_epochs.items()},
        "non_convergence_flags": flags,
    }
    out = _resolve(args.out)
    json.dump(result, open(out, "w"), indent=2)

    # ------------------------------------------------------------- markdown
    L = []
    L.append("### GPU 600-epoch 5-fold -- test metrics (section 5.9)\n")
    L.append(f"Paired bootstrap, {args.n_boot} resamples of the same "
             f"{len(dp)} test rows, seed {args.seed}. "
             f"**Not comparable to the CPU runs in 5.2/5.3/5.8.**\n")
    for rule in args.rules:
        rows = [r for r in abs_rows if r["rule"] == rule]
        if not rows:
            continue
        tag = {"auprc": "AUPRC (primary)", "composite": "composite (sensitivity)",
               "rmax": "r-max (reference)"}[rule]
        L.append(f"\n**Selection rule: {tag}**\n")
        L.append("| model | AUROC [95% CI] | AUPRC [95% CI] | Pearson r [95% CI] | MSE [95% CI] |")
        L.append("|---|---|---|---|---|")
        for r in rows:
            cells = [f"{fmt(r[x])} [{fmt(r[f'{x}_ci'][0])}, {fmt(r[f'{x}_ci'][1])}]"
                     for x in METRICS]
            L.append(f"| {r['model']} | " + " | ".join(cells) + " |")
        drs = [r for r in diff_rows if r["rule"] == rule]
        if drs:
            L.append("")
            L.append("| difference | Pearson r | AUPRC | AUROC |")
            L.append("|---|---|---|---|")
            for r in drs:
                cs = []
                for x in ["pearson_r", "auprc", "auroc"]:
                    cs.append(f"{fmt(r[x])} [{fmt(r[f'{x}_ci'][0])}, "
                              f"{fmt(r[f'{x}_ci'][1])}] p={fmt(r[f'{x}_p'], 3)}")
                L.append(f"| {r['a']} - {r['b']} | " + " | ".join(cs) + " |")

    L.append("\n**Per-fold (that fold's single model, on test)**\n")
    L.append("| model | rule | fold | sel epoch | AUROC | AUPRC | Pearson r | MSE |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in fold_rows:
        L.append(f"| {r['model']} | {r['rule']} | {r['fold']} | "
                 f"{r['selected_epoch']} | {fmt(r['auroc'])} | {fmt(r['auprc'])} | "
                 f"{fmt(r['pearson_r'])} | {fmt(r['mse'])} |")

    if flags:
        L.append("\n**Non-convergence flags (pre-registered: selected epoch >= 540/600)**\n")
        for f in flags:
            L.append(f"- `{f['model']}` / `{f['rule']}`: folds {f['folds']}")
    else:
        L.append("\nNo model hit the pre-registered non-convergence threshold "
                 "(selected epoch >= 540/600).")

    md = _resolve(args.md)
    open(md, "w", encoding="utf-8").write("\n".join(L) + "\n")

    print("\n" + "\n".join(L[:40]))
    print(f"\njson -> {out}\nmd   -> {md}")
    if flags:
        print("\n!! NON-CONVERGENCE FLAGS RAISED -- see the table above")


if __name__ == "__main__":
    main()
