"""Does the MHC branch contribute anything on a held-out allele? (no retraining)

Every row of ``mhc_ii_pseudo.csv`` is ``source=placeholder``: the 34-mer is a
sha256 of the allele key (``mhc_pseudo.placeholder_pseudo``), so the MHC input
carries allele *identity* but no pocket chemistry. Under LOMO the held-out
allele's fingerprint was never seen in training and bears no relation to any
training allele's, so there is nothing for the model to transfer -- the
prediction should be driven by the peptide alone.

Headline result (both alleles tested): the MHC branch is inert on a held-out
allele, AND the predicted core never moves -- the latter for an ARCHITECTURAL
reason, not the placeholders. ``common.CoreRegister`` scores candidate core
positions from the peptide feature map alone and ``mhc_vec`` is fused only after
the core is picked, so ``pos_prob`` cannot depend on the allele even with real
pseudo-sequences.

This script tests that claim directly on an existing LOMO checkpoint by
corrupting only the held-out allele's pseudo input and re-predicting:

  real     pseudo-sequence as stored                       (reference)
  zero     all-zero MHC input                              (branch silenced)
  permute  same 34 residues, positions shuffled            (composition kept)
  swap     a *different* allele's pseudo-sequence          (wrong allele, seen in training)
  unseen   a fresh sha256 fingerprint                      (what a new allele looks like)

Two readouts:
  * affinity/binding metrics per condition -- flat means the branch is inert
  * core-position agreement with ``real`` (argmax of pos_prob, 0-16) -- this is
    the register-shift question: if the predicted 9-mer core does not move when
    the MHC input is destroyed, register selection is peptide-intrinsic and no
    allele-dependent shift claim can be supported by this model.

``real`` is also checked against the saved ``phase{2,3}_pred_<allele>.tsv`` so a
mismatch in the harness shows up instead of being averaged in.

Usage:
  python ablate_pseudo.py                                  # DRB1*03:01, both phases
  python ablate_pseudo.py --allele "DRB1*11:01" --repeats 5
  python ablate_pseudo.py --reports-dir reports_lomo --out ablation.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from data_provider import DataProvider
from evaluate import pooled_metrics
from lomo import _build_pool, _resolve, _safe_name, load_config
from mhc_pseudo import placeholder_pseudo
from models.phase2 import build_from_config as build_phase2
from models.phase3 import build_from_config as build_phase3
from seq_encoding import simple_blosum10_encode

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device(os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))

STOCHASTIC = ("permute", "swap", "unseen")
CONDITIONS = ("real", "zero") + STOCHASTIC


def ablate(dp, allele, mode, rng):
    """The (10, 34) MHC tensor to feed for ``allele`` under ``mode``."""
    real = dp._encode_mhc(allele)
    if mode == "real":
        return real
    if mode == "zero":
        return torch.zeros_like(real)
    if mode == "permute":
        return real[:, torch.from_numpy(rng.permutation(real.shape[1]))]
    if mode == "swap":
        others = sorted(set(dp.alleles.tolist()) - {allele})
        return dp._encode_mhc(others[int(rng.integers(len(others)))])
    if mode == "unseen":
        seq = placeholder_pseudo(f"ABLATE*{int(rng.integers(10 ** 9)):09d}")
        return simple_blosum10_encode(seq, dp.pseudo_len)[0]
    raise ValueError(mode)


@torch.no_grad()
def predict(model, dp, indices, batch_size, allele, mhc_tensor):
    """Predict with ``allele``'s MHC input forced to ``mhc_tensor``.

    ``_encode_mhc`` is cache-first, so seeding the cache overrides the encoding
    without touching DataProvider. The entry is restored by the caller.
    """
    dp._mhc_cache[allele] = mhc_tensor
    model.eval()
    probs, yhats, cores = [], [], []
    for b in dp.iter_batches(indices, batch_size=batch_size):
        out = model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
        probs.append(torch.sigmoid(out[0]).cpu().numpy().ravel())
        yhats.append(out[1].cpu().numpy().ravel())
        cores.append(out[2].argmax(dim=1).cpu().numpy().ravel())
    return (np.concatenate(probs), np.concatenate(yhats), np.concatenate(cores))


def run_phase(name, model, dp, allele, test_idx, batch_size, repeats, saved_pred):
    frame = pd.DataFrame({
        "HLA": dp.alleles[test_idx],
        "label": dp.labels[test_idx],
        "affinity": dp.affinities[test_idx],
    })
    real_tensor = dp._encode_mhc(allele).clone()
    rows = []

    prob0, yhat0, core0 = predict(model, dp, test_idx, batch_size, allele, real_tensor)
    if saved_pred is not None:
        d = float(np.max(np.abs(saved_pred["cls_prob"].to_numpy() - prob0)))
        flag = "OK" if d < 1e-4 else f"MISMATCH (max |delta| = {d:.2e})"
        print(f"  {name}: reproduces saved pred TSV -> {flag}")

    for mode in CONDITIONS:
        n_rep = repeats if mode in STOCHASTIC else 1
        for rep in range(n_rep):
            rng = np.random.default_rng(1234 + rep)
            if mode == "real":
                prob, yhat, core = prob0, yhat0, core0
            else:
                prob, yhat, core = predict(model, dp, test_idx, batch_size, allele,
                                           ablate(dp, allele, mode, rng))
                dp._mhc_cache[allele] = real_tensor  # restore before next condition
            m = pooled_metrics(frame.assign(cls_prob=prob, reg_yhat=yhat))
            rows.append({
                "phase": name, "condition": mode, "rep": rep,
                "auroc": m["auroc"], "auprc": m["auprc"],
                "pearson_r": m["pearson_r"], "mse": m["mse"],
                "core_same_as_real": float(np.mean(core == core0)),
                "core_mean_abs_shift": float(np.mean(np.abs(core - core0))),
                "core_mode_pos": int(np.bincount(core, minlength=17).argmax()),
                "core_frac_at_mode": float(np.bincount(core, minlength=17).max() / len(core)),
            })
    dp._mhc_cache[allele] = real_tensor
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--allele", default="DRB1*03:01")
    ap.add_argument("--reports-dir", default=cfg["out_dir"])
    ap.add_argument("--repeats", type=int, default=5,
                    help="draws for the stochastic conditions (permute/swap/unseen)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reports = _resolve(args.reports_dir)
    ckpt_dir = os.path.join(reports, "checkpoints")
    safe = _safe_name(args.allele)
    dp = DataProvider(_build_pool(), _resolve(cfg["pseudo_csv"]))
    test_idx = np.where(dp.alleles == args.allele)[0]
    if len(test_idx) == 0:
        raise SystemExit(f"{args.allele} not in pool")

    n_pos = int(dp.labels[test_idx].sum())
    print(f"{args.allele}: {len(test_idx)} held-out rows, n_pos={n_pos} | device {DEVICE}")
    print(f"checkpoints: {ckpt_dir}")
    n_all = len(dp.pseudo)
    print(f"pseudo table: {n_all - dp.pseudo.n_placeholder} real / {n_all} alleles "
          f"({dp.pseudo.n_placeholder} placeholder)")

    out = []
    for name, builder, arch, n_out in [("phase2", build_phase2, cfg["phase2_arch"], 3),
                                       ("phase3", build_phase3, cfg["phase3_arch"], 5)]:
        ckpt = os.path.join(ckpt_dir, f"{name}_lomo_{safe}.pt")
        if not os.path.exists(ckpt):
            print(f"  {name}: no checkpoint at {ckpt}, skipping")
            continue
        model = builder(arch).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        pred_path = os.path.join(reports, f"{name}_pred_{safe}.tsv")
        saved = pd.read_csv(pred_path, sep="\t") if os.path.exists(pred_path) else None
        out.append(run_phase(name, model, dp, args.allele, test_idx,
                             cfg["batch_size"], args.repeats, saved))
    if not out:
        raise SystemExit("no checkpoints found -- point --reports-dir at the LOMO run")

    df = pd.concat(out, ignore_index=True)
    agg = (df.groupby(["phase", "condition"], sort=False)
             .agg(auroc=("auroc", "mean"), auprc=("auprc", "mean"),
                  pearson_r=("pearson_r", "mean"), mse=("mse", "mean"),
                  core_same=("core_same_as_real", "mean"),
                  core_shift=("core_mean_abs_shift", "mean"),
                  core_mode=("core_mode_pos", "median"),
                  at_mode=("core_frac_at_mode", "mean"),
                  reps=("rep", "count")))
    print(f"\n=== pseudo-sequence ablation: {args.allele} ===")
    print(agg.round(4).to_string())

    print("\n=== interpretation ===")
    for phase in agg.index.get_level_values(0).unique():
        a = agg.loc[phase]
        ref = a.loc["real"]
        worst = max(abs(a.loc[c, "auroc"] - ref["auroc"]) for c in CONDITIONS if c != "real")
        rworst = max(abs(a.loc[c, "pearson_r"] - ref["pearson_r"]) for c in CONDITIONS if c != "real")
        core_min = min(a.loc[c, "core_same"] for c in CONDITIONS if c != "real")
        print(f"{phase}: max AUROC change {worst:.4f} | max Pearson r change {rworst:.4f} | "
              f"core position unchanged in >= {core_min*100:.1f}% of peptides")
        verdict = ("MHC branch is effectively inert for this held-out allele"
                   if worst < 0.02 and rworst < 0.02 else
                   "MHC branch does move the predictions")
        print(f"        -> {verdict}")
        if core_min > 0.95:
            print("        -> register choice is peptide-only. Note the cause is ARCHITECTURAL, "
                  "not the placeholders:\n"
                  "           common.CoreRegister.forward(x) scores positions from the peptide "
                  "feature map alone\n"
                  "           and mhc_vec is fused only AFTER the core is selected, so pos_prob "
                  "cannot depend on\n"
                  "           the allele even with real pseudo-sequences. Conditioning "
                  "CoreRegister on mhc_vec is\n"
                  "           a prerequisite for any allele-dependent register-shift claim.")

    out_path = args.out or os.path.join(reports, f"pseudo_ablation_{safe}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nper-rep rows -> {out_path}")


if __name__ == "__main__":
    main()
