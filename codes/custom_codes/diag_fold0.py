"""Fold-0 diagnostic: why does training peak at epoch 0, and does allele
conditioning of the core register do anything now that pseudo-sequences are real?

Runs fold 0 twice -- core_conditioning off and on -- printing the per-epoch
curve for both. Three questions answered in one pass:

  1. Is the epoch-0 peak overfitting (train down, val up) or something else?
  2. Does conditioning the core register on the allele change held-out metrics?
  3. What value does ``cond_gate`` learn? It starts at exactly 0 (the two arms
     are identical at step 0), so a gate that stays near 0 means the model
     declined to let the pocket move the register even when allowed to.

  python -u diag_fold0.py [--epochs 15]
"""

import argparse
import json
import time
import warnings

import numpy as np
import torch
from sklearn.model_selection import KFold

from data_provider import DataProvider
from evaluate import pooled_metrics
from losses import Phase2Loss
from models.phase2 import build_from_config
from train_phase2 import (load_config, _resolve, set_seed, predict, pred_frame,
                          DEVICE)

warnings.filterwarnings("ignore")


def run_arm(cfg, dp, tr_idx, va_idx, conditioning, epochs):
    set_seed(cfg["seed"])
    arm_cfg = dict(cfg)
    arm_cfg["core_conditioning"] = conditioning
    model = build_from_config(arm_cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    lossf = Phase2Loss(pos_weight=torch.tensor(dp.pos_weight(tr_idx)).to(DEVICE),
                       lambda_reg=cfg["lambda_reg"],
                       entropy_beta=cfg["entropy_beta"])

    print(f"\n{'='*74}\ncore_conditioning = {conditioning}\n{'='*74}")
    print(f"{'ep':>3} {'train':>8} {'val':>8} {'AUROC':>7} {'AUPRC':>7} "
          f"{'r':>7} {'gate':>8} {'sec':>6}")

    history = []
    best = {"val_loss": float("inf"), "epoch": -1}
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tot = n = 0
        for b in dp.iter_batches(tr_idx, batch_size=cfg["batch_size"],
                                 shuffle=True, seed=cfg["seed"] + ep):
            opt.zero_grad()
            cl, ry, pp = model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
            out = lossf(cl, ry, pp, b["label"].to(DEVICE), b["affinity"].to(DEVICE))
            loss = out[0] if isinstance(out, tuple) else out
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tot += float(loss) * len(b["label"])
            n += len(b["label"])
        train_loss = tot / n

        # validation -- batch-mean val loss, matching train_phase2.py exactly so
        # this diagnoses the same quantity the real run early-stops on
        model.eval()
        vtot = vb = 0.0
        with torch.no_grad():
            for b in dp.iter_batches(va_idx, batch_size=cfg["batch_size"]):
                cl, ry, pp = model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
                out = lossf(cl, ry, pp, b["label"].to(DEVICE), b["affinity"].to(DEVICE))
                loss = out[0] if isinstance(out, tuple) else out
                vtot += float(loss)
                vb += 1
        val_loss = vtot / max(vb, 1)

        vi, vp, vy = predict(model, dp, va_idx, cfg["batch_size"])
        met = pooled_metrics(pred_frame(dp, vi, vp, vy))
        gate = (float(model.core.cond_gate.item())
                if model.core.cond_gate is not None else float("nan"))

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": ep, "metrics": met}
        history.append({"epoch": ep, "train_loss": train_loss,
                        "val_loss": val_loss, "gate": gate, **met})
        print(f"{ep:3d} {train_loss:8.4f} {val_loss:8.4f} {met['auroc']:7.4f} "
              f"{met['auprc']:7.4f} {met['pearson_r']:7.4f} {gate:8.4f} "
              f"{time.time()-t0:6.1f}")

    print(f"best epoch {best['epoch']} | val_loss {best['val_loss']:.4f} | "
          f"AUROC {best['metrics']['auroc']:.4f} r {best['metrics']['pearson_r']:.4f}")
    return {"conditioning": conditioning, "best": best, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="reports_diag_fold0.json")
    args = ap.parse_args()

    cfg = load_config()
    dp = DataProvider(_resolve(cfg["train_txt"]), _resolve(cfg["pseudo_csv"]))
    print(f"rows {len(dp)} | pos_weight {dp.pos_weight():.2f} | device {DEVICE} "
          f"| threads {torch.get_num_threads()}")

    kf = KFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    tr_idx, va_idx = next(iter(kf.split(np.arange(len(dp)))))
    print(f"fold 0: train {len(tr_idx)} val {len(va_idx)}")

    results = [run_arm(cfg, dp, tr_idx, va_idx, c, args.epochs)
               for c in (False, True)]

    with open(_resolve(args.out), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")

    off, on = results[0]["best"], results[1]["best"]
    print(f"\nconditioning OFF -> epoch {off['epoch']:2d} AUROC {off['metrics']['auroc']:.4f} "
          f"r {off['metrics']['pearson_r']:.4f}")
    print(f"conditioning ON  -> epoch {on['epoch']:2d} AUROC {on['metrics']['auroc']:.4f} "
          f"r {on['metrics']['pearson_r']:.4f}")
    print(f"final cond_gate  -> {results[1]['history'][-1]['gate']:.4f}  "
          "(0 = model declined to use the allele for register selection)")


if __name__ == "__main__":
    main()
