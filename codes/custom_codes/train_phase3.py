"""Phase 3 training: 5-fold CV on train_data.txt, OOF metrics, test prediction.

Multi-branch CapsNet + Dynamic Routing (briefing section 2, Phase 3), warm-started
per-fold from the matching Phase 2 checkpoint (same KFold seed/n_folds as
train_phase2.py -> fold k's train/val split is identical across the two scripts,
so ``phase2_fold{k}.pt`` is a coherent warm start for Phase 3 fold k).

  out_dir  = reports_phase3_capsnet/
  summary  = phase3_capsnet_summary.json
  checkpoints/phase3_fold{k}.pt
  batch 32, lr 2e-4, 5 folds x 40 epochs, lambda_reg 0.5, auto pos_weight.

Quick smoke test (no warm-start needed -- falls back to scratch with a warning
if reports_phase2_affinity_from_scratch/checkpoints/ is empty):
  EPOCHS=2 N_FOLDS=2 python train_phase3.py

Disable warm-start explicitly, or point at a different Phase2 run:
  WARM_START=0 python train_phase3.py
  PHASE2_CKPT_DIR=/path/to/checkpoints python train_phase3.py
  WARM_START_CORE_ONLY=1 python train_phase3.py   # reproduce the old (buggy) server behaviour
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from data_provider import DataProvider
from evaluate import evaluate, pooled_metrics
from losses import Phase3Loss
from models.phase3 import build_from_config, warm_start_from_phase2
from resume_utils import fold_paths, fold_is_done, load_fold_result, save_fold_result

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "")


def load_config(path=None):
    path = path or os.path.join(HERE, "config", "phase3.json")
    with open(path) as fh:
        cfg = json.load(fh)
    # env overrides for smoke tests / ad-hoc runs
    for key, env in [("epochs", "EPOCHS"), ("n_folds", "N_FOLDS"),
                     ("batch_size", "BATCH_SIZE")]:
        if os.environ.get(env):
            cfg[key] = int(os.environ[env])
    if os.environ.get("OUT_DIR"):
        cfg["out_dir"] = os.environ["OUT_DIR"]
    cfg["warm_start"] = _env_bool("WARM_START", cfg.get("warm_start", True))
    cfg["warm_start_core_only"] = _env_bool("WARM_START_CORE_ONLY",
                                             cfg.get("warm_start_core_only", False))
    if os.environ.get("PHASE2_CKPT_DIR"):
        cfg["phase2_ckpt_dir"] = os.environ["PHASE2_CKPT_DIR"]
    return cfg


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, dp, indices, batch_size):
    model.eval()
    probs, yhats, idxs = [], [], []
    for b in dp.iter_batches(indices, batch_size=batch_size):
        cls_logit, reg_yhat, _, _, _ = model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
        probs.append(torch.sigmoid(cls_logit).cpu().numpy().ravel())
        yhats.append(reg_yhat.cpu().numpy().ravel())
        idxs.append(b["index"])
    return np.concatenate(idxs), np.concatenate(probs), np.concatenate(yhats)


def pred_frame(dp, idx, prob, yhat):
    return pd.DataFrame({
        "HLA": dp.alleles[idx],
        "label": dp.labels[idx],
        "affinity": dp.affinities[idx],
        "cls_prob": prob,
        "reg_yhat": yhat,
    })


def _warm_start(model, cfg, fold):
    """Load the fold-matched Phase2 checkpoint into model in place. Falls back
    to training from scratch (with a printed note) if warm_start is off or the
    checkpoint doesn't exist yet -- e.g. Phase 2 hasn't been trained locally."""
    if not cfg.get("warm_start", True):
        return "scratch (warm_start disabled)"
    ckpt_dir = _resolve(cfg["phase2_ckpt_dir"])
    p2_path = os.path.join(ckpt_dir, f"phase2_fold{fold}.pt")
    if not os.path.exists(p2_path):
        return f"scratch (no Phase2 checkpoint at {p2_path})"
    p2_state = torch.load(p2_path, map_location=DEVICE)
    core_only = cfg.get("warm_start_core_only", False)
    loaded = warm_start_from_phase2(model, p2_state, core_only=core_only)
    mode = "core-only" if core_only else "core+MHC"
    return f"warm-start [{mode}, {len(loaded)} tensors] <- {p2_path}"


def train_one_fold(cfg, dp, tr_idx, va_idx, fold, ckpt_path):
    set_seed(cfg["seed"] + fold)
    model = build_from_config(cfg).to(DEVICE)
    print(f"  fold {fold} init: {_warm_start(model, cfg, fold)}")

    pw = dp.pos_weight(tr_idx) if cfg.get("auto_pos_weight", True) else cfg["pos_weight"]
    criterion = Phase3Loss(pw, cfg["lambda_reg"], cfg["entropy_beta"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])

    best = {"val_loss": float("inf"), "epoch": -1, "metrics": None}
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        n_batches = 0
        for b in dp.iter_batches(tr_idx, batch_size=cfg["batch_size"],
                                 shuffle=True, drop_last=True,
                                 seed=cfg["seed"] + fold * 100 + epoch):
            opt.zero_grad()
            cls_logit, reg_yhat, pos_prob, _, _ = model(b["peptide"].to(DEVICE),
                                                         b["pseudo"].to(DEVICE))
            loss, _ = criterion(cls_logit, reg_yhat, pos_prob,
                                b["label"].to(DEVICE), b["affinity"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            run_loss += loss.item()
            n_batches += 1

        # validation
        vi, vp, vy = predict(model, dp, va_idx, cfg["batch_size"])
        vdf = pred_frame(dp, vi, vp, vy)
        vm = pooled_metrics(vdf)
        with torch.no_grad():
            vt = torch.tensor(0.0)
            vb = 0
            for b in dp.iter_batches(va_idx, batch_size=cfg["batch_size"]):
                cl, ry, pp, _, _ = model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
                l, _ = criterion(cl, ry, pp, b["label"].to(DEVICE), b["affinity"].to(DEVICE))
                vt += l
                vb += 1
            val_loss = (vt / max(vb, 1)).item()

        print(f"  fold {fold} epoch {epoch:02d} | "
              f"train_loss {run_loss/max(n_batches,1):.4f} | val_loss {val_loss:.4f} | "
              f"AUROC {vm['auroc']} r {vm['pearson_r']} | {time.time()-t0:.1f}s")

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "metrics": vm}
            torch.save(model.state_dict(), ckpt_path)

    # reload best for OOF / test
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    oi, op, oy = predict(model, dp, va_idx, cfg["batch_size"])
    return model, best, pred_frame(dp, oi, op, oy)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    out_dir = _resolve(cfg["out_dir"])
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dp = DataProvider(_resolve(cfg["train_txt"]), _resolve(cfg["pseudo_csv"]))
    test_dp = DataProvider(_resolve(cfg["test_txt"]), _resolve(cfg["pseudo_csv"]))
    print(f"train rows {len(train_dp)} | test rows {len(test_dp)} | "
          f"pos_weight(all) {train_dp.pos_weight():.2f} | device {DEVICE} | "
          f"warm_start {cfg['warm_start']} (core_only={cfg['warm_start_core_only']})")

    # Same seed/n_folds as train_phase2.py -> identical fold splits, so
    # phase2_fold{k}.pt is a valid warm start for phase3 fold k.
    kf = KFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    all_idx = np.arange(len(train_dp))

    fold_reports = []
    oof_frames = []
    test_prob_acc = np.zeros(len(test_dp))
    test_yhat_acc = np.zeros(len(test_dp))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(all_idx)):
        paths = fold_paths(out_dir, ckpt_dir, "phase3", fold)
        if fold_is_done(paths):
            print(f"\n=== fold {fold} already complete -- reusing saved result "
                  f"({paths['oof']}) ===")
            report, oof, test_prob, test_yhat = load_fold_result(paths)
            fold_reports.append(report)
            oof_frames.append(oof)
            test_prob_acc += test_prob
            test_yhat_acc += test_yhat
            continue

        print(f"\n=== fold {fold} | train {len(tr_idx)} val {len(va_idx)} ===")
        model, best, oof = train_one_fold(cfg, train_dp, tr_idx, va_idx, fold, paths["ckpt"])
        report = {"fold": fold, "best_epoch": best["epoch"],
                  "best_val_loss": best["val_loss"], "val_metrics": best["metrics"]}
        fold_reports.append(report)
        oof_frames.append(oof)

        ti, tp, ty = predict(model, test_dp, np.arange(len(test_dp)), cfg["batch_size"])
        order = np.argsort(ti)
        test_prob, test_yhat = tp[order], ty[order]
        test_prob_acc += test_prob
        test_yhat_acc += test_yhat
        save_fold_result(paths, report, oof, test_prob, test_yhat)

    # pooled OOF metrics
    oof_df = pd.concat(oof_frames, ignore_index=True)
    oof_df.to_csv(os.path.join(out_dir, "oof_predictions.tsv"), sep="\t", index=False)
    oof_eval = evaluate(oof_df)

    # ensemble-of-folds test metrics
    k = cfg["n_folds"]
    test_df = pd.DataFrame({
        "HLA": test_dp.alleles,
        "label": test_dp.labels,
        "affinity": test_dp.affinities,
        "cls_prob": test_prob_acc / k,
        "reg_yhat": test_yhat_acc / k,
    })
    test_df.to_csv(os.path.join(out_dir, "test_predictions.tsv"), sep="\t", index=False)
    test_eval = evaluate(test_df)

    summary = {
        "config": cfg,
        "device": str(DEVICE),
        "n_train": len(train_dp),
        "n_test": len(test_dp),
        "folds": fold_reports,
        "oof": oof_eval,
        "test": test_eval,
    }
    spath = os.path.join(out_dir, "phase3_capsnet_summary.json")
    with open(spath, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as fh:
        json.dump(test_eval, fh, indent=2)

    print("\n=== OOF pooled ===")
    print(json.dumps(oof_eval["pooled"], indent=2))
    print("=== TEST pooled ===")
    print(json.dumps(test_eval["pooled"], indent=2))
    print(f"\nsummary -> {spath}")


if __name__ == "__main__":
    main()
