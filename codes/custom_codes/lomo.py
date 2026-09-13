"""LOMO (Leave-One-MHC-allele-Out) evaluation (briefing section 4, item 3).

For each eligible allele (>= min_test_samples rows in the pooled train+test
data), train Phase2 from scratch on every OTHER allele's data, then Phase3
(warm-started from that allele's own Phase2 checkpoint, full warm-start
core+MHC -- the briefing's preferred mode), then evaluate both plus their
alpha=0.5 ensemble on the held-out allele's rows only.

Deliberately NOT reproducing the known server bug (briefing 4.3): early
stopping here uses a random 90/10 subsplit of the *training* pool (every
OTHER allele's rows), never the held-out allele's own data. The old
`val_df = test_df` leak is documented as a limitation, not replayed.

lambda_reg = 0.5 for BOTH Phase2 and Phase3 in LOMO (briefing section 3: this
differs from Phase2's normal 5-fold lambda_reg=0.1 -- LOMO unifies both to 0.5).

Architecture params (pep_channels/core_kernel/pseudo_channels/cap_dim/...) are
read straight from config/phase2.json and config/phase3.json so they can never
drift out of sync with the main training runs (which would silently break
warm-start shape matching). config/lomo.json only holds LOMO-specific knobs.

Resumable per (allele, model): each allele+model's held-out predictions are
written to disk right after that allele finishes (reports_lomo/{phase2,phase3}
_pred_<allele>.tsv), and lomo_per_allele.csv is rewritten after every allele --
so an interrupted run only re-does the alleles that hadn't finished yet.

Usage:
  python lomo.py                                     # every eligible allele, full epochs
  ALLELE_LIMIT=5 python lomo.py                       # smoke test: 5 richest eligible alleles
  PHASE2_EPOCHS=10 PHASE3_EPOCHS=13 python lomo.py    # reduced-epoch run
  ALLELES="DRB1*07:01,DRB1*15:01" python lomo.py      # explicit allele subset
"""

import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, mean_absolute_error

from data_provider import DataProvider
from evaluate import pooled_metrics
from losses import Phase2Loss, Phase3Loss
from models.phase2 import build_from_config as build_phase2_from_config
from models.phase3 import build_from_config as build_phase3_from_config, warm_start_from_phase2

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")
DATASET_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "custom_dataset", "Anthem_dataset"))
POOL_PATH = os.path.join(DATASET_DIR, "lomo_pool.txt")


def _resolve(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def load_config():
    with open(os.path.join(HERE, "config", "phase2.json")) as fh:
        phase2_arch = json.load(fh)
    with open(os.path.join(HERE, "config", "phase3.json")) as fh:
        phase3_arch = json.load(fh)
    with open(os.path.join(HERE, "config", "lomo.json")) as fh:
        cfg = json.load(fh)
    cfg["phase2_arch"] = phase2_arch
    cfg["phase3_arch"] = phase3_arch
    cfg["pseudo_csv"] = phase2_arch["pseudo_csv"]

    for key, env in [("phase2_epochs", "PHASE2_EPOCHS"), ("phase3_epochs", "PHASE3_EPOCHS"),
                     ("batch_size", "BATCH_SIZE"), ("min_test_samples", "MIN_TEST_SAMPLES")]:
        if os.environ.get(env):
            cfg[key] = int(os.environ[env])
    if os.environ.get("OUT_DIR"):
        cfg["out_dir"] = os.environ["OUT_DIR"]
    if os.environ.get("ALLELE_LIMIT"):
        cfg["allele_limit"] = int(os.environ["ALLELE_LIMIT"])
    if os.environ.get("ALLELES"):
        cfg["alleles"] = [a.strip() for a in os.environ["ALLELES"].split(",") if a.strip()]
    return cfg


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_pool():
    """train_data.txt + test_data.txt pooled into one file -- LOMO cuts across
    the 80/20 CV split (that split was for 5-fold CV, not this protocol)."""
    if not os.path.exists(POOL_PATH):
        tr = pd.read_csv(os.path.join(DATASET_DIR, "train_data.txt"), sep="\t")
        te = pd.read_csv(os.path.join(DATASET_DIR, "test_data.txt"), sep="\t")
        pd.concat([tr, te], ignore_index=True).to_csv(POOL_PATH, sep="\t", index=False)
    return POOL_PATH


def pred_frame(dp, idx, prob, yhat):
    return pd.DataFrame({
        "HLA": dp.alleles[idx],
        "label": dp.labels[idx],
        "affinity": dp.affinities[idx],
        "cls_prob": prob,
        "reg_yhat": yhat,
    })


def extra_metrics(df, threshold=0.5):
    label = df["label"].to_numpy()
    pred = (df["cls_prob"].to_numpy() >= threshold).astype(int)
    return {
        "mae": float(mean_absolute_error(df["affinity"], df["reg_yhat"])),
        "f1": float(f1_score(label, pred, zero_division=0)) if len(set(label)) > 1 else None,
    }


@torch.no_grad()
def _predict(model, dp, indices, batch_size, n_outputs):
    model.eval()
    probs, yhats = [], []
    for b in dp.iter_batches(indices, batch_size=batch_size):
        out = model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
        cls_logit, reg_yhat = out[0], out[1]
        probs.append(torch.sigmoid(cls_logit).cpu().numpy().ravel())
        yhats.append(reg_yhat.cpu().numpy().ravel())
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(yhats)


@torch.no_grad()
def _eval_loss(model, criterion, dp, indices, batch_size, n_outputs):
    model.eval()
    total, n = 0.0, 0
    for b in dp.iter_batches(indices, batch_size=batch_size):
        out = model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
        cls_logit, reg_yhat, pos_prob = out[0], out[1], out[2]
        loss, _ = criterion(cls_logit, reg_yhat, pos_prob, b["label"].to(DEVICE), b["affinity"].to(DEVICE))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def train_phase2_lomo(cfg, dp, tr_idx, va_idx, ckpt_path, seed):
    set_seed(seed)
    model = build_phase2_from_config(cfg["phase2_arch"]).to(DEVICE)
    pw = dp.pos_weight(tr_idx) if cfg.get("auto_pos_weight", True) else cfg["pos_weight"]
    criterion = Phase2Loss(pw, cfg["lambda_reg"], cfg["entropy_beta"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val = float("inf")
    for epoch in range(cfg["phase2_epochs"]):
        model.train()
        for b in dp.iter_batches(tr_idx, batch_size=cfg["batch_size"], shuffle=True,
                                 drop_last=True, seed=seed * 100 + epoch):
            opt.zero_grad()
            cls_logit, reg_yhat, pos_prob = model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
            loss, _ = criterion(cls_logit, reg_yhat, pos_prob,
                                b["label"].to(DEVICE), b["affinity"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        val_loss = _eval_loss(model, criterion, dp, va_idx, cfg["batch_size"], 3)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


def train_phase3_lomo(cfg, dp, tr_idx, va_idx, ckpt_path, phase2_ckpt_path, seed):
    set_seed(seed)
    model = build_phase3_from_config(cfg["phase3_arch"]).to(DEVICE)
    if os.path.exists(phase2_ckpt_path):
        p2_state = torch.load(phase2_ckpt_path, map_location=DEVICE)
        warm_start_from_phase2(model, p2_state, core_only=cfg.get("warm_start_core_only", False))

    pw = dp.pos_weight(tr_idx) if cfg.get("auto_pos_weight", True) else cfg["pos_weight"]
    criterion = Phase3Loss(pw, cfg["lambda_reg"], cfg["entropy_beta"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val = float("inf")
    for epoch in range(cfg["phase3_epochs"]):
        model.train()
        for b in dp.iter_batches(tr_idx, batch_size=cfg["batch_size"], shuffle=True,
                                 drop_last=True, seed=seed * 100 + epoch):
            opt.zero_grad()
            cls_logit, reg_yhat, pos_prob, _, _ = model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
            loss, _ = criterion(cls_logit, reg_yhat, pos_prob,
                                b["label"].to(DEVICE), b["affinity"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        val_loss = _eval_loss(model, criterion, dp, va_idx, cfg["batch_size"], 5)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


def main():
    cfg = load_config()
    out_dir = _resolve(cfg["out_dir"])
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    pool_path = _build_pool()
    dp = DataProvider(pool_path, _resolve(cfg["pseudo_csv"]))
    print(f"LOMO pool: {len(dp)} rows, {len(set(dp.alleles))} alleles")

    vc = pd.Series(dp.alleles).value_counts()
    eligible = [a for a in vc.index if vc[a] >= cfg["min_test_samples"]]
    eligible.sort(key=lambda a: -vc[a])  # richest (most test samples) first

    if cfg.get("alleles"):
        alleles = [a for a in cfg["alleles"] if a in eligible]
        missing = set(cfg["alleles"]) - set(alleles)
        if missing:
            print(f"WARNING: not eligible/found, skipped: {sorted(missing)}")
    else:
        alleles = eligible
    if cfg.get("allele_limit"):
        alleles = alleles[: cfg["allele_limit"]]

    print(f"LOMO run: {len(alleles)} alleles (>= {cfg['min_test_samples']} samples) | "
          f"phase2_epochs={cfg['phase2_epochs']} phase3_epochs={cfg['phase3_epochs']} | "
          f"lambda_reg={cfg['lambda_reg']}")

    results = []
    results_csv = os.path.join(out_dir, "lomo_per_allele.csv")
    if os.path.exists(results_csv):
        results = pd.read_csv(results_csv).to_dict("records")
        done_alleles = {r["allele"] for r in results}
        print(f"resuming: {len(done_alleles)} alleles already in {results_csv}")
    else:
        done_alleles = set()

    for i, allele in enumerate(alleles):
        if allele in done_alleles:
            print(f"[{i+1}/{len(alleles)}] {allele}: already in results, skipping")
            continue

        t0 = time.time()
        mask = dp.alleles == allele
        test_idx = np.where(mask)[0]
        other_idx = np.where(~mask)[0]
        rng = np.random.default_rng(cfg["seed"] + i)
        perm = rng.permutation(other_idx)
        n_val = max(1, int(len(perm) * cfg["val_frac"]))
        va_idx, tr_idx = perm[:n_val], perm[n_val:]

        safe = _safe_name(allele)
        p2_ckpt = os.path.join(ckpt_dir, f"phase2_lomo_{safe}.pt")
        p3_ckpt = os.path.join(ckpt_dir, f"phase3_lomo_{safe}.pt")
        p2_pred_path = os.path.join(out_dir, f"phase2_pred_{safe}.tsv")
        p3_pred_path = os.path.join(out_dir, f"phase3_pred_{safe}.tsv")

        print(f"[{i+1}/{len(alleles)}] {allele} | train {len(tr_idx)} val {len(va_idx)} "
              f"test {len(test_idx)}")

        if os.path.exists(p2_pred_path):
            p2_pred = pd.read_csv(p2_pred_path, sep="\t")
        else:
            model2 = train_phase2_lomo(cfg, dp, tr_idx, va_idx, p2_ckpt, seed=cfg["seed"] + i)
            prob, yhat = _predict(model2, dp, test_idx, cfg["batch_size"], 3)
            p2_pred = pred_frame(dp, test_idx, prob, yhat)
            p2_pred.to_csv(p2_pred_path, sep="\t", index=False)

        if os.path.exists(p3_pred_path):
            p3_pred = pd.read_csv(p3_pred_path, sep="\t")
        else:
            model3 = train_phase3_lomo(cfg, dp, tr_idx, va_idx, p3_ckpt, p2_ckpt, seed=cfg["seed"] + i)
            prob, yhat = _predict(model3, dp, test_idx, cfg["batch_size"], 5)
            p3_pred = pred_frame(dp, test_idx, prob, yhat)
            p3_pred.to_csv(p3_pred_path, sep="\t", index=False)

        alpha = cfg["ensemble_alpha"]
        ens_pred = p2_pred[["HLA", "label", "affinity"]].copy()
        ens_pred["cls_prob"] = alpha * p2_pred["cls_prob"] + (1 - alpha) * p3_pred["cls_prob"]
        ens_pred["reg_yhat"] = alpha * p2_pred["reg_yhat"] + (1 - alpha) * p3_pred["reg_yhat"]

        row = {"allele": allele, "n_test": len(test_idx)}
        for name, pred in [("phase2", p2_pred), ("phase3", p3_pred), ("ensemble", ens_pred)]:
            m = pooled_metrics(pred)
            m.update(extra_metrics(pred))
            for k, v in m.items():
                if k in ("n", "n_pos"):
                    continue
                row[f"{name}_{k}"] = v
        results.append(row)
        pd.DataFrame(results).to_csv(results_csv, index=False)

        print(f"[{i+1}/{len(alleles)}] {allele} done in {time.time()-t0:.0f}s | "
              f"P2 AUROC {row.get('phase2_auroc')} r {row.get('phase2_pearson_r')} | "
              f"P3 AUROC {row.get('phase3_auroc')} r {row.get('phase3_pearson_r')} | "
              f"Ens AUROC {row.get('ensemble_auroc')} r {row.get('ensemble_pearson_r')}")

    res_df = pd.DataFrame(results)
    summary = {"n_alleles": len(res_df), "config": cfg}
    for name in ("phase2", "phase3", "ensemble"):
        for metric in ("auroc", "auprc", "pearson_r", "mse", "mae", "f1"):
            col = f"{name}_{metric}"
            if col in res_df:
                summary[f"{col}_mean"] = float(res_df[col].mean(skipna=True))
                summary[f"{col}_std"] = float(res_df[col].std(skipna=True))
    with open(os.path.join(out_dir, "lomo_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== LOMO summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nper-allele -> {results_csv}")


if __name__ == "__main__":
    main()
