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
CAVEAT for write-up: because of this, LOMO Phase2 and 5-fold Phase2 are not the
same model. Putting the two side by side in one table confounds the protocol
comparison with a lambda_reg change, so either footnote it explicitly or rerun
LOMO Phase2 at 0.1 (LAMBDA_REG=0.1 OUT_DIR=reports_lomo_l01) as a sensitivity
check.

Reporting rules baked in here (all three were silent failure modes before):
  * n_pos is kept in the per-allele CSV. Alleles with a single positive have
    AUROC = (n-k)/(n-1) and AP = 1/k for that positive's rank k -- a lottery
    result, not a measurement -- so lomo_summary.json averages ranking metrics
    only over alleles with >= summary_min_pos positives, and lists the excluded
    ones under "degenerate_alleles".
  * The F1 threshold is fitted on train-pool rows, not fixed at 0.5. pos_weight
    shifts cls_prob upward by an amount that differs per held-out allele.
  * The ensemble weight is fitted on train-pool rows too, separately for
    classification and regression, instead of a hardcoded 0.5.

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

from sklearn.metrics import roc_auc_score

from data_provider import DataProvider
from evaluate import pooled_metrics
from losses import Phase2Loss, Phase3Loss
from models.phase2 import build_from_config as build_phase2_from_config
from models.phase3 import build_from_config as build_phase3_from_config, warm_start_from_phase2
from train_phase2 import selection_score

HERE = os.path.dirname(os.path.abspath(__file__))
# Auto-detects a GPU (e.g. on Colab) and falls back to CPU otherwise; force with
# DEVICE=cpu / DEVICE=cuda if the auto-detect picks the wrong one.
DEVICE = torch.device(os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
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
    for key, env in [("lambda_reg", "LAMBDA_REG"), ("ensemble_alpha", "ENSEMBLE_ALPHA")]:
        if os.environ.get(env):
            cfg[key] = float(os.environ[env])
    # The two arms of the conditioning ablation, without editing any JSON:
    #   CORE_CONDITIONING=0 ... arm (1), the current architecture
    #   CORE_CONDITIONING=1 ... arm (2), register conditioned on the allele
    if os.environ.get("CORE_CONDITIONING") is not None:
        on = os.environ["CORE_CONDITIONING"].strip().lower() in ("1", "true", "yes")
        cfg["phase2_arch"]["core_conditioning"] = on
        cfg["phase3_arch"]["core_conditioning"] = on
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


def extra_metrics(df, threshold=None):
    """MAE, plus F1 at ``threshold``.

    ``threshold`` must be derived from the *training* pool (see
    ``best_f1_threshold``) -- never a fixed 0.5. The losses apply
    ``dp.pos_weight(tr_idx)``, which pushes sigmoid outputs systematically
    upward, so 0.5 is not the decision boundary; and because pos_weight is
    recomputed from a different train pool for every held-out allele, cls_prob
    is not even on a comparable scale across alleles. With no train-pool
    threshold available (e.g. resuming from pred TSVs written before val
    predictions were saved), F1 is reported as None instead of at a made-up
    cutoff.
    """
    out = {"mae": float(mean_absolute_error(df["affinity"], df["reg_yhat"]))}
    label = df["label"].to_numpy()
    if threshold is None or len(set(label)) < 2:
        out["f1"] = None
        return out
    pred = (df["cls_prob"].to_numpy() >= threshold).astype(int)
    out["f1"] = float(f1_score(label, pred, zero_division=0))
    return out


def best_f1_threshold(df, grid=201):
    """cls_prob cutoff maximising F1 on ``df``.

    Only ever called on train-pool rows (the val split carved out of the OTHER
    alleles); picking it on the held-out allele would be the same leak the
    module docstring refuses to replay.
    """
    if df is None:
        return None
    label = df["label"].to_numpy()
    if len(set(label)) < 2:
        return None
    prob = df["cls_prob"].to_numpy(dtype=float)
    best_t, best_f1 = None, -1.0
    for t in np.unique(np.quantile(prob, np.linspace(0.0, 1.0, grid))):
        f1 = f1_score(label, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def tune_alpha(p2_val, p3_val, grid=None):
    """Ensemble weights (Phase2's share), fitted on train-pool predictions.

    A hardcoded alpha=0.5 only beats both members when their errors are
    sufficiently uncorrelated; here Phase3's regression is clearly the stronger
    of the two (r ~0.48 vs ~0.39) and a 50:50 blend drags it down. Tuning on the
    held-out allele would be new leakage, so the sweep runs on the val split
    taken from the OTHER alleles' rows.

    Classification and regression get separate weights -- they do not rank the
    two models the same way. Returns ``(alpha_cls, alpha_reg)``, either of which
    is None when it could not be fitted.
    """
    if p2_val is None or p3_val is None or len(p2_val) != len(p3_val):
        return None, None
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 4) if grid is None else np.asarray(grid)

    label = p2_val["label"].to_numpy(dtype=float)
    a_cls = None
    if 0 < label.sum() < len(label):
        a_cls = float(max(grid, key=lambda a: roc_auc_score(
            label, a * p2_val["cls_prob"].to_numpy() + (1 - a) * p3_val["cls_prob"].to_numpy())))

    aff = p2_val["affinity"].to_numpy(dtype=float)
    a_reg = float(min(grid, key=lambda a: float(np.mean(
        (aff - (a * p2_val["reg_yhat"].to_numpy()
                + (1 - a) * p3_val["reg_yhat"].to_numpy())) ** 2))))
    return a_cls, a_reg


def blend(p2, p3, alpha_cls, alpha_reg):
    out = p2[["HLA", "label", "affinity"]].copy()
    out["cls_prob"] = alpha_cls * p2["cls_prob"].to_numpy() + (1 - alpha_cls) * p3["cls_prob"].to_numpy()
    out["reg_yhat"] = alpha_reg * p2["reg_yhat"].to_numpy() + (1 - alpha_reg) * p3["reg_yhat"].to_numpy()
    return out


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


def _select_score(cfg, model, criterion, dp, va_idx, batch_size, n_outputs):
    """Checkpoint-selection score for one epoch, and whether bigger is better.

    Mirrors train_phase2/3: selecting on val_loss picks the least-trained epoch,
    because the pos_weight-heavy BCE keeps rising while ranking improves. Both
    LOMO arms must use the SAME criterion as the 5-fold runs, or the LOMO
    numbers are not comparable to them.
    """
    if cfg.get("early_stop_metric", "auprc") == "val_loss":
        return _eval_loss(model, criterion, dp, va_idx, batch_size, n_outputs), False
    idx, prob, yhat = _predict(model, dp, va_idx, batch_size, n_outputs)
    vm = pooled_metrics(pred_frame(dp, idx, prob, yhat))
    score, bigger_is_better, _ = selection_score(cfg, vm, float("inf"))
    return score, bigger_is_better


def train_phase2_lomo(cfg, dp, tr_idx, va_idx, ckpt_path, seed):
    set_seed(seed)
    model = build_phase2_from_config(cfg["phase2_arch"]).to(DEVICE)
    pw = dp.pos_weight(tr_idx) if cfg.get("auto_pos_weight", True) else cfg["pos_weight"]
    criterion = Phase2Loss(pw, cfg["lambda_reg"], cfg["entropy_beta"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # -inf when a metric is maximised, +inf when val_loss is minimised
    best_val = (float("inf") if cfg.get("early_stop_metric", "auprc") == "val_loss"
                else -float("inf"))
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
        score, bigger_is_better = _select_score(
            cfg, model, criterion, dp, va_idx, cfg["batch_size"], 3)
        if (score > best_val) if bigger_is_better else (score < best_val):
            best_val = score
            torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


def train_phase3_lomo(cfg, dp, tr_idx, va_idx, ckpt_path, phase2_ckpt_path, seed):
    set_seed(seed)
    model = build_phase3_from_config(cfg["phase3_arch"]).to(DEVICE)

    # Phase3 in LOMO is *defined* as warm-started from this allele's own Phase2
    # checkpoint. A missing ckpt used to fall through to a cold start without a
    # word -- the dangerous shape being resume mode, where p2_pred.tsv survived
    # but the ckpt did not. Same trap as the old pseudo_encoder/mhc_encoder
    # naming mismatch that silently broke warm-start: fail loudly instead.
    core_only = cfg.get("warm_start_core_only", False)
    if not os.path.exists(phase2_ckpt_path):
        if not cfg.get("allow_cold_start", False):
            raise RuntimeError(
                f"Phase2 checkpoint missing: {phase2_ckpt_path}\n"
                "Phase3 would cold start and be reported as a warm-started model. "
                "Delete this allele's phase2_pred_*.tsv so Phase2 retrains (which "
                "rewrites the ckpt), or set \"allow_cold_start\": true in "
                "config/lomo.json to opt out deliberately."
            )
        print("  WARNING: no Phase2 ckpt -> Phase3 COLD START (allow_cold_start=true)")
    else:
        p2_state = torch.load(phase2_ckpt_path, map_location=DEVICE)
        loaded = warm_start_from_phase2(model, p2_state, core_only=core_only)
        if not loaded:
            raise RuntimeError(
                f"warm_start_from_phase2 matched 0 tensors from {phase2_ckpt_path} "
                "-- parameter names or shapes have drifted between phase2.json and "
                "phase3.json. Phase3 is silently cold starting; fix the configs."
            )
        print(f"  warm-start: {len(loaded)} tensors copied from Phase2 "
              f"(core_only={core_only})")

    pw = dp.pos_weight(tr_idx) if cfg.get("auto_pos_weight", True) else cfg["pos_weight"]
    criterion = Phase3Loss(pw, cfg["lambda_reg"], cfg["entropy_beta"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # -inf when a metric is maximised, +inf when val_loss is minimised
    best_val = (float("inf") if cfg.get("early_stop_metric", "auprc") == "val_loss"
                else -float("inf"))
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
        score, bigger_is_better = _select_score(
            cfg, model, criterion, dp, va_idx, cfg["batch_size"], 5)
        if (score > best_val) if bigger_is_better else (score < best_val):
            best_val = score
            torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


RANK_METRICS = ("auroc", "auprc", "f1")       # need >= min_pos positives to mean anything
REG_METRICS = ("pearson_r", "mse", "mae")     # independent of n_pos


def _wmean(df, col, weight_col="n_test"):
    """n-weighted mean -- a 62-row allele should not count as much as a 2086-row one."""
    s = df[[col, weight_col]].dropna()
    if s.empty or float(s[weight_col].sum()) <= 0:
        return None
    return float(np.average(s[col].to_numpy(dtype=float), weights=s[weight_col].to_numpy(dtype=float)))


def _stats(df, col):
    vals = df[col].dropna()
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()) if len(vals) else None,
        "std": float(vals.std()) if len(vals) > 1 else None,
        "weighted_mean": _wmean(df, col),
    }


def summarize(res_df, cfg):
    """Aggregate per-allele rows.

    Ranking metrics are averaged only over alleles with >= ``summary_min_pos``
    positives: with a single positive, AUROC/AUPRC are a closed-form function of
    that one peptide's rank, so including them reports luck as performance (and
    inflates the std enormously). Regression metrics do not depend on n_pos and
    are averaged over every allele. Both a macro mean and an n-weighted mean are
    emitted, plus ``*_mean_all_alleles`` for comparison with the unfiltered
    numbers.
    """
    min_pos = int(cfg.get("summary_min_pos", 2))
    if "n_pos" in res_df:
        scored = res_df[res_df["n_pos"] >= min_pos]
        degenerate = res_df[res_df["n_pos"] < min_pos]
    else:  # pre-fix CSV with no n_pos column -- cannot filter, say so
        scored, degenerate = res_df, res_df.iloc[0:0]

    summary = {
        "n_alleles": len(res_df),
        "summary_min_pos": min_pos,
        "n_alleles_rank_scored": len(scored),
        "n_pos_available": "n_pos" in res_df,
        "degenerate_alleles": (
            degenerate[["allele", "n_test", "n_pos"]].to_dict("records") if len(degenerate) else []
        ),
        "config": cfg,
    }
    for name in ("phase2", "phase3", "ensemble"):
        for metric in RANK_METRICS + REG_METRICS:
            col = f"{name}_{metric}"
            if col not in res_df:
                continue
            base = scored if metric in RANK_METRICS else res_df
            st = _stats(base, col)
            summary[f"{col}_mean"] = st["mean"]
            summary[f"{col}_std"] = st["std"]
            summary[f"{col}_weighted_mean"] = st["weighted_mean"]
            summary[f"{col}_n"] = st["n"]
            if metric in RANK_METRICS:
                summary[f"{col}_mean_all_alleles"] = _stats(res_df, col)["mean"]
    for col in ("alpha_cls", "alpha_reg"):
        if col in res_df:
            summary[f"{col}_mean"] = _stats(res_df, col)["mean"]
    return summary


def main():
    cfg = load_config()
    out_dir = _resolve(cfg["out_dir"])
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    pool_path = _build_pool()
    dp = DataProvider(pool_path, _resolve(cfg["pseudo_csv"]))
    print(f"LOMO pool: {len(dp)} rows, {len(set(dp.alleles))} alleles | device {DEVICE}")

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
          f"lambda_reg={cfg['lambda_reg']} | "
          f"core_conditioning={cfg['phase2_arch'].get('core_conditioning', False)}")
    if cfg["phase2_arch"].get("core_conditioning") != cfg["phase3_arch"].get("core_conditioning"):
        raise RuntimeError("core_conditioning differs between phase2.json and phase3.json -- "
                           "the ablation arms would be mixed within one run")

    results = []
    results_csv = os.path.join(out_dir, "lomo_per_allele.csv")
    if os.path.exists(results_csv):
        results = pd.read_csv(results_csv).to_dict("records")
        done_alleles = {r["allele"] for r in results}
        print(f"resuming: {len(done_alleles)} alleles already in {results_csv}")
        stale = [r["allele"] for r in results if r.get("n_pos") is None or pd.isna(r.get("n_pos"))]
        if stale:
            # Rows from before n_pos/alpha were recorded would be averaged as if
            # degenerate and mix two ensemble definitions in one table.
            print(f"WARNING: {len(stale)} resumed row(s) predate the n_pos/tuned-alpha "
                  f"columns: {stale[:5]}{' ...' if len(stale) > 5 else ''}\n"
                  f"         Rerun `python reaggregate_lomo.py` afterwards to rebuild "
                  f"every row from the pred TSVs on one consistent definition.")
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
        # Val-split (train-pool) predictions: the only rows allowed to pick the
        # F1 threshold and the ensemble alpha. Held-out rows must not be touched
        # by either choice.
        p2_val_path = os.path.join(out_dir, f"phase2_valpred_{safe}.tsv")
        p3_val_path = os.path.join(out_dir, f"phase3_valpred_{safe}.tsv")

        print(f"[{i+1}/{len(alleles)}] {allele} | train {len(tr_idx)} val {len(va_idx)} "
              f"test {len(test_idx)}")

        if os.path.exists(p2_pred_path):
            p2_pred = pd.read_csv(p2_pred_path, sep="\t")
        else:
            model2 = train_phase2_lomo(cfg, dp, tr_idx, va_idx, p2_ckpt, seed=cfg["seed"] + i)
            prob, yhat = _predict(model2, dp, test_idx, cfg["batch_size"], 3)
            p2_pred = pred_frame(dp, test_idx, prob, yhat)
            p2_pred.to_csv(p2_pred_path, sep="\t", index=False)
            vprob, vyhat = _predict(model2, dp, va_idx, cfg["batch_size"], 3)
            pred_frame(dp, va_idx, vprob, vyhat).to_csv(p2_val_path, sep="\t", index=False)

        if os.path.exists(p3_pred_path):
            p3_pred = pd.read_csv(p3_pred_path, sep="\t")
        else:
            # Phase3 gets its own seed stream: identical initialisation makes the
            # two members' errors correlate, which is exactly what kills the
            # ensemble's advantage over Phase3 alone.
            p3_seed = cfg["seed"] + i + cfg.get("phase3_seed_offset", 10000)
            model3 = train_phase3_lomo(cfg, dp, tr_idx, va_idx, p3_ckpt, p2_ckpt, seed=p3_seed)
            prob, yhat = _predict(model3, dp, test_idx, cfg["batch_size"], 5)
            p3_pred = pred_frame(dp, test_idx, prob, yhat)
            p3_pred.to_csv(p3_pred_path, sep="\t", index=False)
            vprob, vyhat = _predict(model3, dp, va_idx, cfg["batch_size"], 5)
            pred_frame(dp, va_idx, vprob, vyhat).to_csv(p3_val_path, sep="\t", index=False)

        p2_val = pd.read_csv(p2_val_path, sep="\t") if os.path.exists(p2_val_path) else None
        p3_val = pd.read_csv(p3_val_path, sep="\t") if os.path.exists(p3_val_path) else None

        a_cls, a_reg = tune_alpha(p2_val, p3_val)
        tuned = a_cls is not None and a_reg is not None
        if not tuned:
            a_cls = a_reg = cfg["ensemble_alpha"]
            print(f"  WARNING: no train-pool val preds -> ensemble uses fixed, UNTUNED "
                  f"alpha={a_cls} (delete this allele's pred TSVs to refit)")
        ens_pred = blend(p2_pred, p3_pred, a_cls, a_reg)
        ens_val = blend(p2_val, p3_val, a_cls, a_reg) if tuned else None

        n_pos = int(p2_pred["label"].sum())
        row = {"allele": allele, "n_test": len(test_idx), "n_pos": n_pos,
               "alpha_cls": a_cls, "alpha_reg": a_reg, "alpha_tuned": tuned}
        if n_pos < 2:
            # AP = 1/k and AUROC = (n-k)/(n-1) for a single positive at rank k:
            # ranking metrics here describe where one peptide happened to land,
            # not model quality. Kept in the CSV, excluded from the summary.
            print(f"  NOTE: n_pos={n_pos} -> ranking metrics are degenerate for this allele")
        for name, pred, val in [("phase2", p2_pred, p2_val),
                                ("phase3", p3_pred, p3_val),
                                ("ensemble", ens_pred, ens_val)]:
            thr = best_f1_threshold(val)
            m = pooled_metrics(pred)
            m.update(extra_metrics(pred, threshold=thr))
            row[f"{name}_threshold"] = thr
            for k, v in m.items():
                if k in ("n", "n_pos"):  # both already single top-level columns
                    continue
                row[f"{name}_{k}"] = v
        results.append(row)
        pd.DataFrame(results).to_csv(results_csv, index=False)

        print(f"[{i+1}/{len(alleles)}] {allele} done in {time.time()-t0:.0f}s | "
              f"P2 AUROC {row.get('phase2_auroc')} r {row.get('phase2_pearson_r')} | "
              f"P3 AUROC {row.get('phase3_auroc')} r {row.get('phase3_pearson_r')} | "
              f"Ens AUROC {row.get('ensemble_auroc')} r {row.get('ensemble_pearson_r')}")

    res_df = pd.DataFrame(results)
    summary = summarize(res_df, cfg)
    with open(os.path.join(out_dir, "lomo_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== LOMO summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nper-allele -> {results_csv}")


if __name__ == "__main__":
    main()
