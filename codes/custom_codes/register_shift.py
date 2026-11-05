"""Register-shift robustness experiment for Phase 2 (core-definition-free).

Question
--------
Does the model's prediction depend on WHERE a peptide sits, rather than on what
it contains? We never need a ground-truth 9-mer core: the whole peptide is
moved, so whatever core it holds moves with it.

Why controls are mandatory
--------------------------
Shift-invariance alone proves nothing -- a model that ignores position entirely
scores perfectly on it, and so does a bag-of-amino-acids model. The claim
"the model recognises a core" only stands if invariance is paired with
sensitivity. Hence four conditions per anchor case:

  shift_pad      peptide placed at offset k in the 25-wide tensor, zero padding
                 elsewhere. Pure architecture probe. NOTE: training only ever
                 saw offset 0, so k>0 is out-of-distribution by construction.
  shift_flank    total length held at T, peptide placed with j random real
                 residues on the left and T-L-j on the right, tensor offset 0.
                 In-distribution. This is the biologically meaningful condition.
  flank_resample flanks resampled R times at a FIXED j. The proper null for
                 shift_flank: it measures how much prediction moves when only
                 flank identity changes, with position held constant.
  core_mutate    central 9-mer replaced with random residues -> expect LARGE
                 change (sensitivity to content).
  scramble_all   whole peptide shuffled, composition preserved, order destroyed
                 -> expect LARGE change. This is what rules out a model that
                 only reads amino-acid composition.

Read the output as: shift_flank spread should be SMALL and comparable to
flank_resample, while core_mutate / scramble_all shifts should be LARGE.
If shift_flank spread exceeds flank_resample spread, position is leaking.

Usage
-----
  python register_shift.py --ckpt-dir reports_phase2_shiftexp/checkpoints \
                           --out-dir reports_register_shift
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from data_provider import DataProvider
from models.phase2 import build_from_config
from models.phase3 import build_from_config as build_phase3
from mhc_pseudo import PseudoSeqTable
from seq_encoding import (
    BLOSUM62, STANDARD_AA, PEP_DIM, PEP_MAX_LEN, PSEUDO_LEN,
    simple_blosum10_encode,
)
from train_phase2 import load_config, _resolve

DEVICE = torch.device(os.environ.get("DEVICE") or
                      ("cuda" if torch.cuda.is_available() else "cpu"))

# External, literature-standard IC50 cutoffs -- deliberately NOT tuned on this
# dataset. affinity = 1 - log(IC50)/log(50000).
LOG_BASE = 50000.0


def ic50_to_affinity(ic50):
    return 1.0 - np.log(ic50) / np.log(LOG_BASE)


THRESHOLDS = {
    "500nM": ic50_to_affinity(500.0),   # conventional "binder"
    "50nM": ic50_to_affinity(50.0),     # conventional "strong binder"
}


# ---------------------------------------------------------------------------
# encoding with an explicit offset (seq_encoding.blosum_encode is left-aligned)
# ---------------------------------------------------------------------------
def encode_pep_at(seq, offset=0, max_len=PEP_MAX_LEN):
    """(23, max_len) BLOSUM tensor with ``seq`` starting at column ``offset``."""
    t = torch.zeros((PEP_DIM, max_len), dtype=torch.float32)
    for i, aa in enumerate(seq):
        col = offset + i
        if col >= max_len:
            break
        row = BLOSUM62.get(aa.upper())
        if row is not None:
            t[:, col] = torch.tensor(row, dtype=torch.float32)
    return t


# ---------------------------------------------------------------------------
# variant generation
# ---------------------------------------------------------------------------
def aa_background(peptides):
    """Background amino-acid frequency over the training peptides."""
    counts = {a: 0 for a in STANDARD_AA}
    for p in peptides:
        for a in p:
            if a in counts:
                counts[a] += 1
    aas = sorted(counts)
    freq = np.array([counts[a] for a in aas], dtype=np.float64)
    return aas, freq / freq.sum()


def sample_flank(rng, aas, probs, n):
    if n <= 0:
        return ""
    return "".join(rng.choice(aas, size=n, p=probs))


def central_9mer_span(seq):
    """Start index of the centred 9-mer."""
    return max(0, (len(seq) - 9) // 2)


def make_variants(seq, rng, aas, probs, total_len, n_resample):
    """Return list of (condition, variant_seq, offset, j) tuples.

    ``offset`` is the tensor column the sequence starts at; ``j`` is the number
    of left-hand flanking residues (NaN where the notion does not apply).
    """
    L = len(seq)
    out = []

    # --- reference: exactly how the model saw it in training -----------------
    out.append(("reference", seq, 0, np.nan))

    # --- 1. padding-offset shift (architecture probe, OOD for k>0) -----------
    for k in range(0, PEP_MAX_LEN - L + 1):
        out.append(("shift_pad", seq, k, np.nan))

    # --- 2. flank-scaffold shift (in-distribution, primary condition) --------
    pad_total = total_len - L
    if pad_total >= 0:
        for j in range(0, pad_total + 1):
            left = sample_flank(rng, aas, probs, j)
            right = sample_flank(rng, aas, probs, pad_total - j)
            out.append(("shift_flank", left + seq + right, 0, j))

        # --- 3. flank resampling at fixed j (null for condition 2) ----------
        j_fixed = pad_total // 2
        for _ in range(n_resample):
            left = sample_flank(rng, aas, probs, j_fixed)
            right = sample_flank(rng, aas, probs, pad_total - j_fixed)
            out.append(("flank_resample", left + seq + right, 0, j_fixed))

    # --- 4. central 9-mer replaced (content sensitivity) ---------------------
    s = central_9mer_span(seq)
    for _ in range(n_resample):
        new_core = sample_flank(rng, aas, probs, min(9, len(seq)))
        mut = seq[:s] + new_core + seq[s + len(new_core):]
        out.append(("core_mutate", mut, 0, np.nan))

    # --- 5. whole-peptide scramble (rules out bag-of-amino-acids) -----------
    for _ in range(n_resample):
        chars = list(seq)
        rng.shuffle(chars)
        out.append(("scramble_all", "".join(chars), 0, np.nan))

    return out


# ---------------------------------------------------------------------------
# training-set overlap audit
# ---------------------------------------------------------------------------
def overlap_audit(variant_df, train_df):
    """Flag each variant for leakage against the training set.

    exact_pair     same (allele, sequence) present in training
    seq_any_allele sequence present in training under any allele
    core9_in_train the variant's central 9-mer occurs inside some training
                   peptide for the SAME allele

    ``core9_in_train`` is expected to be high by construction -- the whole
    point is to reuse a real peptide's core -- and is reported for honesty,
    not as something to filter out. ``exact_pair`` is the one that must be ~0.
    """
    pair_set = set(zip(train_df["HLA"], train_df["peptide"]))
    seq_set = set(train_df["peptide"])
    by_allele = train_df.groupby("HLA")["peptide"].apply(list).to_dict()

    exact, any_allele, core_in = [], [], []
    core_cache = {}
    for hla, seq in zip(variant_df["HLA"], variant_df["variant_seq"]):
        exact.append((hla, seq) in pair_set)
        any_allele.append(seq in seq_set)

        s = central_9mer_span(seq)
        core = seq[s:s + 9]
        key = (hla, core)
        hit = core_cache.get(key)
        if hit is None:
            hit = any(core in p for p in by_allele.get(hla, ()))
            core_cache[key] = hit
        core_in.append(hit)

    variant_df["exact_pair_in_train"] = exact
    variant_df["seq_in_train_any_allele"] = any_allele
    variant_df["core9_in_train_same_allele"] = core_in
    return variant_df


# ---------------------------------------------------------------------------
# inference over fold checkpoints
# ---------------------------------------------------------------------------
def load_models(ckpt_dir, cfg, build=build_from_config):
    """Load every fold checkpoint, taking the architecture from the WEIGHTS.

    core_conditioning is read off the state dict rather than the config: a
    checkpoint trained without it has no ``core.cond_gate``, and trusting a
    config that has since been flipped would either crash here or, worse,
    quietly evaluate a different architecture than the one that was trained.
    """
    paths = sorted(p for p in os.listdir(ckpt_dir) if p.endswith(".pt"))
    if not paths:
        raise FileNotFoundError(f"no .pt checkpoints in {ckpt_dir}")
    models, conditioned = [], []
    for p in paths:
        sd = torch.load(os.path.join(ckpt_dir, p), map_location=DEVICE)
        has_cond = "core.cond_gate" in sd
        arch = dict(cfg)
        arch["core_conditioning"] = has_cond
        m = build(arch).to(DEVICE)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
        conditioned.append(has_cond)

    if len(set(conditioned)) > 1:
        raise ValueError(
            f"checkpoints in {ckpt_dir} disagree on core_conditioning "
            f"({dict(zip(paths, conditioned))}) -- they cannot be ensembled")
    if conditioned[0] != cfg.get("core_conditioning", False):
        print(f"  note: checkpoints have core_conditioning={conditioned[0]}, "
              f"config says {cfg.get('core_conditioning', False)}; using the checkpoints")
    gates = [float(m.core.cond_gate.item()) for m in models if m.core.cond_gate is not None]
    if gates:
        print(f"  learned cond_gate per fold: "
              f"{', '.join(f'{g:.4f}' for g in gates)}")
    return models, paths


@torch.no_grad()
def predict_variants(models, variant_df, pseudo_table, batch_size=256):
    """Mean cls_prob / reg_yhat across fold checkpoints (matches test-time use)."""
    mhc_cache = {}

    def enc_mhc(key):
        t = mhc_cache.get(key)
        if t is None:
            t = simple_blosum10_encode(pseudo_table.get(key), PSEUDO_LEN)[0]
            mhc_cache[key] = t
        return t

    probs = np.zeros(len(variant_df))
    yhats = np.zeros(len(variant_df))
    core_idx = np.zeros(len(variant_df), dtype=int)
    core_conf = np.zeros(len(variant_df))
    seqs = variant_df["variant_seq"].tolist()
    offs = variant_df["offset"].tolist()
    hlas = variant_df["HLA"].tolist()

    for start in range(0, len(seqs), batch_size):
        sl = slice(start, start + batch_size)
        pep = torch.stack([encode_pep_at(s, o) for s, o in zip(seqs[sl], offs[sl])])
        mhc = torch.stack([enc_mhc(h) for h in hlas[sl]])
        pep, mhc = pep.to(DEVICE), mhc.to(DEVICE)

        p_acc = np.zeros(pep.shape[0])
        y_acc = np.zeros(pep.shape[0])
        pos_acc = None
        for m in models:
            # phase2 returns 3 values, phase3 adds the two coupling tensors
            out = m(pep, mhc)
            cls_logit, reg_yhat, pos_prob = out[0], out[1], out[2]
            p_acc += torch.sigmoid(cls_logit).cpu().numpy().ravel()
            y_acc += reg_yhat.cpu().numpy().ravel()
            pp = pos_prob.cpu().numpy()
            pos_acc = pp if pos_acc is None else pos_acc + pp
        probs[sl] = p_acc / len(models)
        yhats[sl] = y_acc / len(models)
        # the core the ensemble points at: argmax of the averaged position
        # distribution, in TENSOR coordinates (0 .. 25-9)
        pos_mean = pos_acc / len(models)
        core_idx[sl] = pos_mean.argmax(axis=1)
        core_conf[sl] = pos_mean.max(axis=1)

    variant_df["cls_prob"] = probs
    variant_df["reg_yhat"] = yhats
    variant_df["core_idx_tensor"] = core_idx
    variant_df["core_conf"] = core_conf

    # Where does the peptide itself start in the tensor? shift_pad moves the
    # whole tensor window (offset=k); shift_flank keeps offset 0 and pushes the
    # peptide right with j flanking residues. Subtracting this gives the core
    # position RELATIVE TO THE PEPTIDE, which is what must stay constant if the
    # model is tracking the same 9-mer as it moves.
    pep_start = (variant_df["offset"].fillna(0).astype(int)
                 + variant_df["left_flank_n"].fillna(0).astype(int))
    variant_df["peptide_start"] = pep_start
    variant_df["core_rel"] = variant_df["core_idx_tensor"] - pep_start

    anchor_len = variant_df["anchor_seq"].str.len()
    valid = (variant_df["core_rel"] >= 0) & (variant_df["core_rel"] <= anchor_len - 9)
    variant_df["core_rel_valid"] = valid
    variant_df["core_9mer"] = [
        (a[c:c + 9] if v else "")
        for a, c, v in zip(variant_df["anchor_seq"],
                           variant_df["core_rel"], valid)
    ]
    return variant_df


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def summarise(df):
    """Per-(case, condition) spread, plus deviation from the reference."""
    ref = (df[df.condition == "reference"]
           .set_index("case_id")[["reg_yhat", "cls_prob", "core_rel",
                                  "core_9mer", "core_idx_tensor"]]
           .rename(columns={"reg_yhat": "ref_yhat", "cls_prob": "ref_prob",
                            "core_rel": "ref_core_rel", "core_9mer": "ref_core_9mer",
                            "core_idx_tensor": "ref_core_idx"}))
    d = df[df.condition != "reference"].join(ref, on="case_id")
    d["abs_dev_yhat"] = (d.reg_yhat - d.ref_yhat).abs()
    d["abs_dev_prob"] = (d.cls_prob - d.ref_prob).abs()
    # Core tracking: does the model still point at the SAME 9-mer after the
    # peptide moved? core_rel is already peptide-relative, so "same core" means
    # core_rel unchanged. This is the direct question -- prediction stability
    # alone does not answer it, since a model can keep its output while picking
    # a different 9-mer.
    d["core_drift"] = (d.core_rel - d.ref_core_rel).abs()
    d["same_core"] = (d.core_9mer == d.ref_core_9mer) & (d.core_9mer != "")
    # Degenerate alternative: the model points at the same ABSOLUTE tensor slot
    # no matter where the peptide went -- a fixed positional habit, not core
    # recognition. These two are mutually exclusive once the peptide has
    # actually moved, so a high rate here is the failure mode "same_core" must
    # beat, not just a footnote.
    d["fixed_index"] = (d.core_idx_tensor == d.ref_core_idx) & (d.peptide_start != 0)
    d["peptide_moved"] = d.peptide_start != 0

    rows = []
    for (case, cond), g in d.groupby(["case_id", "condition"]):
        row = {
            "case_id": case,
            "condition": cond,
            "HLA": g.HLA.iloc[0],
            "n_variants": len(g),
            "ref_yhat": g.ref_yhat.iloc[0],
            "yhat_sd": g.reg_yhat.std(ddof=0),
            "yhat_range": g.reg_yhat.max() - g.reg_yhat.min(),
            "yhat_mean_absdev": g.abs_dev_yhat.mean(),
            "yhat_max_absdev": g.abs_dev_yhat.max(),
            "prob_sd": g.cls_prob.std(ddof=0),
            "prob_range": g.cls_prob.max() - g.cls_prob.min(),
            # --- core tracking ---
            "same_core_rate": float(g.same_core.mean()),
            "core_drift_mean": float(g.core_drift.mean()),
            "core_drift_max": float(g.core_drift.max()),
            "core_conf_mean": float(g.core_conf.mean()),
            "ref_core_9mer": g.ref_core_9mer.iloc[0],
            # restricted to variants where the peptide actually moved
            "n_moved": int(g.peptide_moved.sum()),
            "same_core_rate_moved": (float(g.loc[g.peptide_moved, "same_core"].mean())
                                     if g.peptide_moved.any() else float("nan")),
            "fixed_index_rate_moved": (float(g.loc[g.peptide_moved, "fixed_index"].mean())
                                       if g.peptide_moved.any() else float("nan")),
        }
        # binding-call flips against fixed external cutoffs
        for name, thr in THRESHOLDS.items():
            ref_call = g.ref_yhat.iloc[0] >= thr
            row[f"flip_rate_{name}"] = float(((g.reg_yhat >= thr) != ref_call).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def headline(per_case):
    """Condition-level medians and the two comparisons that carry the claim."""
    agg = (per_case.groupby("condition")
           .agg(n_cases=("case_id", "nunique"),
                median_yhat_sd=("yhat_sd", "median"),
                median_yhat_range=("yhat_range", "median"),
                median_mean_absdev=("yhat_mean_absdev", "median"),
                median_max_absdev=("yhat_max_absdev", "median"),
                median_prob_range=("prob_range", "median"),
                median_flip_500nM=("flip_rate_500nM", "median"),
                mean_flip_500nM=("flip_rate_500nM", "mean"),
                mean_same_core_rate=("same_core_rate", "mean"),
                mean_core_drift=("core_drift_mean", "mean"),
                mean_core_conf=("core_conf_mean", "mean"),
                same_core_moved=("same_core_rate_moved", "mean"),
                fixed_index_moved=("fixed_index_rate_moved", "mean"))
           .reset_index())

    def med(cond, col):
        s = agg.loc[agg.condition == cond, col]
        return float(s.iloc[0]) if len(s) else float("nan")

    # ratio_1 compares spread with spread (range vs range).
    # ratios 2-3 compare deviation-from-reference with deviation-from-reference
    # (mean_absdev vs mean_absdev) -- never mix the two metrics.
    shift_range = med("shift_flank", "median_yhat_range")
    null_range = med("flank_resample", "median_yhat_range")
    shift_dev = med("shift_flank", "median_mean_absdev")
    core_dev = med("core_mutate", "median_mean_absdev")
    scram_dev = med("scramble_all", "median_mean_absdev")

    verdict = {
        "shift_flank_range_over_flank_resample_range":
            shift_range / null_range if null_range else float("nan"),
        "core_mutate_dev_over_shift_flank_dev":
            core_dev / shift_dev if shift_dev else float("nan"),
        "scramble_all_dev_over_shift_flank_dev":
            scram_dev / shift_dev if shift_dev else float("nan"),
        "reading": (
            "ratio_1 near 1 => position adds no instability beyond flank identity. "
            "ratio_2 and ratio_3 >> 1 => the model is sensitive to content, so the "
            "invariance is not the degenerate kind. ratio_3 specifically rules out "
            "a composition-only (bag-of-amino-acids) model."
        ),
    }
    # The direct question: with the peptide moved, does the model still point
    # at the same 9-mer? Reported only for the conditions where the peptide
    # moved and its content was untouched.
    def grab(cond, col):
        s = agg.loc[agg.condition == cond, col]
        return float(s.iloc[0]) if len(s) else float("nan")

    verdict["core_tracking"] = {
        cond: {
            "same_core_rate": grab(cond, "same_core_moved"),
            "fixed_index_rate": grab(cond, "fixed_index_moved"),
            "mean_core_drift": grab(cond, "mean_core_drift"),
        }
        for cond in ("shift_pad", "shift_flank")
    }
    verdict["core_tracking"]["reading"] = (
        "same_core_rate is the answer to 'does it find the same 9-mer when the "
        "position changes'. fixed_index_rate is the competing explanation -- the "
        "model pointing at a fixed absolute slot regardless of where the peptide "
        "went. High same_core with low fixed_index means real tracking; the "
        "reverse means a positional habit that prediction-stability alone would "
        "have hidden."
    )
    verdict["paired_tests"] = paired_tests(per_case)
    return agg, verdict


def paired_tests(per_case):
    """Case-paired Wilcoxon tests. Each case contributes one pair, so the
    peptide itself is controlled for and the n is the number of cases."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"error": "scipy not available; ratios reported without p-values"}

    def pair(cond_a, col_a, cond_b, col_b, alt):
        a = per_case[per_case.condition == cond_a].set_index("case_id")[col_a]
        b = per_case[per_case.condition == cond_b].set_index("case_id")[col_b]
        joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        if len(joined) < 10 or (joined.a - joined.b).abs().sum() == 0:
            return {"n": int(len(joined)), "note": "too few pairs or zero differences"}
        stat, p = wilcoxon(joined.a, joined.b, alternative=alt)
        return {
            "n_cases": int(len(joined)),
            "median_a": float(joined.a.median()),
            "median_b": float(joined.b.median()),
            "statistic": float(stat),
            "p_value": float(p),
            "alternative": alt,
        }

    return {
        # is shift spread LARGER than the flank-identity null? we hope not
        "shift_flank_gt_flank_resample": pair(
            "shift_flank", "yhat_range", "flank_resample", "yhat_range", "greater"),
        # is core mutation deviation LARGER than shift deviation? we hope yes
        "core_mutate_gt_shift_flank": pair(
            "core_mutate", "yhat_mean_absdev", "shift_flank", "yhat_mean_absdev",
            "greater"),
        # is scramble deviation LARGER than shift deviation? we hope yes
        "scramble_all_gt_shift_flank": pair(
            "scramble_all", "yhat_mean_absdev", "shift_flank", "yhat_mean_absdev",
            "greater"),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="reports_phase2_shiftexp/checkpoints")
    ap.add_argument("--out-dir", default="reports_register_shift")
    ap.add_argument("--n-cases", type=int, default=300)
    ap.add_argument("--total-len", type=int, default=21,
                    help="fixed scaffold length for shift_flank (must be <=25)")
    ap.add_argument("--max-anchor-len", type=int, default=17,
                    help="only anchor on peptides this short, so there is room to shift")
    ap.add_argument("--n-resample", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=("phase2", "phase3"), default="phase2",
                    help="which architecture the checkpoints hold")
    ap.add_argument("--config", default=None,
                    help="config json to take the architecture from; defaults "
                         "to config/phase2.json or config/phase3.json")
    args = ap.parse_args()

    build = build_from_config if args.model == "phase2" else build_phase3
    cfg_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config",
        f"{args.model}.json")
    cfg = load_config(cfg_path)
    print(f"model={args.model} | arch config {cfg_path}")
    out_dir = _resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train_df = pd.read_csv(_resolve(cfg["train_txt"]), sep="\t")
    test_df = pd.read_csv(_resolve(cfg["test_txt"]), sep="\t")
    pseudo = PseudoSeqTable.load(_resolve(cfg["pseudo_csv"]))

    # anchors: held-out peptides only, short enough to leave shift room,
    # stratified across the affinity range so the result is not binder-only.
    cand = test_df[test_df.peptide.str.len() <= args.max_anchor_len].copy()
    cand = cand[cand.peptide.apply(lambda p: set(p) <= STANDARD_AA)]
    cand["stratum"] = pd.qcut(cand.affinity, q=5, labels=False, duplicates="drop")
    per = max(1, args.n_cases // cand.stratum.nunique())
    anchors = (cand.groupby("stratum", group_keys=False)
               .apply(lambda g: g.sample(min(per, len(g)), random_state=args.seed))
               .reset_index(drop=True))
    print(f"anchors: {len(anchors)} (len<={args.max_anchor_len}, from held-out test set)")

    aas, probs_bg = aa_background(train_df.peptide.tolist())

    records = []
    for i, r in anchors.iterrows():
        for cond, vseq, off, j in make_variants(
                r.peptide, rng, aas, probs_bg, args.total_len, args.n_resample):
            records.append({
                "case_id": i,
                "HLA": r.HLA,
                "anchor_seq": r.peptide,
                "anchor_affinity": r.affinity,
                "anchor_label": r.label,
                "condition": cond,
                "variant_seq": vseq,
                "offset": off,
                "left_flank_n": j,
            })
    variants = pd.DataFrame(records)
    print(f"variants: {len(variants)}")
    print(variants.condition.value_counts().to_string())

    variants = overlap_audit(variants, train_df)
    leak = {
        "n_variants": int(len(variants)),
        "exact_pair_in_train": int(variants.exact_pair_in_train.sum()),
        "seq_in_train_any_allele": int(variants.seq_in_train_any_allele.sum()),
        "core9_in_train_same_allele": int(variants.core9_in_train_same_allele.sum()),
        "note": ("exact_pair_in_train must be ~0 for the experiment to be clean. "
                 "core9_in_train_same_allele is high BY DESIGN -- the anchors reuse "
                 "real peptides, so their cores were seen in training. This "
                 "experiment measures prediction stability, not generalisation to "
                 "unseen cores."),
    }
    print("\nleakage audit:", json.dumps({k: v for k, v in leak.items() if k != "note"}))

    models, names = load_models(_resolve(args.ckpt_dir), cfg, build=build)
    print(f"loaded {len(models)} checkpoints: {names} | device {DEVICE}")

    variants = predict_variants(models, variants, pseudo)
    variants.to_csv(os.path.join(out_dir, "variant_predictions.tsv"),
                    sep="\t", index=False)

    per_case = summarise(variants)
    per_case.to_csv(os.path.join(out_dir, "per_case_summary.tsv"),
                    sep="\t", index=False)

    agg, verdict = headline(per_case)
    agg.to_csv(os.path.join(out_dir, "condition_summary.tsv"), sep="\t", index=False)

    summary = {
        "args": vars(args),
        "thresholds_affinity": THRESHOLDS,
        # The caveat has to follow the count, not accompany it unconditionally:
        # earlier runs printed "0/68 use PLACEHOLDER" next to "carries no real
        # signal", and that pair was read as the run being placeholder-based.
        "pseudo_sequence_warning": (
            f"{pseudo.n_placeholder}/{len(pseudo)} alleles use PLACEHOLDER "
            "pseudo-sequences. The allele branch carries no real signal there, "
            "so no cross-allele or per-allele claim can be made from this run."
            if pseudo.n_placeholder
            else f"0/{len(pseudo)} alleles use placeholder pseudo-sequences; "
                 "all are real."
        ),
        "leakage_audit": leak,
        "condition_summary": agg.to_dict(orient="records"),
        "verdict": verdict,
    }
    with open(os.path.join(out_dir, "register_shift_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== condition summary ===")
    print(agg.to_string(index=False))
    print("\n=== core tracking: does it find the SAME 9-mer when moved? ===")
    ct = verdict["core_tracking"]
    for cond in ("shift_pad", "shift_flank"):
        v = ct[cond]
        print(f"  {cond:14s} same_core {v['same_core_rate']:.3f} | "
              f"fixed_index {v['fixed_index_rate']:.3f} | "
              f"drift {v['mean_core_drift']:.2f}")
    print("  " + ct["reading"])

    print("\n=== verdict ratios ===")
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("reading", "core_tracking")},
                     indent=2, default=str))
    print("\n" + verdict["reading"])
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
