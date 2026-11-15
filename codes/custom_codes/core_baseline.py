"""Random baseline and calibration for the register-shift same-core rate.

Reproduces the published mean_same_core_rate from the raw TSV first; if that
does not match to 1e-12 the rest of the numbers are not trustworthy and the
script stops.

The choice space matters, so two baselines are reported:

  A  uniform over the model's ACTUAL output space -- core_idx_tensor in
     0..T-1 where T = 17 (25-mer tensor, 9-mer window). The model is not
     constrained to point inside the peptide, and in 27/300 reference cases
     it did not, so this is the true chance level for what it emits.
  B  uniform over the valid anchor windows only (L-9+1 of them). This is the
     chance level GIVEN that the peptide's location is already known, i.e. it
     credits the model for locating the peptide and asks only about register.

Both are computed by enumeration rather than 1/n, because a 9-mer can repeat
inside one peptide and because same_core is defined on the 9-mer STRING, not
on the index.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

T = 17  # 25 - 9 + 1 tensor positions, matches core_idx_tensor range 0..16
RUNS = {
    "Phase2 (CNN)": "reports_register_shift_final",
    "Phase3 (CapsNet)": "reports_register_shift_phase3",
    "no-routing": "reports_register_shift_norouting",
}
CONDS = ["shift_flank", "shift_pad"]


def load(root, run_dir):
    p = os.path.join(root, run_dir, "variant_predictions.tsv")
    return pd.read_csv(p, sep="\t")


def attach_reference(df):
    """Same join as register_shift.summarise(), kept field-for-field."""
    ref = (df[df.condition == "reference"]
           .set_index("case_id")[["core_rel", "core_9mer", "core_idx_tensor"]]
           .rename(columns={"core_rel": "ref_core_rel",
                            "core_9mer": "ref_core_9mer",
                            "core_idx_tensor": "ref_core_idx"}))
    d = df[df.condition != "reference"].join(ref, on="case_id")
    d["same_core"] = (d.core_9mer == d.ref_core_9mer) & (d.core_9mer != "")
    d["core_drift"] = (d.core_rel - d.ref_core_rel).abs()
    return d


def published_rate(d, cond):
    """mean over cases of (mean over that case's variants) -- the published stat."""
    per_case = d[d.condition == cond].groupby("case_id").same_core.mean()
    return float(per_case.mean()), per_case


def windows(anchor):
    """Every 9-mer window of the anchor, as (core_rel, string)."""
    n = len(anchor) - 9 + 1
    return [(r, anchor[r:r + 9]) for r in range(max(n, 0))]


def case_baselines(anchor, ref_core, ref_core_rel, j_values):
    """Expected same-core rate and drift for one case, under A and B.

    ``j_values`` are the peptide_start offsets of that case's variants; the
    published statistic averages over them, so the baseline must too.
    """
    wins = windows(anchor)
    by_rel = dict(wins)
    # A match requires the STRING to equal the reference core, and the
    # reference core must itself be a real 9-mer (27 cases fail this and can
    # never score, by the definition of same_core -- that is not a bug to
    # patch here, it is part of what the published number measures).
    if not ref_core:
        match_rels = []
    else:
        match_rels = [r for r, s in wins if s == ref_core]

    pA, pB, dA, dB = [], [], [], []
    for j in j_values:
        # --- A: uniform over core_idx 0..T-1, core_rel = core_idx - j -------
        rels_A = np.arange(T) - j
        hit_A = np.isin(rels_A, match_rels).mean() if match_rels else 0.0
        pA.append(hit_A)
        dA.append(np.abs(rels_A - ref_core_rel).mean())

        # --- B: uniform over the anchor's own windows -----------------------
        rels_B = np.array([r for r, _ in wins])
        if len(rels_B) == 0:
            pB.append(0.0)
            dB.append(np.nan)
            continue
        hit_B = np.isin(rels_B, match_rels).mean() if match_rels else 0.0
        pB.append(hit_B)
        dB.append(np.abs(rels_B - ref_core_rel).mean())

    return (float(np.mean(pA)), float(np.mean(pB)),
            float(np.mean(dA)), float(np.nanmean(dB)))


def analyse(root, label, run_dir, cond):
    raw = load(root, run_dir)
    d = attach_reference(raw)
    rate, per_case = published_rate(d, cond)

    sf = d[d.condition == cond]
    ref_rows = raw[raw.condition == "reference"].set_index("case_id")

    pA = pB = dA = dB = []
    pA, pB, dA, dB = [], [], [], []
    n_dead = 0
    for cid, g in sf.groupby("case_id"):
        anchor = ref_rows.at[cid, "anchor_seq"]
        ref_core = ref_rows.at[cid, "core_9mer"]
        ref_core = "" if pd.isna(ref_core) else str(ref_core)
        ref_rel = int(ref_rows.at[cid, "core_rel"])
        if not ref_core:
            n_dead += 1
        a, b, da, db = case_baselines(anchor, ref_core, ref_rel,
                                      g.peptide_start.tolist())
        pA.append(a); pB.append(b); dA.append(da); dB.append(db)

    return {
        "label": label,
        "run_dir": run_dir,
        "condition": cond,
        "n_cases": int(sf.case_id.nunique()),
        "n_rows": int(len(sf)),
        "measured_same_core": rate,
        "measured_drift": float(sf.groupby("case_id").core_drift.mean().mean()),
        "baseline_A": float(np.mean(pA)),
        "baseline_B": float(np.mean(pB)),
        "drift_A": float(np.mean(dA)),
        "drift_B": float(np.nanmean(dB)),
        "n_ref_core_invalid": n_dead,
        "per_case": per_case,
        "sf": sf,
    }


def main():
    root = sys.argv[1]
    out = {}
    for cond in CONDS:
        for label, run_dir in RUNS.items():
            r = analyse(root, label, run_dir, cond)
            pub = json.load(open(os.path.join(root, run_dir,
                                              "register_shift_summary.json")))
            want = [c for c in pub["condition_summary"]
                    if c["condition"] == cond][0]["mean_same_core_rate"]
            got = r["measured_same_core"]
            if abs(got - want) >= 1e-12:
                sys.exit(f"{label}/{cond}: recomputed {got} != published {want}")
            out[(cond, label)] = r
    print("all recomputations match the published JSON to 1e-12\n")

    print(f"{'condition':13s} {'model':18s} {'measured':>9s} {'baseA':>7s} {'xA':>5s} "
          f"{'baseB':>7s} {'xB':>5s} {'drift':>6s} {'dA':>5s} {'dB':>5s}")
    for (cond, label), r in out.items():
        print(f"{cond:13s} {label:18s} {r['measured_same_core']:9.4f} "
              f"{r['baseline_A']:7.4f} {r['measured_same_core']/r['baseline_A']:5.2f} "
              f"{r['baseline_B']:7.4f} {r['measured_same_core']/r['baseline_B']:5.2f} "
              f"{r['measured_drift']:6.3f} {r['drift_A']:5.3f} {r['drift_B']:5.3f}")

    from scipy.stats import spearmanr
    payload = {"table": [], "calibration": {}, "spearman": {}}
    for (cond, label), r in out.items():
        payload["table"].append({"condition": cond, "model": label,
            **{k: v for k, v in r.items()
               if k not in ("per_case", "sf", "label", "run_dir")}})

    print()
    for cond in CONDS:
        sf = out[(cond, "Phase2 (CNN)")]["sf"].copy()
        sf["decile"] = pd.qcut(sf.core_conf, 10, labels=False, duplicates="drop")
        cal = sf.groupby("decile").agg(
            n=("same_core", "size"), conf_mid=("core_conf", "median"),
            conf_lo=("core_conf", "min"), conf_hi=("core_conf", "max"),
            same_core=("same_core", "mean"))
        # Wilson 95% CI, so a decile's wobble can be told from a real step.
        import math
        z = 1.959963985
        lo, hi = [], []
        for n, p in zip(cal.n, cal.same_core):
            c = (p + z*z/(2*n)) / (1 + z*z/n)
            h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
            lo.append(max(c-h, 0.0)); hi.append(min(c+h, 1.0))
        rho, pv = spearmanr(sf.core_conf, sf.same_core.astype(int))
        q75 = sf.core_conf.quantile(0.75)
        print(f"=== {cond} (Phase2) deciles of core_conf ===")
        print(cal.assign(ci_lo=lo, ci_hi=hi).to_string())
        print(f"  top quartile (conf>={q75:.4f}): {sf[sf.core_conf>=q75].same_core.mean():.4f}"
              f"   Spearman rho={rho:.4f} p={pv:.3g}\n")
        payload["calibration"][cond] = {
            "decile": [int(i) for i in cal.index],
            "n": [int(x) for x in cal.n],
            "conf_mid": [float(x) for x in cal.conf_mid],
            "same_core": [float(x) for x in cal.same_core],
            "ci_lo": [float(x) for x in lo], "ci_hi": [float(x) for x in hi],
            "baseline_A": out[(cond, "Phase2 (CNN)")]["baseline_A"],
            "baseline_B": out[(cond, "Phase2 (CNN)")]["baseline_B"],
            "top_quartile": float(sf[sf.core_conf >= q75].same_core.mean()),
        }
        payload["spearman"][cond] = {"rho": float(rho), "p": float(pv)}

    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cal.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
