"""Does core-pick confidence predict core-pick correctness?

The same-core rate on its own (41.2% for shift_flank) says nothing about whether
the model knows when it is right. This splits the per-variant rows into deciles
of ``core_conf`` -- the max of the CoreRegister position softmax -- and plots the
agreement rate in each, against the two random baselines.

Both shift conditions are drawn, because they answer different questions and
have been conflated before:

  shift_flank  the primary, in-distribution condition. The peptide really moves
               inside a fixed-length 21-mer and the flanks are resampled.
  shift_pad    the architecture probe. The same string slid to a different
               tensor offset, out of distribution for k>0 -- an easier task,
               and the source of the 0.678 top-quartile figure.

  python plot_core_calibration.py [--dark] [--run reports_register_shift_final]
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

T = 17  # 25-mer tensor, 9-mer window -> core_idx_tensor in 0..16

# dataviz reference palette: categorical slots 1 (blue) and 2 (orange), the same
# pair plot_register_shift.py uses. Validated for both surfaces: adjacent CVD
# dE 24.7 protan / 32.7 tritan (light), 26.8 / 32.4 (dark); normal-vision 33.6 / 31.8.
THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#8a8984", grid="#e4e3df",
                  flank="#2a78d6", pad="#eb6834"),
    "dark": dict(surface="#1a1a19", primary="#f0efec", secondary="#c3c2b7",
                 muted="#8a8984", grid="#333331",
                 flank="#3987e5", pad="#d95926"),
}

SERIES = [
    ("shift_flank", "flank", "Register shift (in-distribution)"),
    ("shift_pad", "pad", "Padding offset (OOD probe)"),
]

Z = 1.959963985


def wilson(k, n):
    """Wilson score interval -- an exact-ish CI that behaves near 0 and 1.

    A normal approximation would put a decile's band outside [0,1] at the low
    end, where the first decile sits.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return max(c - h, 0.0), min(c + h, 1.0)


def load(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "variant_predictions.tsv"), sep="\t")
    # anchor_seq is already on every variant row (it is the case's peptide), so
    # it must not come across in the join or pandas refuses the overlap.
    ref = (df[df.condition == "reference"]
           .set_index("case_id")[["core_9mer", "core_rel"]]
           .rename(columns={"core_9mer": "ref_core_9mer",
                            "core_rel": "ref_core_rel"}))
    d = df[df.condition != "reference"].join(ref, on="case_id")
    d["same_core"] = (d.core_9mer == d.ref_core_9mer) & (d.core_9mer != "")
    return d


def baselines(d, cond):
    """Chance level for the same-core rate, by enumeration.

    A  uniform over the model's real output space (core_idx_tensor, T slots).
       The model is not constrained to point inside the peptide and in 27/300
       reference cases it did not, so this is chance for what it actually emits.
    B  uniform over the anchor's own 9-mer windows. Chance GIVEN that the
       peptide's location is already known -- it credits locating the peptide
       and asks only about register.

    Enumerated rather than 1/n because same_core compares the 9-mer STRING, so
    a repeated 9-mer inside one peptide counts as a match at two indices.
    """
    sub = d[d.condition == cond]
    pA, pB = [], []
    for _, g in sub.groupby("case_id"):
        row = g.iloc[0]
        anchor, ref_core = row.anchor_seq, row.ref_core_9mer
        ref_core = "" if pd.isna(ref_core) else str(ref_core)
        wins = [(r, anchor[r:r + 9]) for r in range(len(anchor) - 8)]
        hits = [r for r, s in wins if ref_core and s == ref_core]
        rels_B = np.array([r for r, _ in wins])
        a, b = [], []
        for j in g.peptide_start:
            a.append(np.isin(np.arange(T) - j, hits).mean() if hits else 0.0)
            b.append(np.isin(rels_B, hits).mean() if hits else 0.0)
        pA.append(np.mean(a))
        pB.append(np.mean(b))
    return float(np.mean(pA)), float(np.mean(pB))


def build(mode, d, out_path):
    c = THEME[mode]
    fig, ax = plt.subplots(figsize=(9.6, 5.9))
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    ax.grid(color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    base_A, base_B = baselines(d, "shift_flank")
    for y, label in [(base_B, f"chance if the peptide's position is known  {base_B:.3f}"),
                     (base_A, f"chance over the model's {T} output slots  {base_A:.3f}")]:
        ax.axhline(y, color=c["muted"], linewidth=1.4, linestyle=(0, (5, 4)),
                   zorder=1)
        ax.text(9.42, y + 0.013, label, fontsize=8.6, color=c["muted"],
                ha="right", va="bottom")

    for cond, slot, label in SERIES:
        sub = d[d.condition == cond].copy()
        sub["decile"] = pd.qcut(sub.core_conf, 10, labels=False,
                                duplicates="drop")
        g = sub.groupby("decile").agg(n=("same_core", "size"),
                                      k=("same_core", "sum"),
                                      conf=("core_conf", "median"))
        rate = g.k / g.n
        lo, hi = zip(*[wilson(k, n) for k, n in zip(g.k, g.n)])
        x = np.asarray(g.index, dtype=float)
        ax.fill_between(x, lo, hi, color=c[slot], alpha=0.16, linewidth=0,
                        zorder=2)
        ax.plot(x, rate, color=c[slot], linewidth=2.0, zorder=3)
        # A 2px surface ring keeps the markers readable where the two series
        # cross in the lowest deciles.
        ax.scatter(x, rate, s=46, color=c[slot], zorder=4,
                   edgecolors=c["surface"], linewidths=2)
        # Direct label at the right end, so identity is never colour-alone.
        ax.annotate(label, (x[-1], rate.iloc[-1]),
                    xytext=(-6, 15 if cond == "shift_pad" else -22),
                    textcoords="offset points", fontsize=9.5, color=c[slot],
                    ha="right", fontweight="demibold")

    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{i+1}" for i in range(10)], fontsize=9.5,
                       color=c["secondary"])
    ax.set_xlim(-0.45, 9.45)
    ax.set_ylim(0, 0.84)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 0.81, 0.1)],
                       fontsize=9.5, color=c["secondary"])
    # labelpad clears the median-conf row drawn under the ticks further down.
    ax.set_xlabel("core_conf decile  (1 = least confident, 10 = most)",
                  fontsize=10.5, color=c["secondary"], labelpad=30)
    ax.set_ylabel("same 9-mer as the reference pick", fontsize=10.5,
                  color=c["secondary"], labelpad=9)
    ax.set_title("Core-pick confidence barely predicts core-pick stability",
                 fontsize=13, color=c["primary"], loc="left", pad=16)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # The median core_conf of each decile, so the x axis is readable as a
    # confidence scale and not only as a rank.
    sub = d[d.condition == "shift_flank"].copy()
    sub["decile"] = pd.qcut(sub.core_conf, 10, labels=False, duplicates="drop")
    mids = sub.groupby("decile").core_conf.median()
    for i, m in mids.items():
        ax.text(i, -0.052, f"{m:.2f}", fontsize=7.6, color=c["muted"],
                ha="center", va="top")
    ax.text(-0.45, -0.052, "median\nconf", fontsize=7.6, color=c["muted"],
            ha="right", va="top", linespacing=1.4)

    cap = ("Phase2 (CNN), reports_phase2_final checkpoints, 5-fold ensemble · 300 held-out peptides · "
           "bands are Wilson 95% CIs\n"
           "shift_flank plateaus at ~47% from decile 6 on (Spearman rho = 0.142): the most confident picks are no "
           "more stable than the middling ones.\n"
           "shift_pad reaches 73% in the top decile (rho = 0.255), but the peptide never moves relative to its "
           "flanks there -- it is the easier, OOD task.")
    fig.text(0.012, -0.175, cap, fontsize=8.6, color=c["muted"], linespacing=1.75)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="reports_register_shift_final")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--both", action="store_true", default=True)
    args = ap.parse_args()

    run_dir = args.run if os.path.isabs(args.run) else os.path.join(HERE, args.run)
    d = load(run_dir)
    modes = ["light", "dark"] if args.both and not args.dark else \
            (["dark"] if args.dark else ["light"])
    for mode in modes:
        build(mode, d, os.path.join(run_dir, f"core_calibration_{mode}.png"))

    # The chance levels used to live only inside the PNG, and that caused a real
    # error: section 4.1 quoted 0.416 -- the flank_resample CONTROL value -- as a
    # "random null" for two months, because the actual baselines were not in any
    # file you could grep. They are written out here as well now.
    out = {
        "run_dir": os.path.abspath(run_dir),
        "T_output_slots": T,
        "baseline_definitions": {
            "A_uniform_over_output_slots":
                "uniform over the model's real output space (core_idx_tensor, "
                f"{T} slots); the model is not constrained to point inside the "
                "peptide, so this is chance for what it actually emits",
            "B_uniform_over_anchor_windows":
                "uniform over the anchor's own 9-mer windows; chance GIVEN that "
                "the peptide's location is already known -- it credits locating "
                "the peptide and asks only about register",
        },
        "note": "Enumerated per case, not 1/n: same_core compares the 9-mer "
                "STRING, so a repeated 9-mer inside one peptide counts as a "
                "match at two indices.",
        "conditions": {},
    }
    for cond in ("shift_flank", "shift_pad"):
        sub = d[d.condition == cond]
        if sub.empty:
            continue
        a, b = baselines(d, cond)
        measured = float(sub.same_core.mean())
        out["conditions"][cond] = {
            "chance_A_output_space": a,
            "chance_B_position_known": b,
            "measured_same_core_rate": measured,
            "measured_over_chance_A": (measured / a) if a else None,
            "measured_over_chance_B": (measured / b) if b else None,
            "n_variant_rows": int(len(sub)),
            "n_cases": int(sub.case_id.nunique()),
        }
    jpath = os.path.join(run_dir, "core_calibration_baselines.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {jpath}")
    for cond, v in out["conditions"].items():
        print(f"  {cond:12s} measured {v['measured_same_core_rate']:.4f} | "
              f"chance A {v['chance_A_output_space']:.4f} "
              f"({v['measured_over_chance_A']:.2f}x) | "
              f"chance B {v['chance_B_position_known']:.4f} "
              f"({v['measured_over_chance_B']:.2f}x)")


if __name__ == "__main__":
    main()
