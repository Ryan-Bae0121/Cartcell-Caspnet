"""Figure for the register-shift experiment (reads reports_register_shift/).

Panel A  per-case deviation from the reference prediction, by condition.
         Distributions, not bars-of-medians: the claim rests on the spread.
Panel B  the paired test that carries the claim -- per-case shift spread against
         the flank-identity null. Points on the y=x line mean moving the peptide
         costs no more stability than swapping its flanks.

  python plot_register_shift.py [--dark]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# dataviz reference palette: categorical slots 1 (blue) and 2 (orange).
# Validated: adjacent CVD dE 24.7 protan / 32.7 tritan, normal-vision 33.6.
THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#8a8984", grid="#e4e3df",
                  invariance="#2a78d6", sensitivity="#eb6834"),
    "dark": dict(surface="#1a1a19", primary="#f0efec", secondary="#c3c2b7",
                 muted="#8a8984", grid="#333331",
                 invariance="#3987e5", sensitivity="#d95926"),
}

# order: invariance conditions first (expect small), controls after (expect large)
CONDITIONS = [
    ("shift_pad", "invariance", "Padding offset\n(architecture probe, OOD)"),
    ("flank_resample", "invariance", "Flank identity only\n(null: position fixed)"),
    ("shift_flank", "invariance", "Register shift\n(position varies)"),
    ("scramble_all", "sensitivity", "Whole peptide shuffled\n(order destroyed)"),
    ("core_mutate", "sensitivity", "Central 9-mer replaced\n(content destroyed)"),
]


def build(mode, per_case, verdict, out_path):
    c = THEME[mode]
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor(c["surface"])

    # ---------------- Panel A : distribution by condition ------------------
    axA.set_facecolor(c["surface"])
    data, colors = [], []
    for key, group, _ in CONDITIONS:
        data.append(per_case.loc[per_case.condition == key, "yhat_mean_absdev"].values)
        colors.append(c[group])

    ypos = np.arange(len(CONDITIONS))
    bp = axA.boxplot(data, positions=ypos, vert=False, widths=0.55,
                     showfliers=False, patch_artist=True, zorder=3)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.22)
        patch.set_edgecolor(col)
        patch.set_linewidth(2.0)
    for el, col in zip(bp["medians"], colors):
        el.set_color(col)
        el.set_linewidth(2.6)
    for part in ("whiskers", "caps"):
        for i, el in enumerate(bp[part]):
            el.set_color(colors[i // 2])
            el.set_linewidth(1.4)

    # jittered per-case points -- 300 cases, so show them
    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate(zip(data, colors)):
        axA.scatter(vals, i + rng.uniform(-0.17, 0.17, len(vals)),
                    s=5, color=col, alpha=0.30, linewidths=0, zorder=2)

    # median direct labels (selective: one per condition, in ink not series color)
    for i, (vals, (key, _, _)) in enumerate(zip(data, CONDITIONS)):
        m = float(np.median(vals))
        axA.annotate(f"{m:.4f}", (m, i - 0.36), ha="center", va="top",
                     fontsize=9, color=c["secondary"], zorder=5)

    axA.set_xscale("log")
    axA.set_yticks(ypos)
    axA.set_yticklabels([lab for _, _, lab in CONDITIONS],
                        fontsize=9.5, color=c["primary"])
    axA.set_xlabel("Mean |Δ predicted affinity| vs reference   (log scale)",
                   fontsize=10.5, color=c["secondary"])
    axA.set_title("A   Moving the peptide barely moves the prediction;\n"
                  "      breaking its content does",
                  fontsize=12.5, color=c["primary"], loc="left", pad=14)
    axA.grid(axis="x", color=c["grid"], linewidth=0.8, zorder=0)
    axA.set_axisbelow(True)
    axA.invert_yaxis()
    for s in ("top", "right", "left"):
        axA.spines[s].set_visible(False)
    axA.spines["bottom"].set_color(c["grid"])
    axA.tick_params(colors=c["secondary"], length=0)

    # ---------------- Panel B : the paired test ----------------------------
    axB.set_facecolor(c["surface"])
    piv = (per_case.pivot(index="case_id", columns="condition", values="yhat_range")
           .dropna(subset=["shift_flank", "flank_resample"]))
    x, y = piv["flank_resample"].values, piv["shift_flank"].values

    lims = [min(x.min(), y.min()) * 0.7, max(x.max(), y.max()) * 1.4]
    axB.plot(lims, lims, color=c["muted"], linewidth=1.6, linestyle=(0, (5, 4)),
             zorder=2, label="equal spread (y = x)")
    axB.scatter(x, y, s=22, color=c["invariance"], alpha=0.55,
                linewidths=0.8, edgecolors=c["surface"], zorder=3,
                label="one held-out peptide")

    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlim(lims)
    axB.set_ylim(lims)
    axB.set_xlabel("Spread from flank identity alone  (null)",
                   fontsize=10.5, color=c["secondary"])
    axB.set_ylabel("Spread from shifting the register",
                   fontsize=10.5, color=c["secondary"])
    axB.set_title("B   Shift spread sits on the null line",
                  fontsize=12.5, color=c["primary"], loc="left", pad=14)
    axB.grid(color=c["grid"], linewidth=0.8, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        axB.spines[s].set_color(c["grid"])
    axB.tick_params(colors=c["secondary"])
    leg = axB.legend(loc="upper left", frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(c["secondary"])

    pt = verdict["paired_tests"]
    stat = (
        f"Wilcoxon, {pt['shift_flank_gt_flank_resample']['n_cases']} paired cases\n"
        f"shift > null:  p = {pt['shift_flank_gt_flank_resample']['p_value']:.2f}  (n.s.)\n"
        f"ratio of medians = "
        f"{verdict['shift_flank_range_over_flank_resample_range']:.2f}×"
    )
    axB.annotate(stat, (0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=9.5, color=c["secondary"], linespacing=1.6)

    # ---------------- shared caption --------------------------------------
    cap = (
        f"Phase 2, 5-fold mean · 300 held-out peptides · 16,448 variants · "
        f"no (allele, peptide) pair from training\n"
        f"Content sensitivity: 9-mer replacement {verdict['core_mutate_dev_over_shift_flank_dev']:.1f}× "
        f"the shift deviation (p = {pt['core_mutate_gt_shift_flank']['p_value']:.0e}) · "
        f"shuffling {verdict['scramble_all_dev_over_shift_flank_dev']:.1f}× "
        f"(p = {pt['scramble_all_gt_shift_flank']['p_value']:.0e})\n"
        f"Allele pseudo-sequences are placeholders in this run — no per-allele or "
        f"cross-allele claim is supported."
    )
    fig.text(0.012, -0.045, cap, fontsize=9, color=c["muted"], linespacing=1.7)

    # group legend for Panel A (identity never by color alone -> labels present too)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=c["invariance"], label="Position / flanks changed, core intact"),
               plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          color=c["sensitivity"], label="Core content broken (control)")]
    leg2 = axA.legend(handles=handles, loc="lower left", frameon=False, fontsize=9.5)
    for t in leg2.get_texts():
        t.set_color(c["secondary"])

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default="reports_register_shift")
    args = ap.parse_args()

    rd = args.report_dir if os.path.isabs(args.report_dir) else os.path.join(HERE, args.report_dir)
    per_case = pd.read_csv(os.path.join(rd, "per_case_summary.tsv"), sep="\t")
    with open(os.path.join(rd, "register_shift_summary.json")) as fh:
        verdict = json.load(fh)["verdict"]

    for mode in ("light", "dark"):
        build(mode, per_case, verdict,
              os.path.join(rd, f"register_shift_{mode}.png"))


if __name__ == "__main__":
    main()
