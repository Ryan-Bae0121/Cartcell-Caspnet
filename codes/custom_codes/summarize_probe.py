"""Read the convergence-probe traces and answer the two questions they exist for.

  1. Where does the composite selection score flatten? Reported as the first
     epoch within 0.2% of the run's best, so "it peaked at 142" does not get
     read as "150 was still not enough" when the curve went flat at 70.
  2. Is mean |W| still climbing at the old 50-epoch cutoff? Reported as the
     growth over the last quarter of the run, relative to the growth over the
     first.

Usage:  python summarize_probe.py [reports_probe_*/trace/*.jsonl ...]
"""
import glob
import json
import os
import sys

# Inside this band of the best score the curve is flat for our purposes: the
# fold-to-fold spread of the reported composite is far larger than 0.2%.
PLATEAU_TOL = 0.002
OLD_BUDGET = 50


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def plateau_epoch(scores):
    """First epoch whose score is within PLATEAU_TOL of the best seen overall."""
    best = max(scores)
    # A score can be negative (val_loss selection); scale the tolerance by
    # magnitude rather than assuming the score lives near 1.
    thresh = best - abs(best) * PLATEAU_TOL
    for i, s in enumerate(scores):
        if s >= thresh:
            return i
    return len(scores) - 1


def describe(path):
    rows = load(path)
    if not rows:
        print(f"{path}: empty")
        return
    # phase3 and norouting both write phase3_fold0.jsonl and differ only by
    # output directory, so the run name has to come from the directory.
    run = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    tag = f"{run}/{os.path.basename(path).replace('.jsonl', '')}"
    scores = [r["score"] for r in rows]
    n = len(rows)
    best_ep = max(range(n), key=lambda i: scores[i])
    flat_ep = plateau_epoch(scores)

    print(f"\n--- {tag}  ({n} epochs, {rows[0]['select_on']}) ---")
    print(f"  best score    {scores[best_ep]:.4f} @ epoch {best_ep}")
    print(f"  flat from     epoch {flat_ep}  (within {PLATEAU_TOL:.1%} of best)")
    if best_ep >= n - 5:
        print(f"  ** still improving at the end -- {n} epochs is not enough either")
    elif flat_ep < OLD_BUDGET:
        print(f"  => {OLD_BUDGET} epochs was ENOUGH; best_epoch sitting near the "
              f"budget was selection noise on a flat curve")
    else:
        print(f"  => {OLD_BUDGET} epochs was SHORT; the curve is still rising there")

    # Only meaningful once the probe has actually passed the old budget; a
    # 2-epoch smoke run must not print a number labelled "@ 50".
    if n >= OLD_BUDGET:
        at50 = scores[OLD_BUDGET - 1]
        print(f"  score @ {OLD_BUDGET}   {at50:.4f}   "
              f"(best is {scores[best_ep] - at50:+.4f})")

    w = [r["w_abs_mean"] for r in rows if r.get("w_abs_mean") is not None]
    if not w:
        print("  |W|           n/a (no routing layer)")
        return

    from epoch_trace import INIT_W_ABS_MEAN
    q = max(len(w) // 4, 1)
    first, last = w[q] - w[0], w[-1] - w[-1 - q]
    print(f"  |W| init      {INIT_W_ABS_MEAN:.5f}")
    if len(w) >= OLD_BUDGET:
        print(f"  |W| @ {OLD_BUDGET}     {w[OLD_BUDGET - 1]:.5f}  "
              f"({w[OLD_BUDGET - 1] / INIT_W_ABS_MEAN:.2f}x init)")
    print(f"  |W| final     {w[-1]:.5f}  ({w[-1]/INIT_W_ABS_MEAN:.2f}x init) "
          f"@ epoch {len(w) - 1}")
    # Growth in the last quarter against growth in the first quarter. A ratio
    # near zero means training stopped pushing W; a ratio near one means the
    # 50-epoch cutoff simply interrupted a still-rising curve.
    ratio = last / first if first else float("nan")
    print(f"  growth Q4/Q1  {ratio:.3f}  (Q1 {first:+.5f}, Q4 {last:+.5f})")
    if ratio > 0.5:
        print("  => |W| is STILL CLIMBING: the budget was the binding constraint. "
              "Re-run 5-fold at the longer budget before touching the init scale.")
    elif ratio > 0.1:
        print("  => |W| is slowing but not flat: ambiguous, widen the budget further.")
    else:
        print("  => |W| has PLATEAUED: training found no reason to grow it. "
              "An init-scale experiment is now a controlled comparison.")


def main():
    paths = sys.argv[1:] or sorted(glob.glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "reports_probe_*", "trace", "*.jsonl")))
    if not paths:
        sys.exit("no traces found -- run run_convergence_probe.sh first")
    for p in paths:
        describe(p)
    print()


if __name__ == "__main__":
    main()
