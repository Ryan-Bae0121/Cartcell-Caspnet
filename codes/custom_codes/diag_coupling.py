"""Is dynamic routing doing anything at all?

The no-routing ablation (routing_iters=1) matched full routing on every
metric, which has two very different explanations:

  (a) routing works but the agreement signal is useless for this task, or
  (b) routing collapses to the uniform coupling that routing_iters=1 gives
      by construction, so there was never anything to ablate.

Those are distinguishable from the trained weights alone. With n_iters=1 the
logits b stay zero, softmax over 3 output capsules returns exactly 1/3, and
s is the plain mean of u_hat. So the question is how far the *trained*
3-iteration coupling departs from that 1/3.

Two departures matter and they are not the same thing:

  * magnitude   -- mean |c - 1/3|. Zero means routing literally reproduces
                   the uniform baseline.
  * input-dependence -- the std of c ACROSS peptides, per (primary, output)
                   cell. Routing can sit far from 1/3 yet be identical for
                   every input, in which case it is a fixed re-weighting that
                   the output layer can absorb -- still nothing to ablate.
                   This is the diagnostic the aggregate outputs of forward()
                   cannot show, which is why this script hooks the routing
                   module and takes the full (B, 56, 3) tensor.
"""
import argparse
import json
import os

import numpy as np

import torch

from data_provider import DataProvider
from models.phase3 import build_from_config
from train_phase2 import load_config, _resolve

DEVICE = torch.device("cpu")
N_OUT = 3


def load_fold_models(ckpt_dir, cfg):
    paths = sorted(p for p in os.listdir(ckpt_dir) if p.endswith(".pt"))
    models = []
    for p in paths:
        sd = torch.load(os.path.join(ckpt_dir, p), map_location=DEVICE)
        arch = dict(cfg)
        arch["core_conditioning"] = "core.cond_gate" in sd
        m = build_from_config(arch).to(DEVICE)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, paths


@torch.no_grad()
def collect_coupling(model, dp, idx, batch_size):
    """Return c of shape (N, n_in, 3) by hooking the routing module."""
    grabbed = []

    def hook(_mod, _inp, out):
        grabbed.append(out[1].detach().cpu())   # out = (v, c)

    h = model.routing.register_forward_hook(hook)
    try:
        for b in dp.iter_batches(idx, batch_size=batch_size):
            model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
    finally:
        h.remove()
    return torch.cat(grabbed, dim=0).numpy()


def summarise(c, label):
    uni = 1.0 / N_OUT
    dev = np.abs(c - uni)
    # std across peptides for each (primary capsule, output capsule) cell
    per_cell_std = c.std(axis=0)
    # how much of the spread in c is across inputs vs across cells
    cell_mean = c.mean(axis=0)
    return {
        "label": label,
        "n_peptides": int(c.shape[0]),
        "n_primary": int(c.shape[1]),
        "mean_abs_dev_from_uniform": float(dev.mean()),
        "max_abs_dev_from_uniform": float(dev.max()),
        "p99_abs_dev": float(np.percentile(dev, 99)),
        "mean_c": float(c.mean()),
        "min_c": float(c.min()),
        "max_c": float(c.max()),
        # input-dependence
        "mean_across_input_std": float(per_cell_std.mean()),
        "max_across_input_std": float(per_cell_std.max()),
        # spread of the per-cell MEANS: a fixed, input-independent re-weighting
        "std_of_cell_means": float(cell_mean.std()),
        # entropy per primary capsule; ln(3) = 1.0986 is uniform
        "mean_entropy": float((-(c * np.log(c + 1e-12)).sum(axis=2)).mean()),
        "uniform_entropy": float(np.log(N_OUT)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="reports_phase3_final/checkpoints")
    ap.add_argument("--config", default="config/phase3.json")
    ap.add_argument("--n-peptides", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="reports_coupling_diag.json")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    cfg = load_config(_resolve(args.config))
    dp = DataProvider(_resolve(cfg["test_txt"]), _resolve(cfg["pseudo_csv"]))
    idx = np.arange(min(args.n_peptides, len(dp.df)))

    models, names = load_fold_models(_resolve(args.ckpt_dir), cfg)
    label = args.label or os.path.basename(os.path.dirname(_resolve(args.ckpt_dir)))
    print(f"{label}: {len(models)} folds, {len(idx)} peptides, "
          f"routing_iters={cfg.get('routing_iters', 3)}")

    rows = []
    for m, nm in zip(models, names):
        c = collect_coupling(m, dp, idx, args.batch_size)
        r = summarise(c, f"{label}/{nm}")
        rows.append(r)
        print(f"  {nm}: |c-1/3| mean {r['mean_abs_dev_from_uniform']:.5f} "
              f"max {r['max_abs_dev_from_uniform']:.5f} | "
              f"across-input std {r['mean_across_input_std']:.5f} | "
              f"cell-mean std {r['std_of_cell_means']:.5f} | "
              f"entropy {r['mean_entropy']:.5f} (uniform {r['uniform_entropy']:.5f})")

    out = {"args": vars(args), "routing_iters": cfg.get("routing_iters", 3),
           "folds": rows}
    path = _resolve(args.out)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote -> {path}")


if __name__ == "__main__":
    main()
