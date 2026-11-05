"""Why is the trained coupling 90% input-independent?

diag_coupling.py showed the 3-iteration coupling departs from uniform but
that ~90% of the departure is a fixed per-cell pattern. The routing loop is

    b = 0
    for it in range(n_iters):
        c = softmax(b, dim=2)
        s = einsum(c, u_hat); v = squash(s)
        if it < n_iters - 1:
            b = b + einsum(u_hat, v)

so c can only become input-dependent through b, and b only moves by the
agreement term einsum(u_hat, v). If ||u_hat|| is small the increments are
small, softmax stays near uniform, and the next iteration adds an equally
small increment -- routing never gets going. W is initialised at 0.01*randn,
which makes that the first thing to check.

This replays the loop with the TRAINED W for more iterations than training
used, recording at each step how far c is from uniform and -- the number
that matters -- what fraction of c's variation is across peptides rather
than a fixed per-cell offset. If that fraction climbs with iterations, the
budget was simply too small. If it plateaus, the agreement signal itself
carries little peptide information and more iterations will not help.
"""
import argparse
import json
import os

import numpy as np
import torch

from data_provider import DataProvider
from models.common import squash
from models.phase3 import build_from_config
from train_phase2 import load_config, _resolve

DEVICE = torch.device("cpu")


def input_dependence(c):
    """Share of c's variance that is across peptides, not a fixed per-cell offset.

    c: (N, n_in, n_out). Decomposes total variance into the spread of each
    cell's mean (fixed pattern) plus the mean within-cell variance (input).
    """
    within = c.var(axis=0).mean()
    between = c.mean(axis=0).var()
    return float(within / (within + between + 1e-12))


@torch.no_grad()
def replay(model, dp, idx, batch_size, n_iters):
    """Capture the primary capsules, then run routing by hand for n_iters."""
    caught = []

    def pre_hook(_mod, inp):
        caught.append(inp[0].detach())

    h = model.routing.register_forward_pre_hook(pre_hook)
    try:
        for b in dp.iter_batches(idx, batch_size=batch_size):
            model(b["peptide"].to(DEVICE), b["pseudo"].to(DEVICE))
    finally:
        h.remove()
    u = torch.cat(caught, dim=0)                       # (N, n_in, in_dim)

    W = model.routing.W                                # (n_in, n_out, out, in)
    u_hat = torch.einsum("ijoc,bic->bijo", W, u)       # (N, n_in, n_out, out)

    steps = []
    b = torch.zeros(u.size(0), model.routing.n_in, model.routing.n_out)
    for it in range(n_iters):
        c = torch.softmax(b, dim=2)
        s = torch.einsum("bij,bijo->bjo", c, u_hat)
        v = squash(s, dim=-1)
        agree = torch.einsum("bijo,bjo->bij", u_hat, v)
        cn = c.numpy()
        steps.append({
            "iter": it + 1,
            "mean_abs_dev_from_uniform": float(np.abs(cn - 1.0 / cn.shape[2]).mean()),
            "input_dependence": input_dependence(cn),
            "mean_abs_b": float(b.abs().mean()),
            "mean_abs_agreement_increment": float(agree.abs().mean()),
        })
        b = b + agree
    return {
        "u_hat_norm_mean": float(u_hat.norm(dim=-1).mean()),
        "u_norm_mean": float(u.norm(dim=-1).mean()),
        "W_abs_mean": float(W.abs().mean()),
        "v_norm_mean": float(v.norm(dim=-1).mean()),
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="reports_phase3_final/checkpoints")
    ap.add_argument("--config", default="config/phase3.json")
    ap.add_argument("--n-peptides", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--out", default="reports_routing_iters.json")
    args = ap.parse_args()

    cfg = load_config(_resolve(args.config))
    dp = DataProvider(_resolve(cfg["test_txt"]), _resolve(cfg["pseudo_csv"]))
    idx = np.arange(min(args.n_peptides, len(dp.df)))

    ckpt_dir = _resolve(args.ckpt_dir)
    paths = sorted(p for p in os.listdir(ckpt_dir) if p.endswith(".pt"))
    out = {"args": vars(args), "trained_routing_iters": cfg.get("routing_iters", 3),
           "folds": []}

    for p in paths:
        sd = torch.load(os.path.join(ckpt_dir, p), map_location=DEVICE)
        arch = dict(cfg)
        arch["core_conditioning"] = "core.cond_gate" in sd
        m = build_from_config(arch).to(DEVICE)
        m.load_state_dict(sd)
        m.eval()

        r = replay(m, dp, idx, args.batch_size, args.max_iters)
        r["ckpt"] = p
        out["folds"].append(r)
        print(f"\n{p}  |W| {r['W_abs_mean']:.4f}  ||u|| {r['u_norm_mean']:.4f}  "
              f"||u_hat|| {r['u_hat_norm_mean']:.4f}  ||v|| {r['v_norm_mean']:.4f}")
        print("   iter  |c-1/3|   input-dep   mean|b|   mean|Δb|")
        for s in r["steps"]:
            print(f"   {s['iter']:4d}  {s['mean_abs_dev_from_uniform']:7.5f}  "
                  f"{100*s['input_dependence']:8.1f}%  {s['mean_abs_b']:7.4f}  "
                  f"{s['mean_abs_agreement_increment']:8.5f}")

    path = _resolve(args.out)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
