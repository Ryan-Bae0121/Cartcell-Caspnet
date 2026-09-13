"""Per-fold resume helpers shared by train_phase2.py / train_phase3.py.

CPU training here runs for hours across many folds/epochs, and this machine
has already been observed to die mid-run once (sleep/shutdown killed the
Phase 3 training process partway through fold 3, epoch 17/40, with no error
in the log -- the process just stopped). Without per-fold checkpointing that
means losing every fold, including ones that had already finished all their
epochs.

So each fold's result (OOF predictions, ensemble-test predictions, best-epoch
report) is written to disk right after that fold finishes. On the next run,
``fold_is_done`` lets the training loop skip any fold whose result is already
on disk and just reuse it, instead of re-training from scratch.
"""

import json
import os

import pandas as pd


def fold_paths(out_dir, ckpt_dir, prefix, fold):
    return {
        "ckpt": os.path.join(ckpt_dir, f"{prefix}_fold{fold}.pt"),
        "oof": os.path.join(out_dir, f"oof_fold{fold}.tsv"),
        "test": os.path.join(out_dir, f"test_fold{fold}.tsv"),
        "report": os.path.join(out_dir, f"fold_report_{fold}.json"),
    }


def fold_is_done(paths):
    """True only if every artifact for the fold exists -- in particular the
    OOF/test/report files, which are only written after a fold's training
    loop runs to completion. A checkpoint alone (e.g. from a fold killed
    mid-training) is NOT enough to count as done."""
    return all(os.path.exists(p) for p in paths.values())


def load_fold_result(paths):
    oof = pd.read_csv(paths["oof"], sep="\t")
    test = pd.read_csv(paths["test"], sep="\t")
    with open(paths["report"]) as fh:
        report = json.load(fh)
    return report, oof, test["cls_prob"].to_numpy(), test["reg_yhat"].to_numpy()


def save_fold_result(paths, report, oof_df, test_prob, test_yhat):
    oof_df.to_csv(paths["oof"], sep="\t", index=False)
    pd.DataFrame({"cls_prob": test_prob, "reg_yhat": test_yhat}).to_csv(
        paths["test"], sep="\t", index=False)
    with open(paths["report"], "w") as fh:
        json.dump(report, fh, indent=2)
