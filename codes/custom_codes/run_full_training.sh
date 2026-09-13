#!/usr/bin/env bash
# Full local rebuild training: Phase 2 (5-fold x 30ep) -> Phase 3 (5-fold x 40ep,
# warm-started from the just-trained Phase 2 fold checkpoints). CPU only.
#
# Idempotent / resumable at two levels:
#   * stage level  -- if logs/phase{2,3}.status already says DONE, that whole
#     stage is skipped (Phase 2 finished once already; no need to redo it).
#   * fold level   -- train_phase{2,3}.py itself skips any individual fold
#     whose oof_fold{k}.tsv/test_fold{k}.tsv/fold_report_{k}.json are already
#     on disk (resume_utils.py), so a run killed mid-stage (this happened once:
#     the machine slept/shut down mid Phase-3 fold 3) only re-trains the folds
#     that hadn't finished, not the whole stage.
#
# Logs to logs/phase2_train.log and logs/phase3_train.log (appended across
# resumed runs, not truncated).
set -u
cd "$(dirname "$0")"
mkdir -p logs

STAGE2_MARK="logs/phase2.status"
STAGE3_MARK="logs/phase3.status"

if [ -f "$STAGE2_MARK" ] && [ "$(cat "$STAGE2_MARK")" = "DONE" ]; then
  echo "[$(date)] Phase2 already DONE ($STAGE2_MARK) -- skipping stage" | tee -a logs/phase2_train.log
else
  echo "[$(date)] Phase2 training start (resuming any already-finished folds)" | tee -a logs/phase2_train.log
  python train_phase2.py >> logs/phase2_train.log 2>&1
  if [ $? -eq 0 ]; then
    echo "DONE" > "$STAGE2_MARK"
    echo "[$(date)] Phase2 training DONE" >> logs/phase2_train.log
  else
    echo "FAILED" > "$STAGE2_MARK"
    echo "[$(date)] Phase2 training FAILED" >> logs/phase2_train.log
    exit 1
  fi
fi

if [ -f "$STAGE3_MARK" ] && [ "$(cat "$STAGE3_MARK")" = "DONE" ]; then
  echo "[$(date)] Phase3 already DONE ($STAGE3_MARK) -- skipping stage" | tee -a logs/phase3_train.log
else
  echo "[$(date)] Phase3 training start (warm-start from Phase2 fold checkpoints; resuming any already-finished folds)" | tee -a logs/phase3_train.log
  python train_phase3.py >> logs/phase3_train.log 2>&1
  if [ $? -eq 0 ]; then
    echo "DONE" > "$STAGE3_MARK"
    echo "[$(date)] Phase3 training DONE" >> logs/phase3_train.log
  else
    echo "FAILED" > "$STAGE3_MARK"
    echo "[$(date)] Phase3 training FAILED" >> logs/phase3_train.log
    exit 1
  fi
fi

echo "[$(date)] Full training pipeline DONE" | tee -a logs/phase3_train.log
