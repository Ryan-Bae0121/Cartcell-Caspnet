#!/usr/bin/env bash
# GPU 5-fold rebuild -- the pre-registered protocol in docs/project_status.md 5.9.
#
# Three models x 5 folds x 600 epochs on one RTX 4090, bs=32, AUPRC selection.
# Phase3 and no-routing warm-start from THIS run's Phase2 fold checkpoints, so
# the order matters and the stages run sequentially.
#
# Measured per-epoch: Phase2 4.0s, Phase3 6.9s, no-routing 5.75s
#   -> 3.3h + 5.8h + 4.8h = ~13.9h total.
#
# Resumable at fold level (resume_utils.py): a fold whose
# oof_fold{k}.tsv / test_fold{k}.tsv / fold_report_{k}.json all exist is reused,
# so a killed run re-trains only the unfinished folds. There is no epoch-level
# resume -- at 46 min/fold the worst case loss is one fold, deliberately.
#
# Before running: powercfg standby/hibernate/monitor timeouts must be 0 on AC,
# or Windows Modern Standby silently stalls training (see section 8 환경).
set -u
cd "$(dirname "$0")"
mkdir -p logs

run() {
  local tag="$1" script="$2" config="$3"
  local mark="logs/gpu600_${tag}.status"
  if [ -f "$mark" ] && [ "$(cat "$mark")" = "DONE" ]; then
    echo "[$(date)] ${tag} already DONE -- skipping stage"
    return 0
  fi
  echo ""
  echo "=== [$(date)] ${tag} | 5 folds x 600 epochs | ${config} ==="
  TRACE=1 PHASE2_CONFIG="$config" PHASE3_CONFIG="$config" \
    python -u "$script" 2>&1 | tee -a "logs/gpu600_${tag}.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "DONE" > "$mark"
    echo "[$(date)] ${tag} DONE"
  else
    echo "FAILED rc=${rc}" > "$mark"
    echo "[$(date)] ${tag} FAILED rc=${rc}"
    return "$rc"
  fi
}

run phase2    train_phase2.py config/phase2_gpu600.json
run phase3    train_phase3.py config/phase3_gpu600.json
run norouting train_phase3.py config/norouting_gpu600.json

echo ""
echo "=== [$(date)] all three stages DONE ==="
ls -1 reports_*_gpu600/*summary*.json reports_*_gpu600/test_metrics.json 2>/dev/null
