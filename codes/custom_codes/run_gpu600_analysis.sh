#!/usr/bin/env bash
# Post-run analysis for the GPU 600-epoch 5-fold run (docs/project_status.md 5.9).
#
# Run this only after run_gpu600.sh has finished all three stages. It is CPU/GPU
# light compared with training and is safe to re-run: every step overwrites its
# own output directory.
#
#   1. register_shift.py per model   -> variant_predictions.tsv + register_shift_summary.json
#                                       (same_core_rate, mean_core_drift = core tracking)
#   2. plot_core_calibration.py      -> core_calibration_baselines.json
#                                       (chance A/B, and measured/chance ratios --
#                                        the numbers section 4.1 is waiting on)
#   3. aggregate_gpu600.py           -> reports_gpu600_aggregate.{json,md}
#                                       (3 rules x 3 models on test, paired
#                                        bootstrap CI 1000, per-fold values)
#
# register_shift.py loads every file in the checkpoint dir ending in ".pt", so it
# picks up exactly the 5 PRIMARY (AUPRC) fold checkpoints and ignores the
# ".pt.composite" / ".pt.rmax" files. Core tracking therefore comes from the same
# checkpoints as the primary performance row, which is what 5.9 requires.
set -u
cd "$(dirname "$0")"
mkdir -p logs

for tag in phase2 phase3 norouting; do
  if [ ! -f "logs/gpu600_${tag}.status" ] || \
     [ "$(cat "logs/gpu600_${tag}.status")" != "DONE" ]; then
    echo "ERROR: stage ${tag} is not DONE -- run_gpu600.sh has not finished."
    echo "       refusing to analyse a partial run."
    exit 1
  fi
done

LOG=logs/gpu600_analysis.log
echo "=== [$(date)] post-run analysis start ===" | tee -a "$LOG"

shift_one() {
  local tag="$1" model="$2" cfg="$3" ckpt="$4"
  local out="reports_register_shift_${tag}_gpu600"
  echo "" | tee -a "$LOG"
  echo "--- [$(date)] register_shift: ${tag} ---" | tee -a "$LOG"
  python -u register_shift.py --model "$model" --config "$cfg" \
      --ckpt-dir "$ckpt" --out-dir "$out" 2>&1 | tee -a "$LOG"
  echo "--- [$(date)] core calibration: ${tag} ---" | tee -a "$LOG"
  python -u plot_core_calibration.py --run "$out" 2>&1 | tee -a "$LOG"
}

shift_one phase2    phase2 config/phase2_gpu600.json \
          reports_phase2_gpu600/checkpoints
shift_one phase3    phase3 config/phase3_gpu600.json \
          reports_phase3_gpu600/checkpoints
shift_one norouting phase3 config/norouting_gpu600.json \
          reports_phase3_norouting_gpu600/checkpoints

echo "" | tee -a "$LOG"
echo "--- [$(date)] performance aggregate (bootstrap 1000) ---" | tee -a "$LOG"
python -u aggregate_gpu600.py --n-boot 1000 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== [$(date)] analysis DONE ===" | tee -a "$LOG"
echo "core tracking + chance levels:" | tee -a "$LOG"
for tag in phase2 phase3 norouting; do
  f="reports_register_shift_${tag}_gpu600/core_calibration_baselines.json"
  [ -f "$f" ] && echo "  $f" | tee -a "$LOG"
done
echo "performance: reports_gpu600_aggregate.md" | tee -a "$LOG"
