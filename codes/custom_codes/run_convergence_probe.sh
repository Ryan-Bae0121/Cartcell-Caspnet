#!/usr/bin/env bash
# Convergence probe -- project_status.md section 9, item 1.
#
# Trains ONE fold of each of the three reported models to 150 epochs instead of
# 50, logging the composite selection score and mean |W| every epoch. It answers
# two questions that the 50-epoch runs cannot:
#
#   * where the composite score flattens  -> the budget the real runs should use
#   * whether |W| is still climbing at 50 -> budget problem vs. no incentive
#
# Those two are entangled: best_epoch sat at 34-49 against a 50-epoch budget in
# every model, and |W| ended 2.4-4.7x its initialisation, so raising the init
# scale now would confound "bigger init" with "more training".
#
# Fold 0 only, with n_folds left at 5 so the split matches the reported runs.
# At the measured 41 s/epoch (median, m2_phase3.log) this is ~1.7 h per capsule
# model, ~5 h in total. The full 5-fold version would be ~25 h per model.
#
# Writes to reports_probe_*/ -- separate from the reported runs, so nothing
# already published is overwritten and resume_utils does not reuse a 50-epoch
# fold as "already complete".
#
#   bash run_convergence_probe.sh              # all three, sequentially
#   PROBE_EPOCHS=10 bash run_convergence_probe.sh   # smoke test
set -e
cd "$(dirname "$0")"

EPOCHS="${PROBE_EPOCHS:-150}"
LOGS="${PROBE_LOG_DIR:-probe_logs}"
mkdir -p "$LOGS"

# Phase3 warm-starts from the 50-epoch Phase2 checkpoints, exactly as the
# reported runs do. The question is the budget of the Phase3 stage; changing the
# warm start as well would make the answer about something else.
run() {
  local tag="$1" script="$2" config="$3"
  echo ""
  echo "=== probe: $tag | $EPOCHS epochs | fold 0 ==="
  TRACE=1 ONLY_FOLDS=0 EPOCHS="$EPOCHS" \
    OUT_DIR="reports_probe_${tag}" PHASE3_CONFIG="$config" PHASE2_CONFIG="$config" \
    python "$script" 2>&1 | tee "$LOGS/probe_${tag}.log"
}

run phase2     train_phase2.py config/phase2.json
run phase3     train_phase3.py config/phase3.json
run norouting  train_phase3.py config/phase3_norouting.json

echo ""
echo "=== done. traces: ==="
ls -1 reports_probe_*/trace/*.jsonl
echo "summarise with: python summarize_probe.py"
