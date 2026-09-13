#!/usr/bin/env bash
# LOMO: 49 alleles x (Phase2 30ep + Phase3 40ep + alpha=0.5 ensemble eval), CPU.
# ~50h+ expected (measured ~100-120s/epoch-pair on this machine). Resumable at
# allele+model granularity -- lomo.py itself skips any allele already present
# in reports_lomo/lomo_per_allele.csv, so a run killed partway through (this
# machine has already been observed to die mid-run once, likely sleep/shutdown)
# only re-does the alleles that hadn't finished.
set -u
cd "$(dirname "$0")"
mkdir -p logs
MARK="logs/lomo.status"

if [ -f "$MARK" ] && [ "$(cat "$MARK")" = "DONE" ]; then
  echo "[$(date)] LOMO already DONE ($MARK) -- skipping" | tee -a logs/lomo_train.log
  exit 0
fi

echo "[$(date)] LOMO start (resuming any already-finished alleles)" | tee -a logs/lomo_train.log
python lomo.py >> logs/lomo_train.log 2>&1
if [ $? -eq 0 ]; then
  echo "DONE" > "$MARK"
  echo "[$(date)] LOMO DONE" >> logs/lomo_train.log
else
  echo "FAILED" > "$MARK"
  echo "[$(date)] LOMO FAILED" >> logs/lomo_train.log
  exit 1
fi
