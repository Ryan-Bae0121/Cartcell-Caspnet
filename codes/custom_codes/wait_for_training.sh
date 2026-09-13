#!/usr/bin/env bash
# Polls logs/phase{2,3}.status (written by run_full_training.sh) until Phase3
# finishes (or something fails), then prints a summary and exits -- lets the
# background-task tracker notify on real completion instead of just the launch.
cd "$(dirname "$0")"
for i in $(seq 1 300); do   # 300 x 180s = up to 15h safety cap
  if [ -f logs/phase2.status ] && [ "$(cat logs/phase2.status)" = "FAILED" ]; then
    echo "Phase2 training FAILED -- see logs/phase2_train.log"
    tail -40 logs/phase2_train.log
    exit 1
  fi
  if [ -f logs/phase3.status ]; then
    st=$(cat logs/phase3.status)
    echo "Phase3 training finished with status: $st"
    echo "--- logs/phase3_train.log (tail) ---"
    tail -60 logs/phase3_train.log
    if [ "$st" = "DONE" ]; then exit 0; else exit 1; fi
  fi
  sleep 180
done
echo "TIMEOUT waiting for training to finish (15h cap reached)"
exit 2
