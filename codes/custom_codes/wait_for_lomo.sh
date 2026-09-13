#!/usr/bin/env bash
# Polls logs/lomo.status (written by run_lomo.sh) until it finishes (or fails),
# then prints a summary and exits -- lets the background-task tracker notify on
# real completion instead of just the launch. ~50h+ expected, so poll slowly.
cd "$(dirname "$0")"
for i in $(seq 1 1000); do   # 1000 x 300s = up to ~83h safety cap
  if [ -f logs/lomo.status ]; then
    st=$(cat logs/lomo.status)
    echo "LOMO finished with status: $st"
    echo "--- logs/lomo_train.log (tail) ---"
    tail -60 logs/lomo_train.log
    if [ "$st" = "DONE" ]; then exit 0; else exit 1; fi
  fi
  sleep 300
done
echo "TIMEOUT waiting for LOMO to finish (83h cap reached)"
exit 2
