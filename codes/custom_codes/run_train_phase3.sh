#!/usr/bin/env bash
# Phase 3: 5-fold CV training on train_data.txt, warm-started per-fold from
# reports_phase2_affinity_from_scratch/checkpoints/phase2_fold{k}.pt.
# Smoke test:  EPOCHS=2 N_FOLDS=2 bash run_train_phase3.sh
# From scratch (no warm start): WARM_START=0 bash run_train_phase3.sh
set -e
cd "$(dirname "$0")"
python train_phase3.py "$@"
