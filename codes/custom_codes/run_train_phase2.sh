#!/usr/bin/env bash
# Phase 2: 5-fold CV training on train_data.txt.
# Smoke test:  EPOCHS=2 N_FOLDS=2 bash run_train_phase2.sh
set -e
cd "$(dirname "$0")"
python train_phase2.py "$@"
