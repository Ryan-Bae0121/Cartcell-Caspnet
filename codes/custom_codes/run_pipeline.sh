#!/usr/bin/env bash
# Regenerate custom_dataset/Anthem_dataset/{train_data.txt,test_data.txt,mhc_ii_pseudo.csv}
set -e
cd "$(dirname "$0")"
python data_pipeline.py "$@"
