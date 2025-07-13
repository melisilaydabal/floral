#!/bin/bash

CONFIG="configs/default.yaml"
SEEDS=(42 )
DATASETS=("moon")

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "Running dataset=${DATASET} seed=${SEED}"
    python3 main.py --config $CONFIG --seed $SEED \
      --override data.dataset=$DATASET
  done
done
