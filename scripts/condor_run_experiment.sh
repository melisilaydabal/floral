#!/bin/bash
source ~/.bashrc          # or ~/.zshrc depending on shell
conda activate floral-venv # or any name of your conda env

CONFIG_PATH=$1
SEED=$2

echo "Running with config: $CONFIG_PATH and seed: $SEED"
python3 main.py --config $CONFIG_PATH --seed $SEED
