import os
from src.utils.utils import ensure_dir


def setup_experiment_folder(config, seed):
    exp_name = f"{config['data']['dataset']}_model{config['model']['name']}_seed{seed}"
    exp_dir = os.path.join(config['dump']['dir_dump'], exp_name)
    ensure_dir(exp_dir)
    print(f"[Logger] Created experiment folder: {exp_dir}")
    return exp_dir
