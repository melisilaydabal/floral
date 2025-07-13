import sys
import os
import argparse
from src.core.experiment import run_experiment
from src.utils.utils import set_seed, load_config
import wandb
import torch
import numpy as np
import random
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

sys.path.append('./floral/')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

def parse_args():
    '''
    Parse command line arguments & load configuration from YAML file.
    '''
    parser = argparse.ArgumentParser(description="FLORAL: Adversarial Training Against Label Poisoning")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='YAML configuration file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    with open(os.path.join('workbench/configs', args.config), 'r') as f:
        conf = yaml.load(f, Loader=Loader)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, conf

def main():
    args, config = parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    wandb.login()
    run = wandb.init(
        project='floral',
        config=config,
        name=f"{config['data']['dataset']}_model_{config['model']['name']}_seed_{args.seed}"
    )

    run_experiment(args, config, args.seed)

    run.finish()


if __name__ == "__main__":
    main()
