import glob
import contextlib
from PIL import Image
import os
import random
import numpy as np
import torch
import yaml
from datetime import datetime
from pathlib import Path

repl_perturb = 1

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dump_loss(config, dir_dump, dir_exp_folder, round, experiment_params, losses, loss_type):
    if config.get('training').get('is_perturbed'):
        dump_file = f"{dir_exp_folder}/round{round}_{loss_type}_loss" \
                    f"_{config.get('data').get('dataset')}" \
                    f"_isperturbed_{config.get('training').get('is_perturbed')}_{repl_perturb}.out"
    else:
        dump_file = f"{dir_exp_folder}/round{round}_{loss_type}_loss" \
                    f"_{config.get('data').get('dataset')}" \
                    f"_isperturbed_{config.get('training').get('is_perturbed')}.out"

    with open(os.path.join(dir_dump, dump_file), "w") as file:
        file.write(f"#{loss_type} Loss\n#Experiment Parameters:\n")
        for key, value in experiment_params.items():
            for nested_key, nested_value in experiment_params[key].items():
                file.write(f"#{key}: {nested_key}={nested_value}\n")
        file.write("######\n")
        file.write("Epoch Loss\n")
        for epoch, loss in enumerate(losses, start=1):
            file.write(f"{epoch} {loss}\n")


def dump_nn_accuracy_loss(config, dir_dump, dir_exp_folder, round, experiment_params, accuracies, losses, loss_type):
    if config.get('training').get('is_perturbed'):
        dump_file = f"{dir_exp_folder}/round{round}_{loss_type}_accuracy_loss" \
                    f"_{config.get('data').get('dataset')}" \
                    f"_isperturbed_{config.get('training').get('is_perturbed')}_{repl_perturb}.out"
    else:
        dump_file = f"{dir_exp_folder}/round{round}_{loss_type}_accuracy_loss" \
                    f"_{config.get('data').get('dataset')}" \
                    f"_isperturbed_{config.get('training').get('is_perturbed')}.out"

    with open(os.path.join(dir_dump, dump_file), "w") as file:
        file.write(f"#{loss_type} Loss\n#Experiment Parameters:\n")
        for key, value in experiment_params.items():
            for nested_key, nested_value in experiment_params[key].items():
                file.write(f"#{key}: {nested_key}={nested_value}\n")
        file.write("######\n")
        file.write("Epoch Accuracy Loss\n")
        for epoch, (accuracy, loss) in enumerate(zip(accuracies, losses), start=1):
            file.write(f"{epoch} {accuracy} {loss}\n")


def dump_accuracy_loss(config, dir_dump, dir_exp_folder, experiment_params, loss, accuracy, perf_type, round):
    if config.get('training').get('is_perturbed'):
        dump_file = f"{dir_exp_folder}/{perf_type}_acc_loss_perf_rounds{config.get('game').get('num_rounds')}" \
                    f"_{config.get('data').get('dataset')}" \
                    f"_isperturbed_{config.get('training').get('is_perturbed')}_{repl_perturb}.out"
    else:
        dump_file = f"{dir_exp_folder}/{perf_type}_acc_loss_perf_rounds{config.get('game').get('num_rounds')}" \
                    f"_{config.get('data').get('dataset')}" \
                    f"_isperturbed_{config.get('training').get('is_perturbed')}.out"

    if round == 0:
        with open(os.path.join(dir_dump, dump_file), "w") as file:
            file.write(f"#{perf_type} Performance\n#Experiment Parameters:\n")
            for key, value in experiment_params.items():
                for nested_key, nested_value in experiment_params[key].items():
                    file.write(f"#{key}: {nested_key}={nested_value}\n")
            file.write("######\n")
            file.write("Round Loss Accuracy \n")
            file.write("{} {:.5f} {:.5f}\n".format(round, loss, accuracy))
    else:
        with open(os.path.join(dir_dump, dump_file), "a") as file:
            file.write("{} {:.5f} {:.5f}\n".format(round, loss, accuracy))

def make_gif(fp_in, fp_out):
    with contextlib.ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))
        img = next(imgs)
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=1000, loop=0)

if __name__ == "__main__":

    gif_name = "SVM_DB_change"
    fp_in = "./workbench/*.png"
    fp_out = f"./workbench/{gif_name}.gif"
    make_gif(fp_in, fp_out)



