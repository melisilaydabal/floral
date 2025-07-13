# utils/metrics.py
import os
import torch
import numpy as np
from torchmetrics.classification import Accuracy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, hinge_loss, accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dump_metrics(config, dir_dump, dir_exp_folder, experiment_params, avg_loss, avg_acc, perf_type, round):
    dump_file = f"{dir_exp_folder}/_{perf_type}_perf_rounds{config.get('game').get('num_rounds')}" \
                f"_{config.get('data').get('dataset')}" \
                f"_model{config.get('model').get('name')}" \
                f"_epoch{config.get('training').get('num_epochs')}" \
                f"_lr{config.get('optim').get('lr')}" \
                f"_lr{config.get('training').get('batch_size')}" \
                f"_budget{config.get('player_flip').get('flip_budget')}" \
                f"_budget{config.get('player_flip').get('flip_set_budget')}.out"

    if round == 0:
        # with open(os.path.join(dir_dump, dump_file), "w") as file:
        with open(dump_file, "w") as file:
            file.write(f"#{perf_type} Performance\n#Experiment Parameters:\n")
            for key, value in experiment_params.items():
                for nested_key, nested_value in experiment_params[key].items():
                    file.write(f"#{key}: {nested_key}={nested_value}\n")
            file.write("######\n")
            file.write("Round Loss Accuracy\n")
            file.write(f"{round} {avg_loss} {avg_acc}\n")
    else:
        # with open(os.path.join(dir_dump, dump_file), "a") as file:
        with open(dump_file, "a") as file:
            file.write(f"{round} {avg_loss} {avg_acc}\n")


def calculate_accuracy(y_true, y_pred, num_classes):
    copy_y_true = y_true.detach().clone()
    copy_y_pred = y_pred.detach().clone()
    # Accuracy measure expects (0,1) type of class, not (-1,1)
    copy_y_true[copy_y_true == -1] = 0
    copy_y_pred[copy_y_pred == -1] = 0

    if num_classes > 2:
        accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    else:
        accuracy = Accuracy(task="binary", num_classes=num_classes).to(device)
    return accuracy(copy_y_pred, copy_y_true)

# with scikit-learn:
def calculate_accuracy_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_regression_metrics(y_true, y_pred):
    mse = calculate_mse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    return mse, mae, r2

def calculate_hinge_loss(y_true, y_pred):
    return hinge_loss(y_true, y_pred)
def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true)

def calculate_precision_recall(y_true, y_pred):
    return precision_recall_curve(y_true, y_pred)


def dump_general_metric(config, dir_dump, dir_exp_folder, experiment_params, list_metric, metric_type):
    if config.get('training').get('is_perturbed'):
        dump_file = f"{dir_exp_folder}/{metric_type}" \
                    f"_{config.get('data').get('dataset')}.out"
    else:
        dump_file = f"{dir_exp_folder}/{metric_type}" \
                    f"_{config.get('data').get('dataset')}.out"

    with open(os.path.join(dump_file), "w") as file:
        file.write(f"#{metric_type} Metric\n#Experiment Parameters:\n")
        for key, value in experiment_params.items():
            for nested_key, nested_value in experiment_params[key].items():
                file.write(f"#{key}: {nested_key}={nested_value}\n")
        file.write("######\n")
        if metric_type not in ['flip_history']:
            file.write(f"Epoch {metric_type}\n")
            for i in range(len(list_metric)):
                file.write(f"{i+1} {list_metric[i]}\n")
        else:
            file.write(f"Round num_flips {metric_type}\n")
            if config.get('player_flip').get('is_debug_flip') and config.get('player_flip').get('is_set_flip'):
                for round, metric_value in enumerate(list_metric, start=1):
                    for metric_value_item in metric_value:
                        file.write(f"{round} {np.count_nonzero(metric_value_item)} {metric_value_item}\n")
            else:
                for round, metric_value in enumerate(list_metric, start=1):
                    file.write(f"{round} {np.count_nonzero(metric_value)} {metric_value}\n")
