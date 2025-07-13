import re
import torch
import os
import numpy as np
import argparse
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.colors
from matplotlib.lines import Line2D

plt.style.use('fast')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

LIST_BENCHMARK_DATASETS = ['cifar10', 'jigsaw_toxic', 'imdb']
LIST_DATASETS_W_ADV = ['adult', 'imdb']

if torch.cuda.is_available():
        device = torch.device('cuda:0')
else:
    device = torch.device('cpu') # don't have GPU

def parse_flip_history(file_path):
    experiment_params = {}
    flip_history = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse experiment parameters (lines that start with '#')
    param_section = True
    for line in lines:
        if line.startswith("######"):
            param_section = False
            continue

        if param_section and line.startswith("#"):
            continue

        elif not param_section and line.strip():
            # Parse flip history entries
            match = re.match(r"(\d+)\s+(\d+)\s+\[(.*)\]", line.strip())
            if match:
                round_num = int(match.group(1))
                num_flips = int(match.group(2))
                flip_list = list(map(int, match.group(3).split(',')))
                flip_history.append({
                    "round": round_num,
                    "num_flips": num_flips,
                    "flip_list": flip_list
                })
    return flip_history


def _create_fig(width=15, height=15):
    """Create a figure and axis with the specified size."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    return fig, ax

def _plot_posterior_style(fig, ax, x_label, y_label):
    """Apply the posterior style to the plot."""
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    ax.set_xlabel(x_label, fontsize=50)
    ax.set_ylabel(y_label, fontsize=50)
    ax.grid(True)
    fig.tight_layout()

def _plot_prior_style(fig, ax):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#0023FF', '#CB0003', '#FFA200',
                                                                    '#13B000', '#F500DF', '#E2D402', '#72EEFF',
                                                                    '#FD74C3', '#7BA0E9', '#1f78b4', '#F97511',
                                                                    '#dbdb40', '#37E593', '#3CEAF6', '#9D0505',
                                                                    '#f781bf', '#a65628', '#984ea3', '#4daf4a',
                                                                    '#ff7f00', '#377eb8', '#e41a1c'])
    return cmap

def plot_data(args, X_train, y_train, dataset, dir_dump_data):
    # Plot the generated dataset
    fig, ax = _create_fig(16, 10)
    cmap = _plot_prior_style(fig, ax)
    ax.clear()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, marker='o', s=150)
    _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
    legendElements = [
        Line2D([0], [0], linestyle='none', marker='o', color='blue', markersize=7),
        Line2D([0], [0], linestyle='none', marker='o', color='red', markersize=7)
    ]
    myLegend = plt.legend(legendElements,
                          ['Negative -1', 'Positive +1'],
                          fontsize="15", loc='upper right', facecolor='white', framealpha=1)
    myLegend.get_frame().set_linewidth(3)

    plot_path = os.path.join(dir_dump_data, f'{dataset}_train_dataset_plot_seed{args.seed}.png')
    fig.savefig(plot_path)
    plot_path = os.path.join(dir_dump_data, f'{dataset}_train_dataset_plot_seed{args.seed}.pdf')
    fig.savefig(plot_path)
    plt.close(fig)


def plot_data_w_recovered(args, X_train, y_train, y_train_adv, poisoned_indices, flip_indices_per_round, round, dataset, dir_dump_data):
    # Plot the generated dataset
    fig, ax = _create_fig(16, 10)
    cmap = _plot_prior_style(fig, ax)
    ax.clear()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, marker='o', s=150)


    ax.scatter(X_train[poisoned_indices, 0], X_train[poisoned_indices, 1],
               color= 'k', marker = 'x', s = 60, alpha = 1, edgecolors='k')
    ax.scatter(X_train[flip_indices_per_round, 0], X_train[flip_indices_per_round, 1],
               facecolors='y', edgecolors='y', s=20)

    _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
    legendElements = [
        Line2D([0], [0], linestyle='none', marker='o', color='blue', markersize=7),
        Line2D([0], [0], linestyle='none', marker='o', color='red', markersize=7),
        Line2D([0], [0], linestyle='none', marker='x', color='k', markersize=7, markeredgecolor='k'),
        Line2D([0], [0], linestyle='none', marker='o', color='y', markersize=7, markeredgecolor='y')
    ]
    myLegend = plt.legend(legendElements,
                          ['Negative -1', 'Positive +1', 'Poisoned label', 'Label Identified by FLORAL'],
                          fontsize="15", loc='upper right', facecolor='white', framealpha=1)
    myLegend.get_frame().set_linewidth(3)

    plot_path = os.path.join(dir_dump_data, f'{dataset}_train_dataset_recovered_pts_round{round}_plot_seed{args.seed}.png')
    fig.savefig(plot_path)
    plot_path = os.path.join(dir_dump_data, f'{dataset}_train_dataset_recovered_pts_round{round}_plot_seed{args.seed}.pdf')
    fig.savefig(plot_path)
    plt.close(fig)


def load_data(args, dataset, in_dim, dir_dump_data='', is_perturbed=False, is_pca=False):
    if dataset not in LIST_BENCHMARK_DATASETS:

        # Read the Excel file
        if is_pca:
            excel_path = os.path.join(dir_dump_data, f'pca_{dataset}_dataset_seed{args.seed}.xlsx')
        else:
            if dataset == 'mnist_1vs7':
                excel_path = os.path.join(dir_dump_data, f'{dataset}_seed{args.seed}.xlsx')
            else:
                excel_path = os.path.join(dir_dump_data, f'{dataset}_dataset_seed{args.seed}.xlsx')
        main_df = pd.read_excel(excel_path, sheet_name='main')
        train_df = pd.read_excel(excel_path, sheet_name='train')

        val_df = pd.read_excel(excel_path, sheet_name='validation')
        test_df = pd.read_excel(excel_path, sheet_name='test')

        if dataset in LIST_DATASETS_W_ADV:
            test_adv_df = pd.read_excel(excel_path, sheet_name='test_adv')

        # Extract features and labels from DataFrames
        if dataset == 'mnist_1vs7':
            features = [f'pixel_{i}' for i in range(main_df.shape[1] - 1)]
        else:
            features = [f'feature_{i}' for i in range(main_df.shape[1]-1)]
            train_df = train_df.sort_values(by=['feature_0'], ascending=True)

        X_all = torch.tensor(main_df[features].values, dtype=torch.float32)
        y_all = torch.tensor(main_df['label'].values, dtype=torch.long)

        X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
        print(f'Suggested kernel gamma scale value {1/(in_dim*torch.var(X_train))}')

        X_val = torch.tensor(val_df.iloc[:, :-1].values, dtype=torch.float32)
        y_val = torch.tensor(val_df['label'].values, dtype=torch.long)

        X_test = torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float32)
        y_test = torch.tensor(test_df['label'].values, dtype=torch.long)

        if dataset in LIST_DATASETS_W_ADV:
            y_test_adv = torch.tensor(test_adv_df['label'].values, dtype=torch.long)

        if is_perturbed:
            y_train = torch.tensor(train_df[f'corrupted_label_{repl_perturb}'].values, dtype=torch.long)
        else:
            y_train = torch.tensor(train_df['label'].values, dtype=torch.long)

    else:
        if dataset in ['imdb']:
            label_flip_rate = args.imdb_label_flip_rate

            train_dataset = torch.load(os.path.join(dir_dump_data + f'-seed{args.seed}', f'train_embedding_dataset.pt'))
            val_dataset = torch.load(os.path.join(dir_dump_data + f'-seed{args.seed}', f'valid_embedding_dataset.pt'))
            test_dataset = torch.load(os.path.join(dir_dump_data + f'-seed{args.seed}', f'test_embedding_dataset.pt'))
            train_dataset_adv = torch.load(os.path.join(dir_dump_data + f'-seed{args.seed}', f'train_adv_flip{label_flip_rate}_embedding_dataset.pt'))
            val_dataset_adv = torch.load(os.path.join(dir_dump_data + f'-seed{args.seed}', f'valid_adv_flip{label_flip_rate}_embedding_dataset.pt'))
            test_dataset_adv = torch.load(os.path.join(dir_dump_data + f'-seed{args.seed}', f'test_adv_flip{label_flip_rate}_embedding_dataset.pt'))

            def extract_features_and_labels(dataset):
                X = [entry[0] for entry in dataset]
                y = [entry[1] for entry in dataset]
                X = torch.vstack(X)
                y = torch.stack(y)
                return X.type(torch.float32), y.double()

            X_train, y_train = extract_features_and_labels(train_dataset)
            X_val, y_val = extract_features_and_labels(val_dataset)
            X_test, y_test = extract_features_and_labels(test_dataset)
            X_train_adv, y_train_adv = extract_features_and_labels(train_dataset_adv)
            X_val_adv, y_val_adv = extract_features_and_labels(val_dataset_adv)
            X_test_adv, y_test_adv = extract_features_and_labels(test_dataset_adv)

            X_all = torch.vstack((X_train, X_val, X_test))
            y_all =  torch.cat((y_train, y_val, y_test))

        elif dataset in ['mnist_1vs7']:
            excel_path = os.path.join(dir_dump_data, f'{dataset}_seed{args.seed}.xlsx')
            main_df = pd.read_excel(excel_path, sheet_name='main')
            train_df = pd.read_excel(excel_path, sheet_name='train')
            val_df = pd.read_excel(excel_path, sheet_name='validation')
            test_df = pd.read_excel(excel_path, sheet_name='test')

            if dataset in LIST_DATASETS_W_ADV:
                test_adv_df = pd.read_excel(excel_path, sheet_name='test_adv')

            features = [f'pixel_{i}' for i in range(main_df.shape[1] - 1)]

            X_all = torch.tensor(main_df[features].values, dtype=torch.float32)
            y_all = torch.tensor(main_df['label'].values, dtype=torch.long)
            X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
            X_val = torch.tensor(val_df.iloc[:, :-1].values, dtype=torch.float32)
            y_val = torch.tensor(val_df['label'].values, dtype=torch.long)
            X_test = torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float32)
            y_test = torch.tensor(test_df['label'].values, dtype=torch.long)

            if dataset in LIST_DATASETS_W_ADV:
                y_test_adv = torch.tensor(test_adv_df['label'].values, dtype=torch.long)

            if is_perturbed:
                y_train = torch.tensor(train_df[f'corrupted_label_{repl_perturb}'].values, dtype=torch.long)
            else:
                y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
    if dataset in LIST_DATASETS_W_ADV:
        return (X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test, y_test_adv)

    return (X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test, [])


def use_train_adv_data(args, dataset, in_dim, adv_rate, dir_dump_data='', is_perturbed=False, is_pca=False, seed=42):
    if dataset not in LIST_BENCHMARK_DATASETS:

        # Read the Excel file
        if is_pca:
            excel_path = os.path.join(dir_dump_data, f'pca_{dataset}_dataset_seed{args.seed}.xlsx')
        else:
            if dataset in ['mnist_1vs7']:
                excel_path = os.path.join(dir_dump_data, f'{dataset}_seed{seed}.xlsx')
            else:
                excel_path = os.path.join(dir_dump_data, f'{dataset}_dataset_seed{args.seed}.xlsx')
        main_df = pd.read_excel(excel_path, sheet_name='main')
        train_df = pd.read_excel(excel_path, sheet_name='train')

        # For NN motivation: use adversarial dataset --> for moon dataset
        if dataset == "moon":
            assert adv_rate in [5, 10, 25, 30, 35, 40]
            train_df = pd.read_excel(excel_path, sheet_name=f'train_adv_{adv_rate}')
            train_df = train_df.sort_values(by=['feature_0'], ascending=True)

        val_df = pd.read_excel(excel_path, sheet_name='validation')
        test_df = pd.read_excel(excel_path, sheet_name='test')

        if dataset in LIST_DATASETS_W_ADV:
            test_adv_df = pd.read_excel(excel_path, sheet_name='test_adv')

        # Extract features and labels from DataFrames
        if dataset in ['mnist_1vs7']:
            features = [f'pixel_{i}' for i in range(main_df.shape[1] - 1)]
        else:
            features = [f'feature_{i}' for i in range(main_df.shape[1]-1)]
            train_df = train_df.sort_values(by=['feature_0'], ascending=True)

        X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
        X_val = torch.tensor(val_df.iloc[:, :-1].values, dtype=torch.float32)
        y_val = torch.tensor(val_df['label'].values, dtype=torch.long)
        X_test = torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float32)
        y_test = torch.tensor(test_df['label'].values, dtype=torch.long)

        if dataset in LIST_DATASETS_W_ADV:
            y_test_adv = torch.tensor(test_adv_df['label'].values, dtype=torch.long)

        if is_perturbed:
            y_train = torch.tensor(train_df[f'corrupted_label_{repl_perturb}'].values, dtype=torch.long)
        else:
            y_train = torch.tensor(train_df['label'].values, dtype=torch.long)

        X_all = torch.cat((X_train, X_val), dim=0)
        y_all = torch.cat((y_train, y_val), dim=0)
        X_all = torch.cat((X_all, X_test), dim=0)
        y_all = torch.cat((y_all, y_test), dim=0)

    if dataset in LIST_DATASETS_W_ADV:
        return (X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test, y_test_adv)
    return (X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test, [])


def main(args):
    X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test, y_test_adv = load_data(args,
                                                                                         dataset=args.dataset,
                                                                                         in_dim=2,
                                                                                         dir_dump_data=args.dir_dump_data,
                                                                                         is_perturbed=False,
                                                                                         is_pca=False)
    y_all[y_all == 0] = -1
    y_train[y_train == 0] = -1
    y_val[y_val == 0] = -1
    y_test[y_test == 0] = -1

    X_all, y_all, X_train, y_train_adv, X_val, y_val, X_test, y_test, y_test_adv = use_train_adv_data(args,
                                                                                                  dataset=args.dataset,
                                                                                                  in_dim=2,
                                                                                                  adv_rate=args.adv_train_rate,
                                                                                                  dir_dump_data=args.dir_dump_data,
                                                                                                  is_perturbed=False,
                                                                                                  is_pca=False,
                                                                                                  seed=args.seed)
    y_all[y_all == 0] = -1
    y_train_adv[y_train_adv == 0] = -1
    y_val[y_val == 0] = -1
    y_test[y_test == 0] = -1

    poisoned_indices = torch.nonzero(torch.ne(y_train, y_train_adv), as_tuple=True)[0].tolist()
    print(f"Poisoned indices in the {args.adv_train_rate}_adv training dataset: {poisoned_indices}")

    # Example usage:
    flip_history = parse_flip_history(args.flip_hist_file_path)
    flip_indices = []
    count_recovered_indices = []
    round = 1
    for round_data in flip_history:
        flip_indices_per_round = [i for (i,x) in enumerate(round_data['flip_list']) if x == 1]
        flip_indices.append(flip_indices_per_round)
        check_equal_indices = list(set(flip_indices_per_round).intersection(poisoned_indices))
        count_recovered_indices.append(100 * (len(check_equal_indices)/len(poisoned_indices)))
        print(f"Recovered poisoned {len(check_equal_indices)} indices with {100 * (len(check_equal_indices)/len(poisoned_indices))}%")

        if round <= 10:
            plot_data_w_recovered(args, X_train, y_train, y_train_adv, poisoned_indices,
                                  flip_indices_per_round, round, args.dataset, args.dir_dump_plot)
        round += 1
    print(f"Average recovery ratio over training rounds: {100 * (np.mean(count_recovered_indices[:100]) / len(count_recovered_indices[:100])):.4f}%")
    plot_data(args, X_train, y_train, args.dataset, args.dir_dump_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze Flip Recovery for FLORAL's Effectiveness")

    # Add arguments for hyperparameters
    parser.add_argument("--dataset", type=str, default='moon', help='')
    parser.add_argument("--dir_dump_data", type=str, default="./dataset/", help='')
    parser.add_argument("--adv_train_rate", type=int, default=0, help="poisoned label adv_train_rate")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--flip_hist_file_path", type=str, default="PATH_TO_FLIP_HISTORY_OUT_FILE", help='')
    parser.add_argument("--dir_dump_plot", type=str, default="./workbench/", help='')

    # Parse arguments
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)