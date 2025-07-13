# data/dataset.py
import copy
import sys
import os
import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.datasets import make_classification, make_blobs, make_moons
from src.utils.plotter import _create_fig, plot_perf, _plot_posterior_style, _plot_prior_style

repl_perturb = 1

sys.path.append('.')

LIST_BENCHMARK_DATASETS = ['imdb']
LIST_DATASETS_W_ADV = ['imdb']

if torch.cuda.is_available():
        device = torch.device('cuda:0')
else:
    device = torch.device('cpu') # don't have GPU

def generate_synthetic_data(dataset, num_points=100, test_size=0.2, validation_size=0.2, random_state=999, dir_dump_data='', is_perturbed=False):
    X, y = make_classification(
        n_samples=num_points,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0,
        class_sep=1,
        random_state=random_state
    )
    rng = np.random.RandomState(2)
    X += 1 * rng.uniform(size=X.shape)
    # X, y = make_moons(
    #     n_samples=num_points,
    #     noise=0.2,
    #     random_state=random_state
    # )

    X_adv_5, y_adv_5 = make_classification(
        n_samples=num_points,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.05,
        class_sep=1,
        random_state=random_state
    )
    rng = np.random.RandomState(2)
    X_adv_5 += 1 * rng.uniform(size=X_adv_5.shape)
    # X_adv_5, y_adv_5 = make_moons(
    #     n_samples=num_points,
    #     noise=0.25,
    #     random_state=random_state
    # )

    X_adv_10, y_adv_10 = make_classification(
        n_samples=num_points,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.10,
        class_sep=1,
        random_state=random_state
    )
    rng = np.random.RandomState(2)
    X_adv_10 += 1 * rng.uniform(size=X_adv_10.shape)
    # X_adv_10, y_adv_10 = make_moons(
    #     n_samples=num_points,
    #     noise=0.3,
    #     random_state=random_state
    # )

    X_adv_25, y_adv_25 = make_classification(
        n_samples=num_points,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.25,
        class_sep=1,
        random_state=random_state
    )
    rng = np.random.RandomState(2)
    X_adv_25 += 1 * rng.uniform(size=X_adv_25.shape)
    # X_adv_25, y_adv_25 = make_moons(
    #     n_samples=num_points,
    #     noise=0.35,
    #     random_state=random_state
    # )

    indices = np.arange(num_points)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size,
                                                      random_state=random_state)

    X_train_adv_5, X_test_adv_5, y_train_adv_5, y_test_adv_5 = train_test_split(X_adv_5, y_adv_5, test_size=test_size, random_state=random_state)
    X_train_adv_5, X_val_adv_5, y_train_adv_5, y_val_adv_5 = train_test_split(X_train_adv_5, y_train_adv_5, test_size=validation_size,
                                                      random_state=random_state)

    X_train_adv_10, X_test_adv_10, y_train_adv_10, y_test_adv_10 = train_test_split(X_adv_10, y_adv_10, test_size=test_size, random_state=random_state)
    X_train_adv_10, X_val_adv_10, y_train_adv_10, y_val_adv_10 = train_test_split(X_train_adv_10, y_train_adv_10, test_size=validation_size,
                                                      random_state=random_state)
    X_train_adv_25, X_test_adv_25, y_train_adv_25, y_test_adv_25 = train_test_split(X_adv_25, y_adv_25, test_size=test_size, random_state=random_state)
    X_train_adv_25, X_val_adv_25, y_train_adv_25, y_val_adv_25 = train_test_split(X_train_adv_25, y_train_adv_25, test_size=validation_size,
                                                      random_state=random_state)

    columns = [f"feature_{i}" for i in range(X.shape[1])]

    main_df = pd.DataFrame(data=X, columns=columns)
    main_df["label"] = y

    columns = [f"feature_{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(data=X_train, columns=columns)
    train_df["label"] = y_train

    val_df = pd.DataFrame(data=X_val, columns=columns)
    val_df["label"] = y_val

    test_df = pd.DataFrame(data=X_test, columns=columns)
    test_df["label"] = y_test

    train_adv_5_df = pd.DataFrame(data=X_train_adv_5, columns=columns)
    train_adv_5_df["label"] = y_train_adv_5
    train_adv_10_df = pd.DataFrame(data=X_train_adv_10, columns=columns)
    train_adv_10_df["label"] = y_train_adv_10
    train_adv_25_df = pd.DataFrame(data=X_train_adv_25, columns=columns)
    train_adv_25_df["label"] = y_train_adv_25

    test_adv_5_df = pd.DataFrame(data=X_test_adv_5, columns=columns)
    test_adv_5_df["label"] = y_test_adv_5
    test_adv_10_df = pd.DataFrame(data=X_test_adv_10, columns=columns)
    test_adv_10_df["label"] = y_test_adv_10
    test_adv_25_df = pd.DataFrame(data=X_test_adv_25, columns=columns)
    test_adv_25_df["label"] = y_test_adv_25

    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)
    ax.clear()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap)
    _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
    fig.suptitle(f'Synthetic Train Dataset', fontsize=40, y=1)
    plot_path = os.path.join(dir_dump_data, f'{dataset}_train_dataset_plot.png')
    fig.savefig(plot_path)
    plt.close(fig)

    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)
    ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cmap)
    _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
    fig.suptitle(f'Synthetic Validation Dataset', fontsize=40, y=1)
    plot_path = os.path.join(dir_dump_data, f'{dataset}_validation_dataset_plot.png')
    fig.savefig(plot_path)
    plt.close(fig)

    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap)
    _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
    fig.suptitle(f'Synthetic Test Dataset', fontsize=40, y=1)
    plot_path = os.path.join(dir_dump_data, f'{dataset}_test_dataset_plot.png')
    fig.savefig(plot_path)
    plt.close(fig)

    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)
    ax.clear()
    ax.scatter(X_train_adv_25[:, 0], X_train_adv_25[:, 1], c=y_train_adv_25, cmap=cmap)
    _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
    fig.suptitle(f'Synthetic Train_adv_25 Dataset', fontsize=40, y=1)
    plot_path = os.path.join(dir_dump_data, f'{dataset}_train_adv_25_dataset_plot.png')
    fig.savefig(plot_path)
    plt.close(fig)

    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)
    ax.clear()
    ax.scatter(X_test_adv_25[:, 0], X_test_adv_25[:, 1], c=y_test_adv_25, cmap=cmap)
    _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
    fig.suptitle(f'Synthetic Test_adv_25 Dataset', fontsize=40, y=1)
    plot_path = os.path.join(dir_dump_data, f'{dataset}_test_adv_25_dataset_plot.png')
    fig.savefig(plot_path)
    plt.close(fig)


    with pd.ExcelWriter(os.path.join(dir_dump_data, f'{dataset}_dataset.xlsx')) as writer:
        main_df.to_excel(writer, sheet_name='main', index=False)
        train_df.to_excel(writer, sheet_name='train', index=False)
        val_df.to_excel(writer, sheet_name='validation', index=False)
        test_df.to_excel(writer, sheet_name='test', index=False)
        train_adv_5_df.to_excel(writer, sheet_name='train_adv_5', index=False)
        train_adv_10_df.to_excel(writer, sheet_name='train_adv_10', index=False)
        train_adv_25_df.to_excel(writer, sheet_name='train_adv_25', index=False)
        test_adv_5_df.to_excel(writer, sheet_name='test_adv_5', index=False)
        test_adv_10_df.to_excel(writer, sheet_name='test_adv_10', index=False)
        test_adv_25_df.to_excel(writer, sheet_name='test_adv_25', index=False)


def plot_data(X_train, y_train, X_val, y_val, X_test, y_test, dataset, dir_dump_data):

    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)
    ax.clear()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap)
    _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
    fig.suptitle(f'Synthetic Train Dataset', fontsize=40, y=1)
    plot_path = os.path.join(dir_dump_data, f'{dataset}_train_dataset_plot.png')
    fig.savefig(plot_path)
    plt.close(fig)

    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)
    ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cmap)
    _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
    fig.suptitle(f'Synthetic Validation Dataset', fontsize=40, y=1)
    plot_path = os.path.join(dir_dump_data, f'{dataset}_validation_dataset_plot.png')
    fig.savefig(plot_path)
    plt.close(fig)

    fig, ax = _create_fig()
    cmap = _plot_prior_style(fig, ax)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap)
    _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")
    fig.suptitle(f'Synthetic Test Dataset', fontsize=40, y=1)
    plot_path = os.path.join(dir_dump_data, f'{dataset}_test_dataset_plot.png')
    fig.savefig(plot_path)
    plt.close(fig)


def load_data(args, dataset, in_dim, dir_dump_data='', is_perturbed=False):
    if dataset not in LIST_BENCHMARK_DATASETS:
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

        if dataset == 'mnist_1vs7':
            features = [f'pixel_{i}' for i in range(main_df.shape[1] - 1)]
        else:
            features = [f'feature_{i}' for i in range(main_df.shape[1]-1)]

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
            print(f'Suggested kernel gamma scale value {1 / (in_dim * torch.var(X_train))}')

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


def load_adv_train_data(args, dataset, dir_dump_data=''):
    if dataset in ['imdb']:
        train_dataset_adv_5 = torch.load(os.path.join(dir_dump_data, f'train_adv_flip0.05_dataset.pt'))
        train_dataset_adv_10 = torch.load(os.path.join(dir_dump_data, f'train_adv_flip0.1_dataset.pt'))
        train_dataset_adv_25 = torch.load(os.path.join(dir_dump_data, f'train_adv_flip0.25_dataset.pt'))

        def extract_features_and_labels(dataset):
            X = [x for x in dataset['input_ids']]
            y = [y for y in dataset['label']]
            X = torch.vstack(X)
            y = torch.stack(y)
            return X.type(torch.float32), y.double()

        X_train_adv_5, y_train_adv_5 = extract_features_and_labels(train_dataset_adv_5)
        X_train_adv_10, y_train_adv_10 = extract_features_and_labels(train_dataset_adv_10)
        X_train_adv_25, y_train_adv_25 = extract_features_and_labels(train_dataset_adv_25)
        return y_train_adv_5, y_train_adv_10, y_train_adv_25
    else:
        excel_path = os.path.join(dir_dump_data, f'{dataset}_dataset.xlsx')
        train_adv_5_df = pd.read_excel(excel_path, sheet_name='train_adv_5')
        train_adv_10_df = pd.read_excel(excel_path, sheet_name='train_adv_10')
        train_adv_25_df = pd.read_excel(excel_path, sheet_name='train_adv_25')

        y_train_adv_5 = torch.tensor(train_adv_5_df['label'].values, dtype=torch.long)
        y_train_adv_10 = torch.tensor(train_adv_10_df['label'].values, dtype=torch.long)
        y_train_adv_25 = torch.tensor(train_adv_25_df['label'].values, dtype=torch.long)

        return y_train_adv_5, y_train_adv_10, y_train_adv_25


def load_adv_test_data(dataset, dir_dump_data=''):
    if dataset in ['imdb']:
        test_dataset_adv_5 = torch.load(os.path.join(dir_dump_data, f'test_adv_flip0.05_dataset.pt'))
        test_dataset_adv_10 = torch.load(os.path.join(dir_dump_data, f'test_adv_flip0.1_dataset.pt'))
        test_dataset_adv_25 = torch.load(os.path.join(dir_dump_data, f'test_adv_flip0.25_dataset.pt'))

        def extract_features_and_labels(dataset):
            X = [x for x in dataset['input_ids']]
            y = [y for y in dataset['label']]
            X = torch.vstack(X)
            y = torch.stack(y)
            return X.type(torch.float32), y.double()

        X_test_adv_5, y_test_adv_5 = extract_features_and_labels(test_dataset_adv_5)
        X_test_adv_10, y_test_adv_10 = extract_features_and_labels(test_dataset_adv_10)
        X_test_adv_25, y_test_adv_25 = extract_features_and_labels(test_dataset_adv_25)
        return y_test_adv_5, y_test_adv_10, y_test_adv_25
    else:
        excel_path = os.path.join(dir_dump_data, f'{dataset}_dataset.xlsx')
        test_adv_5_df = pd.read_excel(excel_path, sheet_name='test_adv_5')
        test_adv_10_df = pd.read_excel(excel_path, sheet_name='test_adv_10')
        test_adv_25_df = pd.read_excel(excel_path, sheet_name='test_adv_25')

        y_test_adv_5 = torch.tensor(test_adv_5_df['label'].values, dtype=torch.long)
        y_test_adv_10 = torch.tensor(test_adv_10_df['label'].values, dtype=torch.long)
        y_test_adv_25 = torch.tensor(test_adv_25_df['label'].values, dtype=torch.long)

        return y_test_adv_5, y_test_adv_10, y_test_adv_25


def use_train_adv_data(args, dataset, in_dim, adv_rate, dir_dump_data='', is_perturbed=False):
    if dataset not in LIST_BENCHMARK_DATASETS:
        if dataset in ['mnist_1vs7']:
            excel_path = os.path.join(dir_dump_data, f'{dataset}_seed{args.seed}.xlsx')
        else:
            excel_path = os.path.join(dir_dump_data, f'{dataset}_dataset_seed{args.seed}.xlsx')
        main_df = pd.read_excel(excel_path, sheet_name='main')
        train_df = pd.read_excel(excel_path, sheet_name='train')

        # For NN motivation: use adversarial dataset
        if dataset in ["mnist_1vs7"]:
            assert adv_rate in [5, 10, 15, 20, 25, 30, 35, 40]
            train_df = pd.read_excel(excel_path, sheet_name=f'train_adv_{adv_rate}')

        # For NN motivation: use adversarial dataset --> for moon dataset
        if dataset == "moon":
            assert adv_rate in [5, 10, 15, 20, 25, 30, 35, 40]
            train_df = pd.read_excel(excel_path, sheet_name=f'train_adv_{adv_rate}')


        val_df = pd.read_excel(excel_path, sheet_name='validation')
        test_df = pd.read_excel(excel_path, sheet_name='test')

        if dataset in LIST_DATASETS_W_ADV:
            test_adv_df = pd.read_excel(excel_path, sheet_name='test_adv')

        if dataset in ['mnist_1vs7']:
            features = [f'pixel_{i}' for i in range(main_df.shape[1] - 1)]
        else:
            features = [f'feature_{i}' for i in range(main_df.shape[1]-1)]

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

        X_all = torch.cat((X_train, X_val), dim=0)
        y_all = torch.cat((y_train, y_val), dim=0)
        X_all = torch.cat((X_all, X_test), dim=0)
        y_all = torch.cat((y_all, y_test), dim=0)

    if dataset in LIST_DATASETS_W_ADV:
        return (X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test, y_test_adv)
    return (X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test, [])



def use_alfa_train_adv_data(args, dataset, in_dim, adv_rate, dir_dump_data='', is_perturbed=False):
    if dataset not in LIST_BENCHMARK_DATASETS:

        if dataset == "moon":
            assert adv_rate in [5, 10, 15, 20, 25, 50]
            X_train = torch.load(os.path.join(dir_dump_data, f"{dataset}/alfa_tilt_attack/scikit-synthetic-moon_dataset_features_budget{adv_rate}_seed{args.seed}.pt"))
            y_train = torch.load(os.path.join(dir_dump_data, f"{dataset}/alfa_tilt_attack/scikit-synthetic-moon_dataset_labels_budget{adv_rate}_seed{args.seed}.pt"))

        if dataset == "mnist_1vs7":
            assert adv_rate in [5, 10, 15, 20, 25]
            X_train = torch.load(os.path.join(dir_dump_data, f"mnist-1vs7-alfa_attack/mnist_1vs7_dataset_features_budget{adv_rate}_seed{args.seed}.pt"))
            y_train = torch.load(os.path.join(dir_dump_data, f"mnist-1vs7-alfa_attack/mnist_1vs7_dataset_labels_budget{adv_rate}_seed{args.seed}.pt"))

    return (X_train, y_train)



def split_dataset(X, y, test_size, validation_size, random_state=42):
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size,
                                                      random_state=random_state)

    train_indices = X_train.index
    val_indices = X_val.index
    test_indices = X_test.index

    X_train = torch.from_numpy(X_train.values).float().to(device)
    y_train = torch.from_numpy(y_train.values).float().to(device)
    X_val = torch.from_numpy(X_val.values).float().to(device)
    y_val = torch.from_numpy(y_val.values).float().to(device)
    X_test = torch.from_numpy(X_test.values).float().to(device)
    y_test = torch.from_numpy(y_test.values).float().to(device)

    return (X_train, y_train, train_indices, X_val, y_val, val_indices, X_test, y_test, test_indices)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        sys.exit("usage: python dataset.py <dir_dump_data> <dataset_name>\n")
    else:
        dir_dump_data = sys.argv[1]
        dataset_name = sys.argv[2]

    generate_synthetic_data(dataset=dataset_name, num_points=800, test_size=0.5, validation_size=0.5, random_state=42,
                                dir_dump_data=dir_dump_data, is_perturbed=False)
