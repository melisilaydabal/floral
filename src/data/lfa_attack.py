import os
import pandas as pd
import numpy as np
import torch
import random
import argparse
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split

SEEDS = [42]
BUDGET_SIZES = [5, 10, 25, 50]

def load_custom_data(seed, dir_dump_data, dataset):

    if dataset in ['mnist_1vs7']:
        excel_path = os.path.join(dir_dump_data, f'{dataset}_seed{seed}.xlsx')
    else:
        excel_path = os.path.join(dir_dump_data, f'{dataset}_dataset_seed{seed}.xlsx')
    main_df = pd.read_excel(excel_path, sheet_name='main')
    train_df = pd.read_excel(excel_path, sheet_name='train')
    val_df = pd.read_excel(excel_path, sheet_name='validation')
    test_df = pd.read_excel(excel_path, sheet_name='test')

    # Convert to numpy arrays
    X_train = train_df.iloc[:, :-1].values  # Features (all columns except the last)
    y_train = train_df.iloc[:, -1].values  # Labels (last column)

    X_val = val_df.iloc[:, :-1].values  # Features (all columns except the last)
    y_val = val_df.iloc[:, -1].values  # Labels (last column)

    X_test = test_df.iloc[:, :-1].values  # Features (all columns except the last)
    y_test = test_df.iloc[:, -1].values  # Labels (last column)

    return X_train, y_train, X_val, y_val, X_test, y_test


def LFA_attack(X_train, y_train, X_val, y_val, p):
    """
    Implement LFA attack on a given dataset using an SVM with RBF kernel.

    Parameters:
        X_train, y_train: Training features and labels
        X_val, y_val: Validation features and labels
        p (int): Number of examples to flip
    Returns:
        torch.Tensor: Poisoned X_train and y_train tensors.
    """
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    val_data = [(x, y) for x, y in zip(X_val, y_val)]
    u = torch.zeros(len(train_data), dtype=torch.bool)  # Flip indicator vector
    S_p = train_data.copy()  # Copy of training data
    I = list(range(len(train_data)))  # Indices of training samples

    p = int(np.ceil(p * X_train.shape[0] / 100))

    for _ in range(p):
        e_values = []

        # Iterate over indices in I
        for j in I:
            # Create a modified dataset S' where the label of sample Ij is flipped
            S_prime = S_p.copy()
            x_j, y_j = S_prime[j]
            S_prime[j] = (x_j, -y_j)  # Flip label

            X_train_prime = torch.stack([torch.tensor(x) for x, _ in S_prime]).numpy()
            y_train_prime = torch.tensor([y for _, y in S_prime]).numpy()

            # Train SVM on S'
            svm = SVC(kernel='rbf')
            svm.fit(X_train_prime, y_train_prime)

            # Calculate validation error
            X_val_tensor = torch.tensor(X_val).float()
            y_val_tensor = torch.tensor(y_val).float()
            y_pred = svm.predict(X_val)
            val_error = 1 - accuracy_score(y_val, y_pred)  # Validation error as classification error
            e_values.append(val_error)

        # Find index with maximum error e(j)
        i_k = I[torch.argmax(torch.tensor(e_values))]

        # Mark flip and update set
        u[i_k] = True  # Mark this index as flipped
        I.remove(i_k)  # Remove this index from I
        x_i_k, y_i_k = S_p[i_k]
        S_p[i_k] = (x_i_k, -y_i_k)  # Flip label in training set

    # Prepare poisoned X_train and y_train for saving
    X_train_poisoned = torch.stack([torch.tensor(x) for x, _ in S_p])
    y_train_poisoned = torch.tensor([y for _, y in S_p])

    return X_train_poisoned.numpy(), y_train_poisoned.numpy()  # Return as numpy arrays for saving



def main(args):
    X_train, y_train, X_val, y_val, X_test, y_test = load_custom_data(args.seed, args.dir_dump_data, args.dataset)
    X_train_poisoned, y_train_poisoned = LFA_attack(X_train, y_train, X_val, y_val, args.adv_train_rate)

    excel_path = os.path.join(args.dir_dump_data, f'{args.dataset}_dataset_seed{args.seed}.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
        num_features = X_train_poisoned.shape[1]  # Get number of features
        feature_columns = [f'feature_{i}' for i in range(num_features)]

        df_X_train_poisoned = pd.DataFrame(X_train_poisoned, columns=feature_columns)
        df_y_train_poisoned = pd.DataFrame(y_train_poisoned, columns=['label'])
        poisoned_df = pd.concat([df_X_train_poisoned, df_y_train_poisoned], axis=1)

        poisoned_df.to_excel(writer, sheet_name=f'train_adv_{args.adv_train_rate}_lfa', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Randomized Smoothing for Label-Flipping Robustness")

    # Add arguments for hyperparameters
    parser.add_argument("--dataset", type=str, default='scikit-synthetic-moon', help='')
    parser.add_argument("--dir_dump_data", type=str, default="./dataset/", help='')
    parser.add_argument("--is_adv_train", action='store_true', help='')
    parser.add_argument("--adv_train_rate", type=int, default=0, help="poisoned label adv_train_rate")
    parser.add_argument("--seed", type=int, default=42, help="seed")


    # Parse arguments
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)