import os
import pandas as pd
import numpy as np
import torch
import random
from cvxopt import matrix, solvers
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import linprog, minimize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor

SEEDS = [42]
BUDGET_SIZES = [5, 10, 25]


def load_custom_data(seed):
    # Loads custom dataset from CSV
    dir_dump_data = './dataset/'
    dataset = 'moon'
    excel_path = os.path.join(dir_dump_data, f'{dataset}_seed{seed}.xlsx')

    main_df = pd.read_excel(excel_path, sheet_name='main')
    train_df = pd.read_excel(excel_path, sheet_name='train')
    val_df = pd.read_excel(excel_path, sheet_name='validation')
    test_df = pd.read_excel(excel_path, sheet_name='test')

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_val = val_df.iloc[:, :-1].values
    y_val = val_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_hinge_loss(clf, X, y):
    y_pred = clf.decision_function(X)
    return np.maximum(0, 1 - y * y_pred)

def qp_step(X, y, q, C):
    n, m = X.shape  # n: number of samples, m: number of features

    # Initialize epsilon^0 and epsilon^1 to zeros
    eps_0 = np.zeros(n)
    eps_1 = np.zeros(n)

    # The number of decision variables is m (for w), 1 (for b), and 2*n (for eps_0 and eps_1)
    total_vars = m + 1 + 2 * n
    # Construct the quadratic term matrix (P)
    # Only w terms are quadratic, so we build P with identity for w and zero elsewhere
    P = np.zeros((total_vars, total_vars))
    P[:m, :m] = np.eye(m)  # Quadratic term for w (w^T w)
    # No quadratic terms for b, epsilon^0, or epsilon^1

    q_vec = np.zeros(total_vars)
    q_vec[m + 1:m + 1 + n] = C * (1 - q)  # Linear term for epsilon^0
    q_vec[m + 1 + n:] = C * q  # Linear term for epsilon^1

    # Inequality constraints: G * [w; b; eps^0; eps^1] <= h
    G = np.zeros((4 * n, total_vars))  # 4*n constraints (2 for hinge loss, 2 for non-negativity of epsilon)
    h = np.zeros(4 * n)

    for i in range(n):
        # First constraint: y_i * (w^T x_i + b) >= 1 - eps_0_i
        G[2 * i, :m] = -y[i] * X[i]
        G[2 * i, m] = -y[i]
        G[2 * i, m + 1 + i] = -1  # -eps_0_i
        h[2 * i] = -(1)

        # Second constraint: -y_i * (w^T x_i + b) >= 1 - eps_1_i
        G[2 * i + 1, :m] = y[i] * X[i]
        G[2 * i + 1, m] = y[i]
        G[2 * i + 1, m + 1 + n + i] = -1  # -eps_1_i
        h[2 * i + 1] = -(1)

        # Third constraint: eps_0_i >= 0
        G[2 * n + i, m + 1 + i] = -1  # -eps_0_i <= 0
        h[2 * n + i] = 0  # Right-hand side is 0 for non-negativity

        # Fourth constraint: eps_1_i >= 0
        G[3 * n + i, m + 1 + n + i] = -1  # -eps_1_i <= 0
        h[3 * n + i] = 0  # Right-hand side is 0 for non-negativity

    P_matrix = matrix(P)
    q_matrix = matrix(q_vec)
    G_matrix = matrix(G)
    h_matrix = matrix(h)

    # Solve the QP
    solution = solvers.qp(P_matrix, q_matrix, G_matrix, h_matrix)

    w = np.array(solution['x'][:m])  # First m entries are w
    b = solution['x'][m]  # Next entry is b
    slack = np.array(solution['x'][m + 1:])  # Remaining entries are slack variables

    eps_0_sol = slack[:n]  # First n entries are epsilon^0
    eps_1_sol = slack[n:]  # Next n entries are epsilon^1

    return w, b, eps_0_sol.reshape(-1), eps_1_sol.reshape(-1)


def lp_step(eps_0, eps_1, xi_0, xi_1, C, budget):
    n = len(eps_0)  # Number of samples

    c = C * ((eps_0 - xi_0) - (eps_1 - xi_1))
    c = c.flatten()
    c_matrix = matrix(c, (n, 1), 'd')

    # Constraints:
    # 1. sum(q_i) <= L
    # 2. 0 <= q_i <= 1 for all i

    # Define G and h for inequality constraints G * q <= h
    G = np.vstack([-np.eye(n), np.eye(n)])  # To ensure 0 <= q_i <= 1
    h = np.hstack([np.zeros(n), np.ones(n)])  # Corresponding bounds

    # The sum(q_i) <= L constraint (one row, n columns)
    A = np.ones((1, n))
    b = np.array([budget])

    # Convert to cvxopt format
    G_matrix = matrix(G)
    h_matrix = matrix(h)
    A_matrix = matrix(A)
    b_matrix = matrix(b)

    # Solve the LP
    solution = solvers.lp(c_matrix, G_matrix, h_matrix, A_matrix, b_matrix)

    # Extract the optimal q values
    q_opt = np.array(solution['x']).flatten()
    return q_opt

def save_tensors(X, y, save_path_features, save_path_labels):
    torch.save(torch.tensor(X), save_path_features)
    torch.save(torch.tensor(y), save_path_labels)

def train_initial_svm(X, y, C, gamma):
    svm = SVC(kernel='rbf', C=C, gamma=gamma)
    svm.fit(X, y)
    return svm

def compute_slack_and_costs(X, y, svm):
    predictions = svm.decision_function(X)
    losses_1 = np.maximum(0, 1 - y * predictions)
    losses_2 = np.maximum(0, 1 + y * predictions)
    slacks_1 = np.zeros_like(y, dtype=float)
    slacks_2 = np.zeros_like(y, dtype=float)
    return losses_1, losses_2, slacks_1, slacks_2

def alfa_attack(X, y, budget, C, gamma):
    svm = train_initial_svm(X, y, C, gamma)
    losses_0, losses_1, slacks_0, slacks_1 = compute_slack_and_costs(X, y, svm)

    list_q = []
    for i in range(10):
    # nm_iterations set according to the experiment results of ALFA paper.
        # Solve LP and QP as per the algorithm
        print(f"Solving LP step")
        q = lp_step(slacks_0, slacks_1, losses_0, losses_1, C, budget)
        print(f"Solving QP step")
        w, b, slacks_0, slacks_1 = qp_step(X, y, q, C)
        print(f"QP solution found")
        list_q.append(q)

    # Sort "indices" of points by adversarial cost
    sorted_indices = np.argsort(list_q[-1])[::-1]

    # Flip labels based on sorted indices and budget
    y_prime = y.copy()
    total_flips = 0
    for idx in sorted_indices:
        if total_flips < budget:
            y_prime[idx] = -y[idx]
            total_flips += 1
        else:
            break

    return y_prime

if __name__ == '__main__':

    for budget_size in BUDGET_SIZES:
        for seed in SEEDS:

            # set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            X_train, y_train, X_val, y_val, X_test, y_test = load_custom_data(seed)
            costs = np.random.random(len(y_train))  # Example costs
            budget = np.ceil(X_train.shape[0] * budget_size / 100)
            gamma = 1.0
            C = 1.0

            # Initial SVM accuracy
            clf = train_initial_svm(X_train, y_train, C=C, gamma=gamma)
            y_pred = clf.predict(X_test)
            clean_accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy on clean data: {clean_accuracy:.4f}")

            # Apply ALFA attack
            y_train_attacked = alfa_attack(X_train, y_train, budget, C, gamma)
            clf_attacked = train_initial_svm(X_train, y_train_attacked, C=C, gamma=gamma)
            y_pred_attacked = clf_attacked.predict(X_test)
            attacked_accuracy = accuracy_score(y_test, y_pred_attacked)
            print(f"Accuracy on attacked data: {attacked_accuracy:.4f}")

            save_path_features = f'./dataset/dataset_features_budget{budget_size}_seed{seed}.pt'
            save_path_labels = f'./dataset/labels_budget{budget_size}_seed{seed}.pt'
            save_tensors(X_train, y_train_attacked, save_path_features, save_path_labels)
