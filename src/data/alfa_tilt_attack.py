import os
import pandas as pd
import numpy as np
import torch
import random
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

SEEDS = [42]
BUDGET_SIZES = [5, 10, 25]

def train_initial_svm(X, y, C=1.0, gamma=1.0, kernel='rbf'):
    svm = SVC(kernel='rbf', C=C, gamma=gamma)
    svm.fit(X, y)
    dual_coefs_set = svm.dual_coef_.flatten() if svm.dual_coef_.shape[0] == 1 else svm.dual_coef_.flatten()[0]
    intercept = svm.intercept_[0]
    dual_coefs = np.zeros(y.shape)
    dual_coefs[svm.support_] = dual_coefs_set
    return dual_coefs, intercept, svm

def compute_signed_margins(X, y, clf, kernel_func=rbf_kernel):
    # Get the decision function values for all points
    margins = clf.decision_function(X)  # This gives the margin for all points
    signed_margins = y * margins  # Multiply by labels to get signed margins
    return signed_margins / np.max(np.abs(signed_margins))  # Normalize the margins

def generate_random_svm(n):
    alpha_rnd = np.random.uniform(-1, 1, size=n)
    b_rnd = np.random.uniform(-1, 1)
    return alpha_rnd, b_rnd

def alfa_tilt_attack(X, y, L, N, C=1.0, gamma=1.0, beta1=1.0, beta2=1.0):
    n = len(y)

    # Step 1: Train initial SVM
    alpha, b, clf = train_initial_svm(X, y, C, gamma)  # Get the trained SVM
    si = compute_signed_margins(X, y, clf)  # Compute signed margins using the decision function

    best_tilt_angle = -np.inf
    best_labels = y.copy()

    for _ in range(N):
        # Step 2: Generate a random SVM
        # Random SVM margins (this should also use decision_function instead of dual_coef_)
        alpha_rnd, b_rnd, clf_rnd = train_initial_svm(X, y, C, gamma)
        qi = compute_signed_margins(X, y, clf_rnd)  # Compute random margins

        # Step 3: Compute importance scores
        vi = alpha / C - beta1 * si - beta2 * qi
        sorted_indices = np.argsort(vi)  # Ascending order

        # Step 4: Iteratively flip labels
        z = y.copy()
        for i in range(L):
            z[sorted_indices[i]] = -z[sorted_indices[i]]

        # Step 5: Train SVM on flipped labels and compute tilt angle
        alpha_prime, b_prime, clf_flipped = train_initial_svm(X, z, C)
        tilt_angle = compute_tilt_angle(alpha, alpha_prime, X, y, z, kernel_gamma=gamma)

        # Keep track of the best label flipping
        if tilt_angle > best_tilt_angle:
            best_tilt_angle = tilt_angle
            best_labels = z.copy()

    return X, best_labels

def compute_tilt_angle(alpha, alpha_prime, X, y, z, kernel_gamma=1.0):
    # Compute the RBF kernel matrix
    K = rbf_kernel(X, X, gamma=kernel_gamma)

    # Compute Q matrices
    Q_zy = K * np.outer(z, y)  # Element-wise multiplication with outer product
    Q_zz = K * np.outer(z, z)
    Q_yy = K * np.outer(y, y)

    # Numerator: alpha' T * Qzy * alpha
    numerator = np.dot(alpha_prime.T, np.dot(Q_zy, alpha))

    # Denominator: sqrt(alpha' T * Qzz * alpha') * sqrt(alpha T * Qyy * alpha)
    denom_alpha_prime = np.sqrt(np.dot(alpha_prime.T, np.dot(Q_zz, alpha_prime)))
    denom_alpha = np.sqrt(np.dot(alpha.T, np.dot(Q_yy, alpha)))
    denominator = denom_alpha_prime * denom_alpha

    # Tilt angle calculation
    tilt_angle = numerator / denominator
    return tilt_angle


# Load custom dataset from CSV
def load_custom_data(seed):
    dir_dump_data = './dataset/'
    dataset = 'moon'
    excel_path = os.path.join(dir_dump_data, f'{dataset}_seed{seed}.xlsx')

    main_df = pd.read_excel(excel_path, sheet_name='main')
    train_df = pd.read_excel(excel_path, sheet_name='train')
    val_df = pd.read_excel(excel_path, sheet_name='validation')
    test_df = pd.read_excel(excel_path, sheet_name='test')

    # Convert to numpy arrays
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_val = val_df.iloc[:, :-1].values
    y_val = val_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_tensors(X, y, save_path_features, save_path_labels):
    torch.save(torch.tensor(X), save_path_features)
    torch.save(torch.tensor(y), save_path_labels)


if __name__ == '__main__':

    for budget_size in BUDGET_SIZES:
        for seed in SEEDS:

            # Set random seed
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
            _, _, clf = train_initial_svm(X_train, y_train, C=C, gamma=gamma)
            y_pred = clf.predict(X_test)
            clean_accuracy = accuracy_score(y_test, y_pred)

            # Parameters for the Alfa-Tilt attack
            beta1 = 1.0  # Weighting parameter for signed margin
            beta2 = 1.0  # Weighting parameter for random margin

            # Apply ALFA attack
            X_train, y_train_attacked = alfa_tilt_attack(X_train, y_train, L=int(budget), N=5, C=1.0, gamma=gamma, beta1=1.0, beta2=1.0)

            _, _, clf_attacked = train_initial_svm(X_train, y_train_attacked, C=C, gamma=gamma)
            y_pred_attacked = clf_attacked.predict(X_test)
            attacked_accuracy = accuracy_score(y_test, y_pred_attacked)
            print(f"Accuracy on attacked data: {attacked_accuracy:.4f}")

            save_path_features = f'./dataset/dataset_features_budget{budget_size}_seed{seed}.pt'
            save_path_labels = f'./dataset/labels_budget{budget_size}_seed{seed}.pt'
            save_tensors(X_train, y_train_attacked, save_path_features, save_path_labels)
