# models/svm_baseline.py
import copy
import random
import torch
import cvxopt
import cvxopt.solvers
import torch.nn as nn
from cvxopt import matrix, solvers
from cvxopt import spmatrix
import numpy as np
import numpy.linalg as linalg
from sklearn import svm
from sklearn import linear_model
from sklearn.svm import OneClassSVM, LinearSVC
from sklearn.kernel_approximation import RBFSampler, Nystroem
from scipy.spatial.distance import cdist

smoothed_kernel_list = ['rbf_smooth', 'linear_smooth', 'polynomial_smooth']


class SVM_Baseline():

    def __init__(self, kernel='linear', C=0, reg_param=1.0, gamma=1.0, degree=3):
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma)
        self.w = None
        self.sv_y = None
        self.sv = None
        self.sv_indices = None
        self.flip_indices = None
        self.alphas = None
        self.sv_alphas = None
        self.kernel = kernel
        self.C = C
        self.bias = 0
        self.gamma = gamma
        self.reg_param = float(reg_param)
        # degree for polynomial kernel
        self.degree = int(degree)
        if self.kernel in smoothed_kernel_list:
            self.h = 1.0
        if self.kernel in ['matern']:
            self.nu = 1.5
            self.length_scale = 1.0
        if self.kernel in ['ln_robust_rbf']:
            self.correction_kernel_mu = 0.25
        self.sample_weights = None


    def fit(self, X, y):
        # y_np = y.numpy()
        self.alphas = np.zeros(X.shape[0])

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def rbf_kernel(self, x1, x2, gamma=1.0):
        squared_distance = torch.sum((x1 - x2) ** 2)
        return torch.exp(-gamma * squared_distance)

    def _calculate_gram_matrix(self, X1, X2):
        if X2 is None:
            X2 = copy.deepcopy(X1)
        X1 = X1.cpu()
        X2 = X2.cpu()
        X1 = np.array(X1) if not isinstance(X1, np.ndarray) else X1
        X2 = np.array(X2) if not isinstance(X2, np.ndarray) else X2

        if self.kernel in ['linear', 'linear_smooth']:
            return np.dot(X1, X2.T)
        elif self.kernel in ['rbf', 'rbf_smooth']:
            distances_squared = np.sum(X1 ** 2, axis=1, keepdims=True) - 2 * np.dot(X1, X2.T) + np.sum(
                X2 ** 2, axis=1)
            return np.exp(-self.gamma * distances_squared)
        elif self.kernel in ['polynomial', 'polynomial_smooth']:
            return (np.dot(X1, X2.T) + self.C) ** self.degree
        elif self.kernel in ['matern']:
            distances = cdist(X1, X2, metric='euclidean')

            if self.nu == 1.5:
                matern_kernel_matrix = (1 + np.sqrt(3) * distances / self.length_scale) * np.exp(
                    -np.sqrt(3) * distances / self.length_scale)
            elif self.nu == 2.5:
                matern_kernel_matrix = (1 + np.sqrt(5) * distances / self.length_scale + 5 * distances ** 2 / (
                            3 * self.length_scale ** 2)) * np.exp(-np.sqrt(5) * distances / self.length_scale)
            else:
                raise ValueError("Unsupported nu for Matern kernel. Choose nu=1.5 or nu=2.5.")
            return matern_kernel_matrix
        elif self.kernel in ['ln_robust_rbf']:
            distances_squared = np.sum(X1 ** 2, axis=1, keepdims=True) - 2 * np.dot(X1, X2.T) + np.sum(
                X2 ** 2, axis=1)
            rbf_kernel_matrix = np.exp(-self.gamma * distances_squared)
            sigma_squared = self.correction_kernel_mu * (1 - self.correction_kernel_mu)
            correction_matrix = np.ones((rbf_kernel_matrix.shape[0], rbf_kernel_matrix.shape[1])) * (1 - 4 * sigma_squared)
            np.fill_diagonal(correction_matrix, 1)
            ln_robust_kernel_matrix = correction_matrix * rbf_kernel_matrix
            return ln_robust_kernel_matrix
        else:
            raise ValueError(
                "Unsupported kernel. Choose from 'linear', 'rbf', 'polynomial', 'matern' or 'ln_robust_rbf'; or smoothed versions of those.")


    def approximate_kernel(self, gamma, num_components, X_train, X_val, X_test):
        X_train = X_train.cpu()
        X_val = X_val.cpu()
        X_test = X_test.cpu()
        if self.kernel in ['rbf', 'rbf_smooth']:
            feature_map_nystroem = Nystroem(gamma=gamma,
                                            random_state=42,
                                            n_components=num_components)
            X_train_approx = feature_map_nystroem.fit_transform(X_train)
            X_val_approx = feature_map_nystroem.transform(X_val)
            X_test_approx = feature_map_nystroem.transform(X_test)
            return torch.tensor(X_train_approx, dtype=torch.float32), torch.tensor(X_val_approx, dtype=torch.float32), torch.tensor(X_test_approx, dtype=torch.float32)

    def smoothed_kernel(self, K, X, h):
        n_samples = K.shape[0]  # K is the gram matrix, h is the bandwidth
        K_smoothed = np.zeros((n_samples, n_samples))
        X_scaled = copy.copy(X)
        X_scaled /= h  # Scale the data matrix X by the bandwidth parameter h

        for k in range(n_samples):
            dist_ik = X_scaled - X_scaled[:, k:k + 1]
            dist_ik /= h
            if self.kernel == 'rbf_smooth':
                smoothed_values = self._calculate_gram_matrix(dist_ik, dist_ik)
                print(f'smoothed values with size {smoothed_values.shape} and type {type(smoothed_values)} are {smoothed_values}')
            K_smoothed += smoothed_values
        K_smoothed /= n_samples * h ** X_scaled.shape[1]

        return K_smoothed

    def set_sample_weights(self, sample_weights):
        # Set sample weights for weighted SVM training (K-LID baseline)
        self.sample_weights = sample_weights

    def _project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.bias
        else:
            y_predict = np.zeros(len(X))
            gram_matrix = self._calculate_gram_matrix(X, self.sv)
            if self.kernel in smoothed_kernel_list:
                gram_matrix = self.smoothed_kernel(gram_matrix, X, self.h)

            for i in range(len(X)):
                s = np.sum(self.sv_alphas * self.sv_y * gram_matrix[i]) + self.bias
                y_predict[i] = s

            return y_predict


    def predict(self, X):
        # Hypothesis: sign(sum^S a * y * kernel + b).
        return np.sign(self._project(X))

    def set_flip_indices(self, list_flip_indices):
        self.flip_indices = list_flip_indices

    def get_w(self):
        return self.w

    def get_bias(self):
        return self.bias

    def compute_w(self, X, alphas):
        """
        Computes w from dual variables (Lagrange multipliers) alphas
        """
        num_samples, num_features = X.shape
        if self.kernel == 'linear':
            self.w = np.zeros(len(alphas))
            self.w = ((self.sv_y * alphas).T @ self.sv.numpy())
        else:
            # For a new point x, we compute y_predict by looking its distance to hyperplane defined by the SVs
            self.w = np.zeros(len(num_samples))
            gram_matrix = self._calculate_gram_matrix(X, self.sv)
            if self.kernel in smoothed_kernel_list:
                gram_matrix = self.smoothed_kernel(gram_matrix, X, self.h)
            for i in range(len(X)):
                self.w[i] = np.sum(alphas * self.sv_y * gram_matrix[i])
            return self.w

    def compute_bias(self, X, alphas):
        """Computes bias from dual variables (Lagrange multipliers) alphas"""
        self.bias = 0
        K = self._calculate_gram_matrix(X, X)
        if self.kernel in smoothed_kernel_list:
            K = self.smoothed_kernel(K, X, self.h)
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]

        for n in range(len(ind)):
            self.bias += self.sv_y[n]
            self.bias -= np.sum(self.sv_alphas * self.sv_y * K[ind[n], sv])
        if len(self.sv_alphas) == 0:
            self.bias = 0
        else:
            self.bias = self.bias / len(self.sv_alphas)

    def update_sv(self, X, y, alphas):
        """Updates the dual variables (Lagrange multipliers)"""
        y = y.numpy()
        self.alphas = alphas
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.sv_indices = ind
        self.sv_alphas = alphas[self.sv_indices]
        self.sv = X[self.sv_indices]
        self.sv_y = y[self.sv_indices]

    def compute_gradient(self, X, y):
        """"Computes the gradient of the dual objective evaluated at the current alphas"""
        e = np.ones(len(self.alphas))
        K = self._calculate_gram_matrix(X, X)
        if self.kernel in smoothed_kernel_list:
            K = self.smoothed_kernel(K, X, self.h)
        compute = np.outer(y, y) * K
        # print(f"np.outer(y, y) * K size is {compute.shape}")

        gradient = np.sum(np.outer(y, y) * K * self.alphas, axis=1) - e
        return gradient


    def compute_gradient_at_point(self, X, y, alphas):
        """"Computes the gradient of the dual objective evaluated at given point: alphas"""
        e = np.ones(len(alphas))
        K = self._calculate_gram_matrix(X, X)

        if self.kernel in smoothed_kernel_list:
            K = self.smoothed_kernel(K, X, self.h)
        gradient = np.sum(np.outer(y, y) * K * self.alphas, axis=1) - e
        return gradient

    def pgd_svm_step(self, alphas_init, alphas_gradient, lr):
        """Computes the PGD step for the dual variable update"""
        return alphas_init - lr * alphas_gradient


    def check_project_constraints(self, y, alphas_step, C):
        """Computes whether PGD step should be projected or not i.e. satisfy constraints or not"""
        y = y.numpy()
        equality_const = np.sum(y * alphas_step)
        if np.any(alphas_step < 0) or np.any(alphas_step > C) or not equality_const == 0:
            # Need to project
            return True
        else:
            return False

    def pgd_project(self, y, alphas_step, C):
        """Computes the projection step of PGD"""
        y = y.numpy()
        num_samples = len(alphas_step)

        Q = spmatrix(1.0, range(num_samples), range(num_samples))
        p = cvxopt.matrix(np.ones(num_samples) * -alphas_step)
        A = cvxopt.matrix(y, (1, num_samples), 'd')     # A has to be double matrix
        b = cvxopt.matrix(0.0)

        if C is None or C == 0:
            G = cvxopt.matrix(np.diag(np.ones(num_samples) * -1))
            h = cvxopt.matrix(np.zeros(num_samples))
        else:
            # Restricting the optimisation with parameter C.
            tmp1 = np.diag(np.ones(num_samples) * -1)
            tmp2 = np.identity(num_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(num_samples)
            tmp2 = np.ones(num_samples) * self.C
            if self.sample_weights is not None:
                # Weighted SVM training
                tmp2 = tmp2 * self.sample_weights.numpy()
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Setting options:
        cvxopt.solvers.options['show_progress'] = True
        cvxopt.solvers.options['abstol'] = 1e-10
        cvxopt.solvers.options['reltol'] = 1e-10
        cvxopt.solvers.options['feastol'] = 1e-10

        # Solve QP problem:
        solution = cvxopt.solvers.qp(Q, p, G, h, A, b)
        print(f"Solution is : {solution}")

        # Lagrange multipliers
        alphas = np.ravel(solution['x'])        # Flatten the matrix into a vector of all the Langrangian multipliers.
        solution_size = solution['x'].size

        return alphas   # projected PGD step


    def pgd_project_fpi_based(self, y, alphas_step, C):
        # Approximates the projection step via grid search on \mu*
        y = y.numpy()
        nb_project_iterations = 1000
        error_rate = 0.000000000000000000001
        mu = []
        mu.append(0.0)
        alphas_0 = copy.deepcopy(alphas_step)

        for itr in range(nb_project_iterations):
            check_alphas = alphas_0 - mu[itr] * y
            indices_C = list(np.where(check_alphas >= C)[0])
            indices_x = list(np.where((check_alphas > 0) & (check_alphas < C))[0])
            print(f"If index set_x is empty? : length is {len(indices_x)}")

            eta = np.maximum(2 * len(indices_x), 1)
            update_mu = ((eta - len(indices_x)) / eta) * mu[itr] + (1 / eta) * (
                        np.sum(C * y[indices_C]) + np.sum(alphas_step[indices_x] * y[indices_x]))
            mu.append(update_mu)
            mu_difference = np.abs(mu[itr + 1] - mu[itr])
            if mu_difference <= error_rate:
                print(f'Stopped at iteration {itr}')
                break
        projected_alphas = np.clip(alphas_0 - mu[itr + 1] * y, 0, C)
        print(f'Equality constraint value (with clip) = {np.sum(projected_alphas * y)}')
        return projected_alphas

