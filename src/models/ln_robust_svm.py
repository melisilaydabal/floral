# models/ln_robust_svm.py
"""
Implementation for Robust SVM under Adversarial Label Noise by Biggio et al. (2011)
"""
import copy
import numpy as np
from sklearn import svm
# from sklearn.metrics.pairwise import rbf_kernel


class LNRobustSVM():

    def __init__(self, base_kernel='linear', C=0, reg_param=1.0, gamma=1.0, degree=3):
        # Define the baseline SVM model with a defined kernel
        self.base_kernel = base_kernel
        self.gamma = gamma
        self.C = C
        self.correction_kernel_mu = 0.05
        self.model = svm.SVC(kernel=self.ln_robust_kernel, C=self.C)
        self.train_x = None

    def rbf_gram_matrix(self, X1, X2=None, gamma=1.0):
        gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                gram_matrix[i, j] = self.rbf_kernel(x1, x2, gamma)
        return gram_matrix

    def rbf_kernel(self, x1, x2=None, gamma=1.0):
        squared_distance = np.sum((x1 - x2) ** 2)
        return np.exp(-gamma * squared_distance)

    def ln_robust_kernel(self, X1, X2=None):
        print(f"X1 shape: {X1.shape}")
        if X2 is not None:
            print(f"X2 shape: {X2.shape}")

        if X2 == None:
            X2 = copy.deepcopy(X1)

        X1 = np.array(X1.cpu()) if not isinstance(X1, np.ndarray) else X1
        X2 = np.array(X2.cpu()) if not isinstance(X2, np.ndarray) else X2

        if self.base_kernel == 'rbf':
            K = self.rbf_gram_matrix(X1, X2, gamma=self.gamma)  # Compute the original kernel matrix
        else:
            raise NotImplementedError("Only RBF kernel is implemented in this example.")
        sigma_squared = self.correction_kernel_mu * (1 - self.correction_kernel_mu)
        # Create the correction matrix with the same shape as K
        correction_matrix = np.ones((X1.shape[0], X2.shape[0])) * (1 - 4 * sigma_squared)
        if np.array_equal(X1, X2):
            np.fill_diagonal(correction_matrix, 1)
        K_corrected = correction_matrix * K
        return K_corrected

    def eval_ln_robust_kernel(self, X1, X2=None):
        print(f"X1 shape: {X1.shape}")
        if X2 is not None:
            print(f"X2 shape: {X2.shape}")

        if X2 == None:
            X2 = copy.deepcopy(X1)

        X1 = np.array(X1.cpu()) if not isinstance(X1, np.ndarray) else X1
        X2 = np.array(X2.cpu()) if not isinstance(X2, np.ndarray) else X2

        if self.base_kernel == 'rbf':
            K = self.rbf_kernel(X1, X2, gamma=self.gamma)  # Compute the original kernel matrix
        else:
            raise NotImplementedError("Only RBF kernel is implemented in this example.")
        sigma_squared = self.correction_kernel_mu * (1 - self.correction_kernel_mu)
        # Create the correction matrix with the same shape as K
        correction_matrix = np.ones((X1.shape[0], X2.shape[0])) * (1 - 4 * sigma_squared)
        if np.array_equal(X1, X2):
            np.fill_diagonal(correction_matrix, 1)
        K_corrected = correction_matrix * K
        return K_corrected

    def correct_kernel_matrix(self, K, X1):
        sigma_squared = self.correction_kernel_mu * (1 - self.correction_kernel_mu)
        correction_matrix = np.identity(X1.shape[0])
        correction_matrix[correction_matrix == 0] = 1 - 4 * sigma_squared
        K_corrected = correction_matrix * K
        return K_corrected

    def fit(self, X, y):
        self.train_x = X
        self.kernel = self.ln_robust_kernel(X)
        self.model.fit(X, y)

    def _project(self, X):
        # To use scikit-learn predict() function
        K_test = self.eval_ln_robust_kernel(X, self.train_x)
        y_predict = self.model.predict(K_test)
        return y_predict

    def predict(self, X):
        # Hypothesis: sign(sum^S a * y * kernel + b).
        return np.sign(self._project(X))