import torch
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, FunctionTransformer

class Preprocessor():
    def __init__(self, config):
        self.config = config

    def _custom_transform(self, X):
        return np.sin(X)

    def transform(self, X, type='poly'):
        if type == 'poly':
            transformer = PolynomialFeatures(degree=self.config.get('model').get('degree'))
        elif type == 'custom':
            transformer = FunctionTransformer(self._custom_transform)
        X_transformed = transformer.fit_transform(X)
        return torch.tensor(X_transformed, dtype=torch.float32)

    def scale_transform(self, X):
        scaler = StandardScaler().fit(X)
        X_transformed = scaler.transform(X)
        return torch.tensor(X_transformed, dtype=torch.float32)



