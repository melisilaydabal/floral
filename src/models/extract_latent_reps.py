import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.models.nn import *

MODEL_PATH = 'path_to_your_model.pth'

class FeatureExtractor():
    def __init__(self, dataset, in_dim, out_dim, model_path):
        if dataset in ['moon']:
            self.model = NeuralNetMoon(in_dim=in_dim, out_dim=out_dim)
        elif dataset in ['mnist_1vs7']:
            self.model = NeuralNetMNIST(in_dim=784, out_dim=out_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def extract_features(self, dataloader):
        self.model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                child_model = torch.nn.Sequential(*list(self.model.children())[:-2])
                output = child_model(inputs.double())
                features.append(output.cpu().numpy())
                labels.append(targets.cpu().numpy())

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return torch.Tensor(features), torch.Tensor(labels)

