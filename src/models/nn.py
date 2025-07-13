# models/nn.py
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout

class NeuralNet(nn.Module):

    def __init__(self, in_dim=2, out_dim=3):
        super(NeuralNet, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_dim, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 1)
        self.batch_norm = nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.batch_norm(self.relu(self.fc1(x)))
        x = self.batch_norm(self.relu(self.fc2(x)))
        x = self.batch_norm(self.relu(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x

class NeuralNetMoon(nn.Module):

    def __init__(self, in_dim=2, out_dim=3):
        super(NeuralNetMoon, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class NeuralNetMNIST(nn.Module):

    def __init__(self, in_dim=784, out_dim=3):
        super(NeuralNetMNIST, self).__init__()

        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

