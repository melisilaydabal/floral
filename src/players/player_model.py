# players/player_model.py
"""
Player_model
Updates the model (decision boundary) using flipped labels
"""
import os
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.linalg import eigvals
from torch import nn
from matplotlib.lines import Line2D
from transformers import RobertaForSequenceClassification
from src.models.nn import *
from src.models.svm_baseline import *
from src.models.ln_robust_svm import *
from src.models.svm_opt import *
from src.utils.metrics import *
from src.utils.utils import *
from src.utils.plotter import _create_fig, plot_perf, _plot_posterior_style, _plot_prior_style, plot_perf_training, custom_colormap

# Define some helpers
LIST_BENCHMARK_DATASETS = ['moon', 'imdb', 'mnist_1vs7']
LIST_NN_MODELS = ['nn', 'nn_pgd']
LIST_SVM_MODELS = ['svm', 'ln-robust-svm']

class Player_Model():
    def __init__(self, config):
        # Initialize the model, optimizer, and criterion
        if config.get('model').get('name') == 'svm':
            self.model_name = 'svm'
            if config.get('data').get('dataset') in LIST_BENCHMARK_DATASETS:
                # Because CVXOPT gives matrix overflow error--we cannot use it for large datasets
                self.model = SVM_Baseline(kernel=config.get('model').get('kernel'),
                                     C=config.get('model').get('C'),
                                     gamma=config.get('model').get('gamma'))
            else:
                self.model = SVM(kernel=config.get('model').get('kernel'), C=config.get('model').get('C'),
                                 reg_param=config.get('model').get('reg_param'),
                                 gamma=config.get('model').get('gamma'), degree=config.get('model').get('degree'))

            self.criterion = nn.HingeEmbeddingLoss()

        elif config.get('model').get('name') == 'ln-robust-svm':
            self.model_name = 'ln-robust-svm'
            self.model = LNRobustSVM(base_kernel=config.get('model').get('kernel'),
                                      C=config.get('model').get('C'),
                                      gamma=config.get('model').get('gamma'))
            self.criterion = nn.HingeEmbeddingLoss()

        elif config.get('model').get('name') in LIST_NN_MODELS:
            self.model_name = config.get('model').get('name')
            if config.get('data').get('dataset') in ['moon']:
                self.model = NeuralNetMoon(in_dim=config.get('data').get('in_dim'),
                                   out_dim=config.get('data').get('num_classes'))
            elif config.get('data').get('dataset') in ['mnist_1vs7']:
                self.model = NeuralNetMNIST(in_dim=784,
                                   out_dim=config.get('data').get('num_classes'))

            if not config.get('model').get('is_baseline'):
                optimizer_type = config.get('optim').get('optimizer')
                try:
                    optimizer_class = getattr(importlib.import_module('torch.optim'), optimizer_type)
                except AttributeError:
                    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
                self.optimizer = optimizer_class(self.model.parameters(), lr=config.get('optim').get('lr'), momentum=0.9)
            self.criterion = torch.nn.BCELoss()
        elif config.get('model').get('name') in ['roberta']:
            self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
        else:
            self.model = NeuralNet(in_dim=config.get('data').get('in_dim'),
                                   out_dim=config.get('data').get('num_classes'))
            self.criterion = torch.nn.BCELoss()

    def _svm_loss(self, scores, labels):
        hinge_loss = torch.mean(torch.max(torch.zeros_like(torch.Tensor(scores)), 1 - labels * torch.Tensor(scores)))
        return hinge_loss

    def dump_model(self, model_path):
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)


    def PGD(self, model, X, y, alpha=2/255, epsilon=8/255, nb_pgd_iter=10):
        delta = torch.zeros_like(X, requires_grad=True)
        delta = delta.double()
        for i in range(nb_pgd_iter):
            criterion=torch.nn.BCELoss()
            output = model(X + delta)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        pert = delta.detach()
        X_adv = X + pert
        h_adv = model(X_adv)
        _,y_adv = torch.max(h_adv.data,1)
        return X_adv, h_adv, y_adv, pert

    def train_and_evaluate(self, config, dir_exp_folder, round, model, train_loader, val_loader, test_loader, optimizer, criterion, lr, num_epochs=50):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        test_losses = []
        test_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                if self.model_name in ['nn_pgd']:
                    batch_x_adv, _, _, _ = self.PGD(model, batch_x.double(), batch_y.double(), 2/255, 8/255, 5)
                    output = model(batch_x_adv.double())
                else:
                    output = model(batch_x.double())
                loss = criterion(output.squeeze(), batch_y.double())
                loss.backward()
                optimizer.step()

            # Training
            avg_train_loss, avg_train_acc = self._evaluate(config, model, train_loader, criterion)
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Pre-Train Loss: {avg_train_loss:.4f}, "
                  f"Pre-Train Accuracy: {avg_train_acc:.4f}")
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)

            # Validation
            avg_val_loss, avg_val_acc = self._evaluate(config, model, val_loader, criterion)
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Pre-Validation Loss: {avg_val_loss:.4f}, "
                  f"Pre-Validation Accuracy: {avg_val_acc:.4f}")
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)

            # Test
            avg_test_loss, avg_test_acc = self._evaluate(config, model, test_loader, criterion)
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Pre-Validation Loss: {avg_test_loss:.4f}, "
                  f"Pre-Validation Accuracy: {avg_test_acc:.4f}")
            test_losses.append(avg_test_loss)
            test_accuracies.append(avg_test_acc)

        # Save the model
        dump_model_path = os.path.join(config.get('dump').get('dir_dump'), dir_exp_folder)
        model_path = os.path.join(dump_model_path, f"{config.get('data').get('dataset')}"
                                                   f"_{config.get('model').get('name')}"
                                                   f"_{config.get('optim').get('optimizer')}optim"
                                                   f"_epoch{num_epochs}"
                                                   f"_lr{lr}"
                                                   f"_isperturbed_{config.get('training').get('is_perturbed')}.pt")
        torch.save(model.state_dict(), model_path)

        # Plotting classification accuracy over the training set over time
        plot_perf(config, dir_exp_folder, round, train_losses, 'Train Loss', 'train', num_epochs)
        plot_perf(config, dir_exp_folder, round, train_accuracies, 'Train Accuracy', 'train', num_epochs)
        dump_nn_accuracy_loss(config, config.get('dump').get('dir_dump'), dir_exp_folder, round, config, train_accuracies, train_losses, 'train')

        # Plotting classification accuracy over the validation set over time
        plot_perf(config, dir_exp_folder, round, val_losses, 'Validation Loss', 'validation', num_epochs)
        plot_perf(config, dir_exp_folder, round, val_accuracies, 'Validation Accuracy', 'validation', num_epochs)
        dump_nn_accuracy_loss(config, config.get('dump').get('dir_dump'), dir_exp_folder, round, config, val_accuracies, val_losses, 'validation')

        # Plotting classification accuracy over the test set over time
        plot_perf(config, dir_exp_folder, round, test_losses, 'Test Loss', 'test', num_epochs)
        plot_perf(config, dir_exp_folder, round, test_accuracies, 'Test Accuracy', 'test', num_epochs)
        dump_nn_accuracy_loss(config, config.get('dump').get('dir_dump'), dir_exp_folder, round, config, test_accuracies, test_losses, 'test')

        dict_losses, dict_accuracies = {}, {}
        dict_losses['train'] = train_losses
        dict_losses['validation'] = val_losses
        dict_losses['test'] = test_losses
        plot_perf_training(config, dir_exp_folder, round, dict_losses, 'Pretraining Loss', '', num_epochs)
        del dict_losses
        dict_accuracies['train'] = train_accuracies
        dict_accuracies['validation'] = val_accuracies
        dict_accuracies['test'] = test_accuracies
        plot_perf_training(config, dir_exp_folder, round, dict_accuracies, 'Pretraining Accuracy', '', num_epochs)
        del dict_accuracies


    def _evaluate(self, config, model, loader, criterion):
        if config.get('model').get('name') in LIST_NN_MODELS:
            model.eval()
        losses = []
        accuracies = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                if config.get('model').get('name') in LIST_SVM_MODELS:
                    if config.get('data').get('dataset') in LIST_BENCHMARK_DATASETS:
                        # With SVM_baseline() model class, we use scikit-learn's prediction function.
                        # It requires loader's to use int32. However, this creates trouble if you use it with
                        # your predict function, as precision gets worse than actual.
                        scores = model._project(batch_x)
                    else:
                        scores = model._project(batch_x)
                    loss = self._svm_loss(scores, batch_y)
                else:
                    output = model(batch_x.double())
                    loss = criterion(output.squeeze(), batch_y.double())
                if config.get('model').get('name') not in LIST_NN_MODELS:
                    output = (torch.Tensor(scores) > 0).int()
                    acc = calculate_accuracy(batch_y, output, config.get('data').get('num_classes'))
                else:
                    acc = calculate_accuracy(batch_y, output.squeeze(), config.get('data').get('num_classes'))
                losses.append(loss.item())
                accuracies.append(acc.item())

        avg_loss = sum(losses) / len(losses)
        avg_accuracy = sum(accuracies) / len(accuracies)

        return avg_loss, avg_accuracy

    def train(self, X, y):
        self.model.fit(X, y)

    def _visualize_decision_boundary(self, config, dir_exp_folder, X, y, plot_name, round):
        # Create a meshgrid for plotting the decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        h = 0.02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

        if config.get('model').get('name') in LIST_NN_MODELS:
            self.model.eval()
            with torch.no_grad():
                Z = self.model(grid_tensor.double()).numpy().reshape(xx.shape)
        else:
            if config.get('model').get('is_baseline') == True:
                Z = self.model.decision_function(grid_tensor).reshape(xx.shape)
            else:
                # Prediction on the meshgrid
                Z = self.model._project(grid_tensor).reshape(xx.shape)

        fig, ax = _create_fig()
        cmap = _plot_prior_style(fig, ax)
        cmap_contour = plt.get_cmap('viridis')
        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_contour)

        if config.get('model').get('name') in LIST_NN_MODELS:
            ax.contour(xx, yy, Z, [0.5], colors='black', linestyles='--', linewidths=2, origin='lower')
            ax.contour(xx, yy, Z + 1, [0.5], colors='grey', linestyles='--', linewidths=1, origin='lower')
            ax.contour(xx, yy, Z - 1, [0.5], colors='grey', linestyles='--', linewidths=1, origin='lower')
        else:
            ax.contour(xx, yy, Z, [0.0], colors='black', linestyles='--', linewidths=2, origin='lower')
            ax.contour(xx, yy, Z + 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
            ax.contour(xx, yy, Z - 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, marker='o', s=80, edgecolors='k')
        if config.get('model').get('name') not in LIST_NN_MODELS and plot_name not in ["Test"]:
            if config.get('model').get('is_baseline') == True:
                ax.scatter(self.model.support_vectors_[:, 0], self.model.support_vectors_[:, 1], s=20, color="yellow", edgecolors='k')  # The points designating the support vectors.
            else:
                ax.scatter(self.model.sv[:, 0], self.model.sv[:, 1], s=20, color="yellow", edgecolors='k')  # The points designating the support vectors.
        fig.suptitle(f'Decision Boundary on {plot_name} Dataset | Round {round}', fontsize=40)
        _plot_posterior_style(fig, ax, "Feature 0", "Feature 1")

        plot_path = os.path.join(f"{dir_exp_folder}/"
                        f"round{round}"
                        f"_contour{plot_name}_pca{config.get('data').get('is_pca')}"
                        f"_{config.get('data').get('dataset')}"
                        f"_model{config.get('model').get('name')}"
                        f"_soft_C{config.get('model').get('C')}_reg{config.get('model').get('reg_param')}"
                        f"_epoch{config.get('training').get('num_epochs')}"                                             
                        f"_isperturbed_{config.get('training').get('is_perturbed')}_plot.png")
        # Set the legend
        legendElements = [
            Line2D([0], [0], linestyle='none', marker='o', color='blue', markersize=7, markeredgecolor='k'),
            Line2D([0], [0], linestyle='none', marker='o', color='red', markersize=7, markeredgecolor='k'),
            Line2D([0], [0], linestyle='none', marker='o', color='yellow', markersize=7, markeredgecolor='k'),
            Line2D([0], [0], linestyle='--', linewidth=2, marker='.', color='black', markersize=0),
            Line2D([0], [0], linestyle='--', linewidth=1, marker='.', color='grey', markersize=0),
        ]
        myLegend = plt.legend(legendElements,
                                  ['Negative -1', 'Positive +1', 'Support Vectors', 'Decision Boundary', 'Margin'],
                                  fontsize="9", loc='upper right')
        myLegend.get_frame().set_linewidth(0.3)

        fig.colorbar(contour, ax=ax)
        fig.savefig(plot_path)

        # To plot the pdf version of the last round(s) -- for the paper:
        if round in [499, 500, 501, 999, 1000, 1001]:
            plot_path = os.path.join(f"{dir_exp_folder}/"
                                     f"round{round}"
                                     f"_contour{plot_name}_pca{config.get('data').get('is_pca')}"
                                     f"_{config.get('data').get('dataset')}"
                                     f"_model{config.get('model').get('name')}"
                                     f"_soft_C{config.get('model').get('C')}_reg{config.get('model').get('reg_param')}"
                                     f"_epoch{config.get('training').get('num_epochs')}"
                                     f"_isperturbed_{config.get('training').get('is_perturbed')}_plot.pdf")
            fig.savefig(plot_path)
        plt.close()

    def clip_gradient(self, grad, max_abs_value):
        return torch.clamp(grad, -max_abs_value, max_abs_value)

    def compute_kernel_eigenvals(self, config, dir_exp_folder, X):
        K = self.model._calculate_gram_matrix(X, X)
        eigenValues = eigvals(K)
        complex_mask = np.iscomplex(eigenValues)
        complex_indices = np.where(complex_mask)[0]
        eigenValues[complex_indices] = [-999 for _ in range(len(complex_indices))]
        eigenValues = np.sort(eigenValues)[::-1]
        self._plot_kernel_eigenvals(config, dir_exp_folder, eigenValues)
        return eigenValues

    def _plot_kernel_eigenvals(self, config, dir_exp_folder, eigenValues):
        fig, ax = _create_fig()
        plt.plot([i for i in range(len(eigenValues))], eigenValues, linewidth=5, markersize=12)
        fig.suptitle(f'Eigenvalues of kernel', fontsize=40)
        _plot_posterior_style(fig, ax, f"Eigenvalue idx", "Eigenvalue")
        plt.legend()

        plot_path = os.path.join(f"{dir_exp_folder}/"
                                 f"kernel_eigenvals_plot.png")
        fig.savefig(plot_path)
        plt.close()
