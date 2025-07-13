import torch
import copy
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode

LIST_LARGE_DATASETS = ['imdb']
LIST_SMALL_DATASETS = ['moon']

def label_sanitize(config, seed, X_train, y_train, k=20, eta=0.5):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_samples = len(X_train)
    y_train_relabelled = copy.deepcopy(y_train)

    while True:
        changes_made = False

        nn_model = NearestNeighbors(n_neighbors=k)
        nn_model.fit(X_train)

        for i in range(num_samples):
            # Find k nearest neighbors for the i-th sample
            distances, indices = nn_model.kneighbors([X_train[i]], n_neighbors=k, return_distance=True)
            neighbors_labels = y_train_relabelled[indices[0]]

            common_label, count = mode(neighbors_labels)
            confidence = count / k

            # If the confidence is greater than or equal to the threshold η, relabel the point
            if confidence >= eta and common_label != y_train_relabelled[i]:
                y_train_relabelled[i] = common_label
                changes_made = True
        if not changes_made:
            break
    y_train_relabelled = torch.from_numpy(np.array(y_train_relabelled))
    return y_train_relabelled