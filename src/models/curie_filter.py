import torch
import copy
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from scipy.stats import zscore

LIST_LARGE_DATASETS = ['imdb']
LIST_SMALL_DATASETS = ['moon']

def curie_filtering(config, seed, data, labels, weight=0.5, count=10, theta=1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if config.get('data').get('dataset') in LIST_LARGE_DATASETS:
        # Step 1: Dimensionality Reduction using PCA
        pca = PCA(n_components=0.95)  # Keep 95% of the variance
        pca_data = pca.fit_transform(data)
    else:
        pca_data = copy.deepcopy(data)

    # Step 2: Clustering using DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Parameters can be tuned
    clusters = dbscan.fit_predict(pca_data)

    # Step 3: Distance Calculation
    distances = np.zeros(len(data))
    for idx, point in enumerate(data):
        augmented_point = np.append(point, labels[idx] * weight)
        point_cluster = clusters[idx]
        cluster_points = [i for i in range(len(data)) if clusters[i] == point_cluster]

        if len(cluster_points) < count:
            count = len(cluster_points)

        sampled_indices = random.sample(cluster_points, count)
        total_distance = 0

        for sample_idx in sampled_indices:
            augmented_sample = np.append(data[sample_idx], labels[sample_idx] * weight)
            distance = euclidean(augmented_point, augmented_sample)
            total_distance += distance

        if len(cluster_points) > 0:
            distances[idx] = total_distance / len(cluster_points)

    standardized_distances = zscore(distances)

    filtered_data = []
    filtered_labels = []

    for idx, dist in enumerate(standardized_distances):
        if dist <= theta:
            filtered_data.append(data[idx])
            filtered_labels.append(labels[idx])
    filtered_data = torch.from_numpy(np.array(filtered_data))
    filtered_labels = torch.from_numpy(np.array(filtered_labels))
    return filtered_data, filtered_labels




