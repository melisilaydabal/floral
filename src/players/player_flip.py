# players/player_flip.py
"""
Player_flip
Given  the model (decision boundary), generates the randomized top-k attack
"""
import os
import torch
import numpy as np
import pandas as pd
import pickle
import random
from src.players.player_model import *
from src.models.nn import *
# from src.models.svm_baseline import *
from src.models.svm_opt import *
from src.utils.metrics import *
from src.utils.utils import *


LIST_BINARY_CLASS_DATASETS = ['moon', 'imdb', 'mnist_1vs7']

class Player_Flip():

    def __init__(self, config, seed):
        self.attack_type = config.get('player_flip').get('attack_type')
        # Flip budget: number of labels that can be flipped
        self.budget = config.get('player_flip').get('flip_budget')
        # Flip_set_budget: size of the flip set that will be considered (& averaged)
        self.set_budget = config.get('player_flip').get('flip_set_budget')
        self.attack_model = None  # If a specific attacker model is used

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_player_model(self, model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def _dump_flipped_data(self, config, data_y_flipped):
        if config.get('player_flip').get('is_set_flip'):
            df = pd.DataFrame(data_y_flipped)
        else:
            df = pd.DataFrame(np.array(data_y_flipped))
        data_path = os.path.join(config.get('data').get('dir_dump_data'),
                                 f"{config.get('data').get('dataset')}"
                                 f"_dataset"
                                 f"_flipped_labels"
                                 f"_isperturbed_{config.get('training').get('is_perturbed')}.xlsx")
        df.to_excel(data_path, index=False)

    def get_top_k_points(self, config, model, data_x, data_y):
        satisfying_indices = list(model.sv_indices)
        combined_list = list(zip(satisfying_indices, list(model.sv_alphas)))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)
        satisfying_indices = [item[0] for item in sorted_combined_list]
        satisfying_indices = satisfying_indices[:min(len(satisfying_indices), 6250)]
        top_k_points_x = data_x[satisfying_indices]
        top_k_points_y = data_y[satisfying_indices]
        return satisfying_indices, top_k_points_x, top_k_points_y


    def randomized_sv_flip(self, config, model, data_x, data_y):
        """ Generate randomized top-k attack"""
        list_state_flips = []
        list_data_y_flipped = []

        if config.get('data').get('dataset') in LIST_BINARY_CLASS_DATASETS:
            satisfying_indices = list(model.sv_indices)
            combined_list = list(zip(satisfying_indices, list(model.sv_alphas)))
            sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)
            satisfying_indices = [item[0] for item in sorted_combined_list]
            satisfying_indices = satisfying_indices[:min(len(satisfying_indices), self.budget * 2)]
            subsets = [random.sample(satisfying_indices, k=min(len(satisfying_indices), self.budget)) for _ in
                                      range(self.set_budget)]
            for selected_indices in subsets:
                data_y_flipped = data_y
                state_flip = [1 if i in selected_indices else 0 for i in range(len(data_y))]
                list_state_flips.append(state_flip)
                flipped_idx = [index for index, value in enumerate(state_flip) if value == 1]
                if config.get('model').get('name') in ['nn']:
                    # For NN, we use 0-1 labels.
                    data_y_flipped = [1 - x if i in flipped_idx else x for i, x in enumerate(data_y_flipped)]
                else:
                    # For SVM, we use -1,+1 labels.
                    data_y_flipped = [-x if i in flipped_idx else x for i, x in enumerate(data_y_flipped)]
                list_data_y_flipped.append(data_y_flipped)
        else:
            state_flip = []
        # self._dump_flipped_data(config, list_data_y_flipped)
        return torch.tensor(list_data_y_flipped), list_state_flips, subsets
