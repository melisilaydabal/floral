import os
import torch
import numpy as np
import random
import copy
import pickle
import wandb
import argparse
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.special import expit  # Sigmoid function to convert logits to probabilities
from noisy_trainer import *

# Default values
SEEDS = [42]
DEF_LABEL_FLIP_RATE = 0.05
DEF_NUM_EPOCHS = 3
DEF_TEST_SIZE = 0.2
DEF_LOGGING_STEPS = 50
FLIP_RATES = [0.1, 0.25, 0.3, 0.35, 0.4]
# FLIP_RATES = [0.01]
DEF_DIR_LOAD_DATA = './dataset/PATH_TO_PT_FILE_FOR_IMDB_DATA'
DEF_DIR_DUMP_DATA = './dataset/imdb/'
MODEL_CHECKPOINT = './workbench/imdb/PATH_TO_ROBERTA_CKPT'
# MODEL_CHECKPOINT = 'roberta-base'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, help="number of training epochs: int", default=DEF_NUM_EPOCHS)
    parser.add_argument("--logging_steps", type=int, help="number of logging steps (per-X-step log): int", default=DEF_LOGGING_STEPS)
    parser.add_argument("--test_size", type=float, help="test dataset size: float", default=DEF_TEST_SIZE)
    parser.add_argument("--label_flip_rate", type=float, help="label flip rate to create adversarial training dataset: float", default=DEF_LABEL_FLIP_RATE)
    parser.add_argument("--batch_size", type=float, help="batch_size: float", default=16)
    parser.add_argument("--dir_load_data", type=str, help="path of dir to load data: str", default=DEF_DIR_LOAD_DATA)
    parser.add_argument("--dir_dump_data", type=str, help="path of dir to load data: str", default=DEF_DIR_DUMP_DATA)

    return parser.parse_args()


arglist = parseargs()
num_epochs = arglist.num_epochs
logging_steps = arglist.logging_steps
test_size = arglist.test_size
label_flip_rate = arglist.label_flip_rate
dir_load_data = arglist.dir_load_data
batch_size = arglist.batch_size
dir_dump_data = arglist.dir_dump_data

dataset = load_dataset('imdb')

# ------CREATE ADV FROM LOAD SAVED DATASETS
train_dataset = torch.load(os.path.join(dir_load_data, 'train_dataset.pt'))
# val_dataset = torch.load(os.path.join(dir_load_data, 'valid_dataset.pt'))
# test_dataset = torch.load(os.path.join(dir_load_data, 'test_dataset.pt'))


# # Function to convert datasets to PyTorch tensors
# def dataset_to_tensors(dataset):
#     input_ids = torch.tensor(dataset['input_ids'])
#     attention_mask = torch.tensor(dataset['attention_mask'])
#     labels = torch.tensor(dataset['label'])
#     return TensorDataset(input_ids, attention_mask, labels)
#
# train_dataset = dataset_to_tensors(train_dataset)
# val_dataset = dataset_to_tensors(val_dataset)
# test_dataset = dataset_to_tensors(test_dataset)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Function to get the gradients of the loss with respect to the embeddings
def get_gradients(data_loader, model, device):
    model.eval()
    gradients = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        embeddings = model.roberta.embeddings(input_ids)
        embeddings.retain_grad()
        embeddings = embeddings.clone().detach().requires_grad_(True)

        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        grads = embeddings.grad.data.norm(2, dim=1)
        gradients.append(grads.cpu().numpy())

    gradients = np.concatenate(gradients)
    gradients = [np.mean(row) for row in gradients]
    return gradients


# ----For ROBERTA fine-tuned on ADV datasets
# for label_flip_rate in FLIP_RATES:
#     for seed in SEEDS:
#         # Load the fine-tuned model
#         tokenizer = RobertaTokenizer.from_pretrained(MODEL_CHECKPOINT.format(label_flip_rate, seed))
#         model = RobertaForSequenceClassification.from_pretrained(MODEL_CHECKPOINT.format(label_flip_rate, seed))
#         model.to(device)
#
#         k = 0.25
#         print(f"Processing label flip rate and seed: {label_flip_rate} and {seed}")
#         # Get the gradients for the training set
#         gradients = get_gradients(train_loader, model, device)
#         # Identify the top k most influential data points
#         num_flips = int(len(train_dataset) * k)
#         print(f'number of flips: {num_flips}')
#         top_k_indices = np.argsort(gradients)[-num_flips:]
#         print(f"top_k_indices with size {top_k_indices.size} : {top_k_indices}")
#

#         dump_result_path = './workbench/imdb'
#         dump_file = f"roberta_{label_flip_rate}_seed{seed}_influential_pts.out"
#         with open(os.path.join(dump_result_path, dump_file), "w") as file:
#             file.write("######\n")
#             file.write(f"Point PointIndex\n")
#             for i, metric_value in enumerate(top_k_indices, start=1):
#                 file.write(f"{i} {str(metric_value)}\n")
#             # file.write(','.join(map(str, top_k_indices)))

# # ----For ROBERTA fine-tuned on clean dataset --> the one we used to generate adversarial datasets
# for seed in SEEDS:
#     # Load the fine-tuned model
#     tokenizer = RobertaTokenizer.from_pretrained(MODEL_CHECKPOINT.format(seed))
#     model = RobertaForSequenceClassification.from_pretrained(MODEL_CHECKPOINT.format(seed))
#     model.to(device)
#
#     k = 0.25
#     print(f"Processing seed: {seed}")
#     # Get the gradients for the training set
#     gradients = get_gradients(train_loader, model, device)
#     # Identify the top k most influential data points
#     num_flips = int(len(train_dataset) * k)
#     print(f'number of flips: {num_flips}')
#     top_k_indices = np.argsort(gradients)[-num_flips:]
#     print(f"top_k_indices with size {top_k_indices.size} : {top_k_indices}")
#
#
#     dump_result_path = './workbench/imdb'
#     dump_file = f"roberta_clean_seed{seed}_influential_pts.out"
#     with open(os.path.join(dump_result_path, dump_file), "w") as file:
#         file.write("######\n")
#         file.write(f"Point PointIndex\n")
#         for i, metric_value in enumerate(top_k_indices, start=1):
#             file.write(f"{i} {str(metric_value)}\n")
#         # file.write(','.join(map(str, top_k_indices)))

# ----------------------------------
# GET THE ORIGINAL POINTS (IN TEXT FORM)
seed = 42
tokenizer = RobertaTokenizer.from_pretrained(MODEL_CHECKPOINT.format(seed))

i = 12847
print(f"Datapoint {i}: {train_dataset[i]}")
original_text = tokenizer.decode(train_dataset[i]['input_ids'], skip_special_tokens=True)
print(original_text)
