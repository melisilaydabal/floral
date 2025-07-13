# models/roberta_create_adversarial.py
"""
Generate IMDB datasets with adversarial labels, using a pre-trained RoBERTa model.
"""
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
DEF_LABEL_FLIP_RATE = 0.05
DEF_NUM_EPOCHS = 3
DEF_TEST_SIZE = 0.2
DEF_LOGGING_STEPS = 50
FLIP_RATES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.25]
DEF_DIR_LOAD_DATA = './dataset/PATH_TO_PT_FILE_FOR_IMDB_DATA'
DEF_DIR_DUMP_DATA = './dataset/imdb/'
MODEL_CHECKPOINT = 'roberta-base'

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
val_dataset = torch.load(os.path.join(dir_load_data, 'valid_dataset.pt'))
test_dataset = torch.load(os.path.join(dir_load_data, 'test_dataset.pt'))


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
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Randomly flip labels
def flip_labels(labels, flip_rate=label_flip_rate):
    num_flips = int(len(labels) * flip_rate)
    flip_indices = random.sample(range(len(labels)), num_flips)
    print(flip_indices)
    flipped_labels = copy.deepcopy(labels)
    for idx in flip_indices:
        flipped_labels[idx] = 1 - labels[idx]
    return flipped_labels

# Load the fine-tuned model
tokenizer = RobertaTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = RobertaForSequenceClassification.from_pretrained(MODEL_CHECKPOINT)
model.to(device)

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

# Function to flip the labels of the identified data points
def flip_labels_wrt_indices(dataset, indices):
    dataset = copy.deepcopy(dataset)
    input_ids = dataset['input_ids']
    attention_mask = dataset['attention_mask']
    labels = dataset['label']
    flipped_labels = copy.deepcopy(labels)
    for idx in indices:
        flipped_labels[idx] = 1 - labels[idx]
    return flipped_labels


# Define functions to update the labels
    def update_val_labels(examples, idx):
        return {'label': val_adv_labels[idx]}
    def update_test_labels(examples, idx):
        return {'label': test_adv_labels[idx]}

for label_flip_rate in FLIP_RATES:
    print(f"Processing label flip rate: {label_flip_rate}")
    gradients = get_gradients(train_loader, model, device)
    # Identify the top k most influential data points
    num_flips = int(len(train_dataset) * label_flip_rate)
    top_k_indices = np.argsort(gradients)[-num_flips:]

    train_adv_labels = flip_labels_wrt_indices(train_dataset, top_k_indices)
    difference = torch.eq(train_dataset['label'], train_adv_labels)
    num_differing_elements = len(difference)-sum(difference)

    def update_labels(examples, idx):
        return {'label': train_adv_labels[idx]}

    train_dataset_adv = copy.deepcopy(train_dataset)
    train_dataset_adv = train_dataset_adv.map(update_labels, with_indices=True)
    # Save the adversarial dataset
    torch.save(train_dataset_adv, os.path.join(dir_dump_data, f'train_adv_flip{label_flip_rate}_dataset.pt'))

    difference = torch.eq(train_dataset['input_ids'], train_dataset_adv['input_ids'])
    num_differing_elements = len(difference)-sum(difference)

    # Get the gradients for the training set
    gradients = get_gradients(val_loader, model, device)
    # Identify the top k most influential data points
    num_flips = int(len(val_dataset) * label_flip_rate)
    top_k_indices = np.argsort(gradients)[-num_flips:]

    # Flip the labels in the training set
    val_adv_labels = flip_labels_wrt_indices(val_dataset, top_k_indices)
    def update_val_labels(examples, idx):
        return {'label': val_adv_labels[idx]}

    val_dataset_adv = copy.deepcopy(val_dataset)
    val_dataset_adv = val_dataset_adv.map(update_val_labels, with_indices=True)
    # Save the adversarial dataset
    torch.save(val_dataset_adv, os.path.join(dir_dump_data, f'valid_adv_flip{label_flip_rate}_dataset.pt'))


    # Get the gradients for the training set
    gradients = get_gradients(test_loader, model, device)
    # Identify the top k most influential data points
    num_flips = int(len(test_dataset) * label_flip_rate)
    top_k_indices = np.argsort(gradients)[-num_flips:]
    test_adv_labels = flip_labels_wrt_indices(test_dataset, top_k_indices)
    def update_test_labels(examples, idx):
        return {'label': test_adv_labels[idx]}

    test_dataset_adv = copy.deepcopy(test_dataset)
    test_dataset_adv = test_dataset_adv.map(update_test_labels, with_indices=True)
    # Save the adversarial dataset
    torch.save(test_dataset_adv, os.path.join(dir_dump_data, f'test_adv_flip{label_flip_rate}_dataset.pt'))

