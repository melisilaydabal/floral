# models/roberta.py
"""
Fine-tune a RoBERTa model on the IMDb dataset with adversarial training.
Includes offline preprocessing of the dataset, label flipping to create adversarial examples.
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hinge_loss
from scipy.special import expit  # Sigmoid function to convert logits to probabilities
from noisy_trainer import *
import evaluate

# Default values
DEF_SEED = 42
DEF_LABEL_FLIP_RATE = 0.05
DEF_NUM_EPOCHS = 10
DEF_TEST_SIZE = 0.2
DEF_LOGGING_STEPS = 100
FLIP_RATES = [0.005, 0.01, 0.05, 0.1, 0.25]
DEF_DIR_LOAD_DATA = './dataset/PATH_TO_PT_FILE_FOR_IMDB_DATA'
DEF_DIR_OUTPUT = './workbench'
MODEL_START_CHECKPOINT = 'roberta-base'
warmup_steps = 0
saving_steps = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="seed: int", default=DEF_SEED)
    parser.add_argument("--num_epochs", type=int, help="number of training epochs: int", default=DEF_NUM_EPOCHS)
    parser.add_argument("--logging_steps", type=int, help="number of logging steps (per-X-step log): int", default=DEF_LOGGING_STEPS)
    parser.add_argument("--test_size", type=float, help="test dataset size: float", default=DEF_TEST_SIZE)
    parser.add_argument("--label_flip_rate", type=float, help="label flip rate to create adversarial training dataset: float", default=DEF_LABEL_FLIP_RATE)
    parser.add_argument("--batch_size", type=float, help="batch_size: float", default=16)
    parser.add_argument("--dir_load_data", type=str, help="path of dir to load data: str", default=DEF_DIR_LOAD_DATA)
    parser.add_argument("--lr", type=float, help="learning rate: float", default=2e-5)
    parser.add_argument("--dir_output", type=str, help="path of dir to output model: str", default=DEF_DIR_OUTPUT)

    args = parser.parse_args()
    return args

def log_config_w_wandb(arglist):
    return vars(arglist)

# Parse arguments
arglist = parseargs()

num_epochs = arglist.num_epochs
logging_steps = arglist.logging_steps
test_size = arglist.test_size
label_flip_rate = arglist.label_flip_rate
dir_load_data = arglist.dir_load_data
batch_size = arglist.batch_size
lr = arglist.lr
dir_output = arglist.dir_output

# set seed
torch.manual_seed(arglist.seed)
np.random.seed(arglist.seed)
random.seed(arglist.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(arglist.seed)

# Logging: Save model inputs and hyperparameters
config = log_config_w_wandb(arglist)

# Randomly flip labels
def flip_labels(labels, flip_rate=label_flip_rate):
    num_flips = int(len(labels) * flip_rate)
    flip_indices = random.sample(range(len(labels)), num_flips)
    flipped_labels = copy.deepcopy(labels)
    for idx in flip_indices:
        flipped_labels[idx] = 1 - labels[idx]
    return flipped_labels

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained(MODEL_START_CHECKPOINT)

model = RobertaForSequenceClassification.from_pretrained(MODEL_START_CHECKPOINT, num_labels=2)
model.to(device)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# # ----------PREPROCESSING THE DATASET
#
# # Load the IMDb dataset
# dataset = load_dataset('imdb')
#
# # Tokenize the dataset
# def preprocess_function(examples):
#     return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
#
# tokenized_datasets = dataset.map(preprocess_function, batched=True)
#
# # Split the training data into training and validation datasets
# train_val_split = tokenized_datasets['train'].train_test_split(test_size=test_size)
# train_dataset = train_val_split['train']
# val_test_split = train_val_split['test'].train_test_split(test_size=0.5)
# val_dataset = val_test_split['train']
# test_dataset = val_test_split['test']
#
# train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
#
# with open('./dataset/imdb/train.pt', 'wb') as f:
#     pickle.dump(train_dataset, f)
# with open('./dataset/imdb/val.pt', 'wb') as f:
#     pickle.dump(val_dataset, f)
# with open('./dataset/imdb/test.pt', 'wb') as f:
#     pickle.dump(test_dataset, f)

# # ------CREATE ADV FROM LOAD SAVED DATASETS
# with open('./dataset/imdb/train.pt', 'rb') as data:
#     train_dataset = pickle.load(data)
# with open('./dataset/imdb/val.pt', 'rb') as data:
#     val_dataset = pickle.load(data)
# with open('./dataset/imdb/test.pt', 'rb') as data:
#     test_dataset = pickle.load(data)
#
#
# for label_flip_rate in FLIP_RATES:
#     # Flip labels in the datasets to create adversarial versions
#     train_adv_labels = flip_labels(train_dataset['label'])
#     val_adv_labels = flip_labels(val_dataset['label'])
#     test_adv_labels = flip_labels(test_dataset['label'])
#
#     # Define a function to update the labels
#     def update_labels(examples, idx):
#         return {'label': train_adv_labels[idx]}
#     def update_val_labels(examples, idx):
#         return {'label': val_adv_labels[idx]}
#     def update_test_labels(examples, idx):
#         return {'label': test_adv_labels[idx]}
#
#
#     # Create new datasets with flipped labels
#     train_dataset_adv = copy.deepcopy(train_dataset)
#     # print(train_dataset_adv)
#     train_dataset_adv = train_dataset_adv.map(update_labels, with_indices=True)
#     val_dataset_adv = copy.deepcopy(val_dataset)
#     val_dataset_adv = val_dataset_adv.map(update_val_labels, with_indices=True)
#     test_dataset_adv = copy.deepcopy(test_dataset)
#     test_dataset_adv = test_dataset_adv.map(update_test_labels, with_indices=True)
#
#     train_dataset_adv.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
#     val_dataset_adv.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
#     test_dataset_adv.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
#
#     with open(f'./dataset/imdb/train_adv_flip{label_flip_rate}.pt', 'wb') as f:
#         pickle.dump(train_dataset_adv, f)
#     with open(f'./dataset/imdb/val_adv_flip{label_flip_rate}.pt', 'wb') as f:
#         pickle.dump(val_dataset_adv, f)
#     with open(f'./dataset/imdb/test_adv_flip{label_flip_rate}.pt', 'wb') as f:
#         pickle.dump(test_dataset_adv, f)
# ----------------------------------------


# LOAD SAVED DATASETS
train_dataset = torch.load(os.path.join(dir_load_data, 'train_dataset.pt'))
val_dataset = torch.load(os.path.join(dir_load_data, 'valid_dataset.pt'))
test_dataset = torch.load(os.path.join(dir_load_data, 'test_dataset.pt'))
train_dataset_adv = torch.load(os.path.join(dir_load_data, f'train_adv_flip{label_flip_rate}_dataset.pt'))
val_dataset_adv = torch.load(os.path.join(dir_load_data, f'valid_adv_flip{label_flip_rate}_dataset.pt'))
test_dataset_adv = torch.load(os.path.join(dir_load_data, f'test_adv_flip{label_flip_rate}_dataset.pt'))


# # -------
# PREPROCESS DATASET IF NOT DONE BEFORE
# # Tokenize the dataset
# def preprocess_function(examples):
#     return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
#
# train_dataset = train_dataset.map(preprocess_function, batched=True)
# val_dataset = val_dataset.map(preprocess_function, batched=True)
# test_dataset = test_dataset.map(preprocess_function, batched=True)
# train_dataset_adv = train_dataset_adv.map(preprocess_function, batched=True)
# val_dataset_adv = val_dataset_adv.map(preprocess_function, batched=True)
# test_dataset_adv = test_dataset_adv.map(preprocess_function, batched=True)
#
# train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# train_dataset_adv.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# val_dataset_adv.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# test_dataset_adv.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# print(train_dataset)
#


# # --------------------------
# # FOR NOISY LABEL TRAINING OF ROBERTA W CONFUSION MATRIX
# batch_size = batch_size  # Adjust the batch size as needed
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# test_adv_loader = DataLoader(test_dataset_adv, batch_size=batch_size, shuffle=False)
#
# # def get_features_labels(data_loader):
# #     input_ids, attention_masks, labels = [], [], []
# #     # Iterate over the DataLoader
# #     for batch in data_loader:
# #         batch_input_ids = batch['input_ids'].to(device)
# #         batch_attention_masks = batch['attention_mask'].to(device)
# #         batch_labels = batch['label'].to(device)
# #
# #         # # Assuming each batch is a tuple (features, labels)
# #         # batch_features, batch_labels = batch
# #
# #         # Append to lists
# #         input_ids.append(batch_input_ids)
# #         attention_masks.append(batch_attention_masks)
# #         labels.append(batch_labels)
# #     # Concatenate all batches to form the complete dataset
# #     input_ids = torch.cat(input_ids)
# #     attention_masks = torch.cat(attention_masks)
# #     labels = torch.cat(labels)
# #     return input_ids, attention_masks, labels
# #
# # test_input_ids, test_attention_masks, test_labels = get_features_labels(test_loader)
# # test_adv_input_ids, test_adv_attention_masks, test_adv_labels = get_features_labels(test_adv_loader)
#
# # FOR NOISY LABEL TRAINING OF ROBERTA W CONFUSION MATRIX
# test_labels = test_dataset['label']
# test_adv_labels = test_dataset_adv['label']
# # Assuming a hypothetical function `estimate_noise_matrix` that estimates the noise matrix
# def estimate_noise_matrix(clean_labels, noisy_labels, num_classes=2):
#     conf_matrix = confusion_matrix(clean_labels, noisy_labels, labels=range(num_classes))
#     noise_matrix = conf_matrix.astype(np.float32) / conf_matrix.sum(axis=1)[:, np.newaxis]
#     return noise_matrix
# noise_matrix = estimate_noise_matrix(test_labels, test_adv_labels, num_classes=2)
# # conf_matrix = confusion_matrix(true_labels, noisy_labels, labels=range(num_classes))
# # noise_matrix = conf_matrix.astype(np.float32) / conf_matrix.sum(axis=1)[:, np.newaxis]
#
# def loss_fn(logits, labels, noise_matrix):
#     log_probs = F.log_softmax(logits, dim=1)
#     noise_adjusted_targets = torch.matmul(labels.float(), torch.tensor(noise_matrix).float().to(labels.device))
#     loss = -torch.sum(noise_adjusted_targets * log_probs, dim=1)
#     return loss.mean()
#
# # -------------------

metric = evaluate.load("accuracy")

# Define a function to compute metrics during training and evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=preds, references=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    sigmoid = expit
    criterion = torch.nn.BCEWithLogitsLoss()


    probs = sigmoid(preds)
    loss = criterion(torch.tensor(probs), torch.tensor(labels, dtype=torch.float32)).item()

    labels_binary = np.where(labels == 1, 1, -1)  # convert labels to -1/1 for hinge loss
    hinge_loss_value = hinge_loss(labels_binary, logits[:, 1])

    return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': loss,
            'hinge_loss': hinge_loss_value
            # 'eval_loss': loss
        }

# ###########################################################
# # -----TRAIN CLEAN------
# wandb.init(
#     project="roberta-finetune",
#     name='roberta_train_epochs{}'.format(num_epochs),
#     tags=["roberta"],
#     group="bert",
#     reinit=True
# )
#
# # Define the training arguments
# training_args = TrainingArguments(
#     output_dir='./workbench/imdb/roberta_train',
#     run_name='roberta_train_epochs{}'.format(num_epochs),
#     eval_strategy='steps',
#     save_strategy='steps',
#     save_steps=saving_steps,
#     logging_steps=logging_steps,
#     warmup_steps=warmup_steps, # warmup with low learning rate at the beginning of training
#     learning_rate=lr,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=num_epochs,
#     weight_decay=0.01,
#     report_to="wandb"
# )
# #
# # Define the Trainer for clean data
# trainer_clean = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset={"train": train_dataset, "val": val_dataset, "test": test_dataset,
#                   "train_adv": train_dataset_adv, "val_adv": val_dataset_adv, "test_adv": test_dataset_adv},
#    # eval_dataset={"train": train_dataset, "val": val_dataset, "test": test_dataset,
#    #               "train_adv": train_dataset_adv},
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )
# #
# # # # Define the NOISY LABEL Trainer for clean data
# # # trainer_clean = NoisyLabelTrainer(
# # #     noise_matrix=noise_matrix,
# # #     model=model,
# # #     args=training_args,
# # #     train_dataset=train_dataset,
# # #     eval_dataset={"train": train_dataset, "val": val_dataset, "test": test_dataset,
# # #                   "train_adv": train_dataset_adv, "val_adv": val_dataset_adv, "test_adv": test_dataset_adv},
# # #    # eval_dataset={"train": train_dataset, "val": val_dataset, "test": test_dataset,
# # #    #               "train_adv": train_dataset_adv},
# # #     tokenizer=tokenizer,
# # #     compute_metrics=compute_metrics,
# # # )
# #
# # Train the clean model
# trainer_clean.train()
#
# # Evaluate the clean model on validation set
# eval_clean = trainer_clean.evaluate(eval_dataset=val_dataset)
# clean_accuracy_val = eval_clean['eval_accuracy']
# clean_loss_val = eval_clean['eval_loss']
# trainer_clean.log_metrics('eval', {'accuracy': clean_accuracy_val, 'loss': clean_loss_val})
#
# # Evaluate the clean model on test set
# eval_clean_test = trainer_clean.evaluate(eval_dataset=test_dataset)
# clean_accuracy_test = eval_clean_test['eval_accuracy']
# clean_loss_test = eval_clean_test['eval_loss']
# trainer_clean.log_metrics('test', {'accuracy': clean_accuracy_test, 'loss': clean_loss_test})
#
# # Evaluate the clean model on robust validation set
# eval_robust_clean = trainer_clean.evaluate(eval_dataset=val_dataset_adv)
# clean_robust_accuracy_val = eval_robust_clean['eval_accuracy']
# clean_robust_loss_val = eval_robust_clean['eval_loss']
# trainer_clean.log_metrics('eval', {'accuracy': clean_robust_accuracy_val, 'loss': clean_robust_loss_val})
#
# # Evaluate the clean model on robust test set
# eval_robust_clean = trainer_clean.evaluate(eval_dataset=test_dataset_adv)
# clean_robust_accuracy_test = eval_robust_clean['eval_accuracy']
# clean_robust_loss_test = eval_robust_clean['eval_loss']
# trainer_clean.log_metrics('test', {'accuracy': clean_robust_accuracy_test, 'loss': clean_robust_loss_test})
#
# del model
# wandb.finish()
#
# # Print results
# print(f"Clean Model Validation Accuracy: {clean_accuracy_val}")
# print(f"Clean Model Validation Loss: {clean_loss_val}")
# print(f"Clean Model Test Accuracy: {clean_accuracy_test}")
# print(f"Clean Model Test Loss: {clean_loss_test}")
# print(f"Clean Model Robust Validation Accuracy: {clean_robust_accuracy_val}")
# print(f"Clean Model Robust Validation Loss: {clean_robust_loss_val}")
# print(f"Clean Model Robust Test Accuracy: {clean_robust_accuracy_test}")
# print(f"Clean Model Robust Test Loss: {clean_robust_loss_test}")
#
# #########################################################################

# -----------TRAINING ON ADVERSARIAL DATASET
wandb.init(
    project="roberta-finetune",
    config=config,
    name='roberta_train_adv_targeted_epochs{}_flip{}_warmup{}'.format(num_epochs, label_flip_rate, warmup_steps),
    tags=["roberta"],
    group="bert",
    reinit=True
)
# wandb.init(
#     project="roberta-finetune",
#     config=config,
#     name='roberta_train_adv_noisylabel_epochs{}_flip{}_warmup{}'.format(num_epochs, label_flip_rate, warmup_steps),
#     tags=["roberta"],
#     group="bert",
#     reinit=True
# )

model = RobertaForSequenceClassification.from_pretrained(MODEL_START_CHECKPOINT, num_labels=2)
model.to(device)

# Define the training arguments
# training_adv_args = TrainingArguments(
#     output_dir='./workbench/imdb/roberta_train_adv_targeted_flip{}'.format(label_flip_rate),
#     run_name='roberta_train_adv_epochs{}_flip{}'.format(num_epochs, label_flip_rate),
#     eval_strategy='steps',
#     save_strategy='steps',
# save_steps=saving_steps,
#     logging_steps=logging_steps,
#     warmup_steps=logging_steps, # warmup with low learning rate at the beginning of training
#     learning_rate=lr,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=num_epochs,
#     weight_decay=0.01,
#     report_to="wandb"
# )


training_adv_args = TrainingArguments(
    output_dir=dir_output + '/imdb/roberta_train_adv_targeted_flip{}_warmup{}_seed{}'.format(label_flip_rate, warmup_steps, arglist.seed),
    run_name='roberta_train_adv_epochs{}_flip{}_warmup{}'.format(num_epochs, label_flip_rate, warmup_steps),
    eval_strategy='steps',
    save_strategy='steps',
    save_steps=saving_steps,
    logging_steps=logging_steps,
    warmup_steps=warmup_steps, # warmup with low learning rate at the beginning of training
    learning_rate=lr,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    report_to="wandb"
)

print(f'Using train dataset with {label_flip_rate}: {train_dataset_adv}')
# Define the Trainer for adversarial data
trainer_adv = Trainer(
    model=model,
    args=training_adv_args,
    train_dataset=train_dataset_adv,
    eval_dataset={"train": train_dataset, "test": test_dataset},
    # eval_dataset={"train": train_dataset, "val": val_dataset, "test": test_dataset,
    #               "train_adv": train_dataset_adv},
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# # Define the training arguments
# training_adv_args = TrainingArguments(
#     output_dir='./workbench/imdb/roberta_train_noisylabel/roberta_train_adv_flip{}'.format(label_flip_rate),
#     run_name='roberta_train_adv_noisylabel_epochs{}_flip{}'.format(num_epochs, label_flip_rate),
#     eval_strategy='steps',
#     save_strategy='steps',
#     save_steps=saving_steps,
#     logging_steps=logging_steps,
#     warmup_steps=logging_steps, # warmup with low learning rate at the beginning of training
#     learning_rate=lr,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=num_epochs,
#     weight_decay=0.01,
#     report_to="wandb"
# )
#
# # Define the NOISY LABEL Trainer for adversarial data
# trainer_adv = NoisyLabelTrainer(
#     noise_matrix=noise_matrix,
#     model=model,
#     args=training_adv_args,
#     train_dataset=train_dataset_adv,
#     eval_dataset={"train": train_dataset, "val": val_dataset, "test": test_dataset,
#                   "train_adv": train_dataset_adv, "val_adv": val_dataset_adv, "test_adv": test_dataset_adv},
#     # eval_dataset={"train": train_dataset, "val": val_dataset, "test": test_dataset,
#     #               "train_adv": train_dataset_adv},
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

# Train the adversarial model
trainer_adv.train()

# Evaluate the adversarial model on validation set
eval_adv = trainer_adv.evaluate(eval_dataset=val_dataset)
adv_accuracy_val = eval_adv['eval_accuracy']
adv_loss_val = eval_adv['eval_loss']
trainer_adv.log_metrics('eval', {'accuracy': adv_accuracy_val, 'loss': adv_loss_val})

# Evaluate the adversarial model on test set
eval_adv_test = trainer_adv.evaluate(eval_dataset=test_dataset)
adv_accuracy_test = eval_adv_test['eval_accuracy']
adv_loss_test = eval_adv_test['eval_loss']
trainer_adv.log_metrics('test', {'accuracy': adv_accuracy_test, 'loss': adv_loss_test})

# Evaluate the adversarial model on robust validation set
eval_robust_adv = trainer_adv.evaluate(eval_dataset=val_dataset_adv)
adv_robust_accuracy_val = eval_robust_adv['eval_accuracy']
adv_robust_loss_val = eval_robust_adv['eval_loss']
trainer_adv.log_metrics('eval', {'accuracy': adv_robust_accuracy_val, 'loss': adv_robust_loss_val})

# Evaluate the adversarial model on robust test set
eval_robust_adv_test = trainer_adv.evaluate(eval_dataset=test_dataset_adv)
adv_robust_accuracy_test = eval_robust_adv_test['eval_accuracy']
adv_robust_loss_test = eval_robust_adv_test['eval_loss']
trainer_adv.log_metrics('test', {'accuracy': adv_robust_accuracy_test, 'loss': adv_robust_loss_test})


print(f"Adversarial Model Validation Accuracy: {adv_accuracy_val}")
print(f"Adversarial Model Validation Loss: {adv_loss_val}")
print(f"Adversarial Model Test Accuracy: {adv_accuracy_test}")
print(f"Adversarial Model Test Loss: {adv_loss_test}")
print(f"Adversarial Model Robust Validation Accuracy: {adv_robust_accuracy_val}")
print(f"Adversarial Model Robust Validation Loss: {adv_robust_loss_val}")
print(f"Adversarial Model Robust Test Accuracy: {adv_robust_accuracy_test}")
print(f"Adversarial Model Robust Test Loss: {adv_robust_loss_test}")
wandb.finish()