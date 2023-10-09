import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import pickle
import multiprocessing
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
from tcn import *
from data_processing_r import *

# Suppress warnings from the StandardScaler
warnings.filterwarnings("ignore", category=RuntimeWarning)

# File path for the saved model
FILE = "Sepsis_TCN_model.pth"

def load_scaler(scaler_file):
    with open(scaler_file, 'rb') as file:
        return pickle.load(file)

def scale_input(input, scaler):
    input_flat = input.view(-1, input.size(2))
    input_scaled = torch.tensor(scaler.transform(input_flat))
    original_batch_size = input.size(0)
    new_batch_size = input_flat.size(0)
    num_time_steps = input.size(2)
    input_scaled = input_scaled.view(original_batch_size, new_batch_size // original_batch_size, num_time_steps).float()
    input_scaled[torch.isnan(input_scaled)] = 0
    return input_scaled

def evaluate_model(model, test_loader, criterion, scaler, threshold=0.4):
    test_predictions = []
    test_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for idx, [input, targets] in enumerate(test_loader):
            input_scaled = scale_input(input, scaler)
            padding_mask = ~torch.isnan(targets)
            input_scaled[torch.isnan(input_scaled)] = 0

            outputs = model(input_scaled)
            outputs = outputs[padding_mask]
            targets = targets[padding_mask]

            loss = criterion(outputs, targets)
            labels = targets.view(-1, 1)
            logits = torch.sigmoid(outputs)
            predicted = logits.view(-1, 1)
            predicted = (predicted >= threshold).float()

            test_labels.extend([lab.item() for lab in labels])
            test_predictions.extend([pred_lab.item() for pred_lab in predicted])
            running_loss += loss.item()

            if (idx + 1) % 20 == 0:
                print(f'Test Batch [{idx + 1}/{len(test_loader)}], Loss: {loss.item()}')

    mean_loss = running_loss / len(test_loader)
    no_obs = len(test_labels)
    no_obs_sepsis = len([1 for lab in test_labels if lab == 1])
    balanced_accuracy = balanced_accuracy_score(test_labels, test_predictions)
    auc_score = roc_auc_score(test_labels, test_predictions)
    cm = confusion_matrix(test_labels, test_predictions)

    print("Confusion Matrix:")
    print("  Actual\Predicted   |    0    |    1   |")
    print("_____________________|_________|________|")
    print(f"          0          |  {cm[0, 0]:^5} |  {cm[0, 1]:^5} |")
    print("_____________________|_________|________|")
    print(f"          1          |  {cm[1, 0]:^5}  |  {cm[1, 1]:^5} |")
    print("_____________________|_________|________|")

    print(f'Test AUC: {auc_score}, Test Balanced Accuracy: {balanced_accuracy}, Test Mean Loss: {mean_loss}, No. of Sepsis diagnoses out of all obs.: {no_obs_sepsis}/{no_obs}')


def main():
    # Load the saved model
    model = torch.load(FILE).float()

    # Data directory
    data_dir = '../files/challenge-2019/1.0.0/training'

    max_sequence_length = 100

    test_dataset = SepsisDataset(data_dir, is_train=False, max_sequence_length=max_sequence_length)

    batch_size = 128
    num_workers = round(multiprocessing.cpu_count() / 2)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    scaler = load_scaler('scaler.pkl')

    # Define loss function
    pos_weight = torch.tensor(4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    evaluate_model(model, test_loader, criterion, scaler)


if __name__ == '__main__':
    main()
