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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
from tcn import *
from data_processing import *

# Suppress warnings from the StandardScaler
warnings.filterwarnings("ignore", category=RuntimeWarning)

'''
Note:

Here we can calculate all the things required for final demonstration
And check if it is not creating predictions for padded data (rather not because I have used padding mask)
Manipulate threshold to get the best trade-off
'''


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

def evaluate_model(model, test_loader, criterion, scaler, threshold=0.3):
    test_predicted_labels = []
    test_probs = []
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
            predicted_labs = (predicted >= threshold).float()

            test_labels.extend([lab.item() for lab in labels])
            test_predicted_labels.extend([pred_lab.item() for pred_lab in predicted_labs])
            test_probs.extend([prob.item() for prob in predicted])

            running_loss += loss.item()

            if (idx + 1) % 20 == 0:
                print(f'Test Batch [{idx + 1}/{len(test_loader)}], Loss: {loss.item()}')

    mean_loss = running_loss / len(test_loader)
    no_obs = len(test_labels)
    no_obs_sepsis = len([1 for lab in test_labels if lab == 1])
    balanced_accuracy = balanced_accuracy_score(test_labels, test_predicted_labels)
    auc_score = roc_auc_score(test_labels, test_probs)

    # Calculate classification report
    classification_rep = classification_report(test_labels, test_predicted_labels, target_names=['Class 0', 'Class 1'])
    confusion_m = confusion_matrix(test_labels, test_predicted_labels)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)

    # Create a ROC curve plot
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (TCN Model)')
    plt.legend(loc='lower right')

    # Save the plot to a file (e.g., in PNG format)
    plt.savefig('roc_curve.png')

    # Print classification report
    print("Classification Report:")
    print(classification_rep)
    print("Confusion Matrix:")
    print(confusion_m)

    print(f'Test AUC: {auc_score}, Test Balanced Accuracy: {balanced_accuracy}, Test Mean Loss: {mean_loss}, No. of Sepsis diagnoses out of all obs.: {no_obs_sepsis}/{no_obs}')


def main():

    # File path for the saved model
    FILE = "Sepsis_TCN_model.pth"

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
    pos_weight = torch.tensor(10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    evaluate_model(model, test_loader, criterion, scaler)


if __name__ == '__main__':
    main()
