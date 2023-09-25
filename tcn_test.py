# Neural Net
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import warnings #for standard scaler supression

# Data Processing
import pickle
import numpy as np, os, sys
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

from tcn import *
from data_processing import *

# Retrieve entire model with saved state 
FILE = "Sepsis_TCN_model.pth"
model = torch.load(FILE)
model = model.float()

#------------#
# Fetch data #
#------------#

data_dir = '../files/challenge-2019/1.0.0/training/training_setA'

# The maximum padded sequence length 
max_sequence_length = 100

test_dataset = SepsisDataset(data_dir, is_train = False, max_sequence_length = max_sequence_length)

# Initialize train and test dataloaders
batch_size = 32

# Load the test/train dataset with DataLoader
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load the scaler from a file
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#------#
# Test #
#------#

criterion = nn.BCELoss()

# Evaluate the model on the test set
# Fit the scaler on the training data
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress the specific warning
    with torch.no_grad():
        test_predictions = []
        test_labels = []
        running_loss = 0.0

        for idx, [variable_names, batch] in enumerate(test_loader):
            
            # Flatten batch tensor to 2dims
            #if len(batch.shape) == 3:
            #    batch = batch.view(-1, batch.size(2))

            # Extract batch_X (all columns except the last one) and batch_y (last column)
            batch_X = batch[:, :, :-1].clone().detach()
            
            # Flatten batch tensor to 2dims for scaling
            batch_X_flat = batch_X.view(-1, batch_X.size(2))

            # Scale the flattened tensor (assuming you've defined scaler)
            batch_X_scaled = torch.tensor(scaler.transform(batch_X_flat))
        
            # Calculate the original batch size
            original_batch_size = batch_X.size(0)

            # Calculate the new batch size after flattening
            new_batch_size = batch_X_flat.size(0)

            # Calculate the number of time steps (columns) in the original shape
            num_time_steps = batch_X.size(2)

            # Reshape batch_X_scaled back to its original shape
            batch_X_scaled = batch_X_scaled.view(original_batch_size, new_batch_size // original_batch_size, num_time_steps).float()
            batch_y = batch[:, :, -1].clone().detach().float()  # Assuming the last column contains the labels
            
            # Check if any target values are NaN in batch_y
            padding_mask = ~torch.isnan(batch_y)

            # Zeroing out NaN values as in training
            batch_X_scaled[torch.isnan(batch_X_scaled)] = 0

            # Forward pass through the model
            outputs = model(batch_X_scaled)

            # Select rows from outputs and batch_y where there are no NaNs
            outputs = outputs[padding_mask]
            batch_y = batch_y[padding_mask]

            # Calculate BCE
            loss = criterion(outputs, batch_y)

            # Append predictions and labels
            labels = batch_y.view(-1, 1)
            threshold = 0.4
            predicted = outputs.view(-1, 1)
            predicted = (predicted >= threshold).float()

            test_labels.extend([lab.item() for lab in labels])
            test_predictions.extend([pred_lab.item() for pred_lab in predicted])
            running_loss += loss.item()

            if (idx+1) % 20 == 0:
                print(f'Test Batch [{idx+1}/{len(test_loader)}], Loss: {loss.item()}')
        
        # Calculate the average loss=
        mean_loss = running_loss / len(test_loader)

        # Number of observations for target classes
        no_obs = len(test_labels)
        no_obs_sepsis = len([1 for lab in test_labels if lab == 1])

        # Calculate balanced accuracy for the test set
        balanced_accuracy = balanced_accuracy_score(test_labels, test_predictions)

        # Compute confusion matrix
        cm = confusion_matrix(test_labels, test_predictions)

        # Print the confusion matrix
        print("Confusion Matrix:")
        print("  Actual\Predicted   |    0    |    1   |")
        print("_____________________|_________|________|")
        print(f"          0          |  {cm[0,0]:^5} |  {cm[0,1]:^5} |")
        print("_____________________|_________|________|")
        print(f"          1          |  {cm[1,0]:^5}  |  {cm[1,1]:^5} |")
        print("_____________________|_________|________|")

        print(f'Test Balanced Accuracy: {balanced_accuracy}, Test Mean Loss: {mean_loss}, No. of Sepsis diagnoses out of all obs.: {no_obs_sepsis}/{no_obs}')


#----------------------#
# Results of TCN test: #
#----------------------#
#Confusion Matrix:
#   Actual\Predicted   |    0    |    1   |
# _____________________|_________|________|
#           0          |  149712 |   418  |
# _____________________|_________|________|
#           1          |  1785   |   997  |
# _____________________|_________|________|
# Test Balanced Accuracy: 0.6777955079716917, Test Mean Loss: 0.054629570946417516, No. of Sepsis diagnoses out of all obs.: 2782/152912