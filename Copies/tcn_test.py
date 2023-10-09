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
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np, os, sys
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

from tcn import *
from data_processing import *

# Multiprocessing
import multiprocessing

# Retrieve entire model with saved state 
FILE = "Sepsis_TCN_model.pth"
model = torch.load(FILE)
model = model.float()

#------------#
# Fetch data #
#------------#

data_dir = '../files/challenge-2019/1.0.0/training'

# The maximum padded sequence length 
max_sequence_length = 100

test_dataset = SepsisDataset(data_dir,
                             is_train = False,
                             max_sequence_length = max_sequence_length)

# Initialize train and test dataloaders
batch_size = 128

# Get the number of CPU cores available and divide by 2
num_workers = round(multiprocessing.cpu_count()/2)

# This function will be called when each worker process is initialized
def worker_init_fn(worker_id):
    # Configuring displayed warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load the test/train dataset with DataLoader
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers = num_workers,
                         worker_init_fn = worker_init_fn)

# Load the scaler from a file
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#------#
# Test #
#------#

if __name__ == '__main__':
    #criterion = nn.BCELoss()
    pos_weight = torch.tensor(4) # 5 times more sensitive for positive class
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight) 

    # Evaluate the model on the test set
    # Fit the scaler on the training data
    with torch.no_grad():
        test_predictions = []
        test_labels = []
        running_loss = 0.0

        for idx, [input, targets] in enumerate(test_loader):
            
            # Flatten input tensor to 2dims for scaling
            input_flat = input.view(-1, input.size(2))

            # Scale the flattened tensor (assuming you've defined scaler)
            input_scaled = torch.tensor(scaler.transform(input_flat))
        
            # Calculate the original batch size
            original_batch_size = input.size(0)

            # Calculate the new batch size after flattening
            new_batch_size = input_flat.size(0)

            # Calculate the number of time steps (columns) in the original shape
            num_time_steps = input.size(2)

            # Reshape input_scaled back to its original shape
            input_scaled = input_scaled.view(original_batch_size,
                                                new_batch_size // original_batch_size,
                                                num_time_steps).float()
            
            # Check if any target values are NaN in targets
            padding_mask = ~torch.isnan(targets)

            # Zeroing out NaN values as in training
            input_scaled[torch.isnan(input_scaled)] = 0

            # Forward pass through the model
            outputs = model(input_scaled)

            # Select rows from outputs and targets where there are no NaNs
            outputs = outputs[padding_mask]
            targets = targets[padding_mask]

            # Calculate BCE
            loss = criterion(outputs, targets)

            # Append predictions and labels
            labels = targets.view(-1, 1)
            threshold = 0.4
            logits = torch.sigmoid(outputs)
            predicted = logits.view(-1, 1)
            predicted = (predicted >= threshold).float()

            test_labels.extend([lab.item() for lab in labels])
            test_predictions.extend([pred_lab.item() for pred_lab in predicted])
            running_loss += loss.item()

            #if 1 in labels:
            #    l_p = [(l.item(),p.item()) for l,p in zip(labels,predicted)]
            #    print("Labels,Predictions", l_p)

            if (idx+1) % 20 == 0:
                print(f'Test Batch [{idx+1}/{len(test_loader)}], Loss: {loss.item()}')
        
        # Calculate the average loss=
        mean_loss = running_loss / len(test_loader)

        # Number of observations for target classes
        no_obs = len(test_labels)
        no_obs_sepsis = len([1 for lab in test_labels if lab == 1])

        # Calculate balanced accuracy for the test set
        balanced_accuracy = balanced_accuracy_score(test_labels, test_predictions)

        # Calculate AUC score
        auc_score = roc_auc_score(test_labels, test_predictions)

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

        print(f'Test AUC: {auc_score}, Test Balanced Accuracy: {balanced_accuracy}, Test Mean Loss: {mean_loss}, No. of Sepsis diagnoses out of all obs.: {no_obs_sepsis}/{no_obs}')


#----------------------#
# Results of TCN test: #
#----------------------#
#Confusion Matrix:
#  Actual\Predicted   |    0    |    1   |
#_____________________|_________|________|
#          0          |  636011 |  122289 |
#_____________________|_________|________|
#          1          |  15809  |  32691 |
#_____________________|_________|________|
#Test Balanced Accuracy: 0.7563869643301415, Test Mean Loss: 0.5522358287125826, No. of Sepsis diagnoses out of all obs.: 48500/806800