# Neural Net
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import warnings #for standard scaler supression

# Data Processing
import pickle
import numpy as np, os, sys
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

from tcn import *
from data_processing import *

#See Tensorboard after training by typing into terminal: "tensorboard --logdir=runs --port=6007"
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/TCN') # for comparison with different l

# Enable GPU support if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-----------------#
# Data processing #
#-----------------#
data_dir = '../files/challenge-2019/1.0.0/training/training_setA'

# Set the maximum sequence length
max_sequence_length = 100

train_dataset = SepsisDataset(data_dir, is_train = True, max_sequence_length = max_sequence_length)
test_dataset = SepsisDataset(data_dir, is_train = False, max_sequence_length = max_sequence_length)

# Initialize train and test dataloaders
batch_size = 32

# Load the test/train dataset with DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Fit a StandardScaler on the training data
scaler = StandardScaler()

# Fit the scaler on the training data
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = RuntimeWarning)  # Training on NaN-padded data
    for idx, [variable_names, batch] in enumerate(train_loader):

        # Flatten batch tensor to 2dims
        if len(batch.shape) == 3:
            batch = batch.view(-1, batch.size(2))

        # Extract batch_X with features
        batch_X = batch[:, :-1].clone().detach()

        # Fit the scaler on batch_X
        scaler.partial_fit(batch_X) 
        if (idx+1)%20 == 0:
            print(f'Training StandardScaler: Batch [{idx+1}/{len(train_loader)}]')

# Save the scaler to a file
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

#----------#
# Training #
#----------#

# Initialize the model
input_size = 100
output_size = 100
num_channels = [150, 150] #Default is [150, 150, 150, 150]
kernel_size = 3
model = TCNBC(input_size = input_size, 
              output_size = output_size, 
              num_channels = num_channels,
              kernel_size = kernel_size,
              dropout=0.2)
model = model.to(device).float()

# Define the loss function and optimizer
criterion = nn.BCELoss()
initial_lr = 0.01  # Initial learning rate
new_lr = initial_lr * 0.3  # Decrease learning rate by 0.7
l2_reg = 1e-4  # Adjust regularization strength

optimizer = optim.Adam(model.parameters(), lr=initial_lr)

# Training loop
num_epochs = 15

# Iterate through the training dataset for training
print("Initialising TCN Training...")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = RuntimeWarning)  # Training on NaN-padded data
    for i in range(num_epochs):
        running_loss = 0.0
        true_labels = []
        predicted_labels = []

        # Iterate over batches for training
        for idx, [variable_names, batch] in enumerate(train_loader):

            # Extract batch_X (all columns except the last one) and batch_y (last column)
            batch_X = batch[:, :, :-1].clone().detach()
            
            # Flatten batch tensor to 2dims for scaling
            batch_X_scaled = batch_X_flat = batch_X.view(-1, batch_X.size(2))
            
            # Scale the flattened tensor 
            batch_X_scaled = torch.tensor(scaler.transform(batch_X_flat)).float()

            # Calculate the original batch size
            original_batch_size = batch_X.size(0)

            # Calculate the new batch size after flattening
            new_batch_size = batch_X_flat.size(0)

            # Calculate the number of time steps (columns) in the original shape
            num_time_steps = batch_X.size(2)

            # Reshape batch_X_scaled back to its original shape
            batch_X_scaled = batch_X_scaled.view(original_batch_size, new_batch_size // original_batch_size, num_time_steps)
            batch_y = batch[:, :, -1].clone().detach() # Assuming the last column contains the labels

            # Check if any target values are NaN in batch_y
            padding_mask = ~torch.isnan(batch_y)

            # Check for NaN values in the batch_X (before replacing with 0)
            nan_mask = torch.isnan(batch_X_scaled)
            has_nan = torch.any(nan_mask)

            # Zeroing-out NaN values which could not be mean imputed or interpolated
            if has_nan:
                batch_X_scaled[torch.isnan(batch_X_scaled)] = 0

                # Reduce learning rate and apply regularization
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                # Apply L2 regularization
                l2_loss = 0.0
                for param in model.parameters():
                    l2_loss += torch.norm(param, p=2)

                outputs = model(batch_X_scaled).float()

                # Select rows from outputs and batch_y where there are no NaNs
                outputs = outputs[padding_mask].to(device)
                batch_y = batch_y[padding_mask].to(device)

                # Regularised loss
                loss = criterion(outputs, batch_y) + l2_reg * l2_loss
                
            else:
                outputs = model(batch_X_scaled).float()

                # Select rows from outputs and batch_y where there are no NaNs
                outputs = outputs[padding_mask].to(device)
                batch_y = batch_y[padding_mask].to(device)

                # Use the normal learning rate and no regularization
                if np.nan in outputs or np.nan in batch_y:
                    continue
                
                loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()

            

            optimizer.step()

            labels = batch_y.view(-1, 1)
            threshold = 0.5
            predicted = outputs.view(-1, 1)
            predicted = (predicted >= threshold).float()
        
            # Collect true labels and predicted labels for the epoch
            true_labels.extend([lab.item() for lab in labels])
            predicted_labels.extend([pred_lab.item() for pred_lab in predicted])
            running_loss += loss.item()

            # Print loss every 20th step
            if (idx+1) % 20 == 0:
                print(f'Training Batch [{idx+1}/{len(train_loader)}], Mean Loss: {loss.item()}')

        # Calculate the average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Number of observations for target classes
        no_obs = len(true_labels)
        no_obs_sepsis = len([1 for lab in true_labels if lab == 1])

        # Calculate balanced accuracy for the epoch
        balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

        # Log the balanced accuracy and loss for the epoch
        writer.add_scalar('balanced accuracy', balanced_accuracy, i)
        writer.add_scalar('training loss', epoch_loss, i)

        print(f'Epoch [{i+1}/{num_epochs}], Balanced Accuracy: {balanced_accuracy}, Loss: {epoch_loss}, No. of Sepsis diagnoses out of all obs.: {no_obs_sepsis}/{no_obs}')

    # Loggin the Mean epoch balanced accuracy and loss 
    print(f'Mean Epoch Balanced Accuracy: {balanced_accuracy}, Mean Epoch Loss: {epoch_loss}')

    #Save entire trained model:
    FILE = "Sepsis_TCN_model.pth"
    torch.save(model, FILE)

#--------------------------#
# Results of TCN training: #
#--------------------------#