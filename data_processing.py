import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

# Function to list files recursively in a directory
def list_files_recursive(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

# Function to process a .psv file
def process_psv_file(file_path):
    if '.psv' not in file_path:
        return None, None

    # Read the .psv file
    with open(file_path, 'r') as psv_file:
        lines = psv_file.readlines()

    # Extract variable names from the header row (first row)
    variable_names = lines[0].strip().split('|')

    # Process the data for variable measurements (subsequent rows)
    time_series_data = []
    for line in lines[1:]:
        values = line.strip().split('|')
        values = [np.nan if value == 'nan' else float(value) for value in values]
        time_series_data.append(values)

    time_series_data = np.array(time_series_data, dtype=np.float32)

    # Perform linear interpolation for missing values in each column
    for col_index in range(time_series_data.shape[1]):
        col_values = time_series_data[:, col_index]
        indices = np.arange(len(col_values))
        mask = np.isnan(col_values)

        # Check if the data is nearly random by calculating the standard deviation
        std_dev = np.std(col_values[~mask])

        if np.any(mask):
            x_interp = indices[mask]
            y_interp = col_values[~mask]

            if y_interp.shape[0] == 1:
                col_values[mask] = np.nan
                continue

            elif y_interp.shape[0] == 0:
                continue

            # Interpolate missing values
            interp_func = interp1d(indices[~mask], y_interp, kind='linear', fill_value='extrapolate')
            col_values[mask] = interp_func(x_interp)

        # Check if there are still NaNs after interpolation
        if np.isnan(col_values).any():
            # Replace remaining NaNs with the mean of each column
            mean = np.nanmean(col_values)
            nan_indices = np.isnan(col_values)
            col_values[nan_indices] = mean

    return variable_names, time_series_data

# Custom dataset class
class SepsisDataset(Dataset):
    def __init__(self, data_dir, is_train=True, max_sequence_length=None):
        self.data_dir = data_dir
        self.file_list = [file for file in list_files_recursive(data_dir) if '.psv' in file]
        self.is_train = is_train
        self.max_sequence_length = max_sequence_length

        # Split the dataset into train and test sets
        self.train_files, self.test_files = train_test_split(self.file_list, test_size=0.2, random_state=42)
        self.file_list = self.train_files if is_train else self.test_files
        self.real_max_seq = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        variable_names, time_series_data = process_psv_file(file_name)
        time_series_data = torch.tensor(time_series_data, dtype=torch.float32)

        if self.max_sequence_length is not None:
            if time_series_data.shape[0] > self.max_sequence_length:
                # Truncate sequences to max_sequence_length
                time_series_data = time_series_data[:self.max_sequence_length, :]
            elif len(time_series_data) < self.max_sequence_length:
                # Calculate the number of rows to pad with NaN
                num_padding_rows = self.max_sequence_length - len(time_series_data)
                padding_shape = (num_padding_rows, time_series_data.shape[1])
                
                # Create a padding tensor with NaN values
                padding = torch.tensor(np.nan * np.ones(padding_shape), dtype=torch.float32)
                
                # Concatenate the original data with the padding
                time_series_data = torch.cat((padding, time_series_data))

        return time_series_data[:, :-1], time_series_data[:, -1]
