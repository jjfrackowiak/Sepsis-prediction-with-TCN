import os
import numpy as np, os, sys
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

import numpy as np
from scipy.interpolate import interp1d

def process_psv_file(file_path):
    # Read the .psv file
    with open(file_path, 'r') as psv_file:
        lines = psv_file.readlines()

    # Extract variable names from the header row (first row)
    variable_names = lines[0].strip().split('|')
    
    # Process the data for variable measurements (subsequent rows)
    time_series_data = []
    for line in lines[1:]:
        values = line.strip().split('|')
        # Replace 'nan' with np.nan in the values
        values = [np.nan if value == 'nan' else float(value) for value in values]
        time_series_data.append(values)
    
    time_series_data = np.array(time_series_data, dtype=np.float32)  # Convert to NumPy array

    # Perform linear interpolation for missing values in each column
    for col_index in range(time_series_data.shape[1]):
        col_values = time_series_data[:, col_index]
        indices = np.arange(len(col_values))
        mask = np.isnan(col_values)
        
        # Check if the data is nearly random by calculating the standard deviation
        std_dev = np.std(col_values[~mask])
        
        if np.any(mask):  # You can adjust the threshold (0.1) as needed
            x_interp = indices[mask]
            y_interp = col_values[~mask]
            if y_interp.shape[0] == 1:  # If there's only one non-NaN value
                    col_values[mask] = np.nan  # Impute all NaNs with that single value
                    continue
            elif y_interp.shape[0]== 0:
                continue
            interp_func = interp1d(indices[~mask], y_interp, kind='linear', fill_value='extrapolate')
            col_values[mask] = interp_func(x_interp)
    
        # Check if there are still NaNs after interpolation
        if np.isnan(col_values).any():
            # Replace remaining NaNs with the mean of each column
            mean = np.nanmean(col_values)
            nan_indices = np.isnan(col_values)
            col_values[nan_indices] = mean
            print(col_values)
    return variable_names, time_series_data



data_dir = '/Users/mac/Desktop/Data Science - Studies/TCN_Project/physionet.org/files/challenge-2019/1.0.0/training/training_setA'

'''
# tests
i = 0
for psv in os.listdir(data_dir):
    print(i)
    if i > 100:
        break
    if '.psv' in psv:
        psv_path = os.path.join(data_dir, psv)
        process_psv_file(psv_path)
        i+=1
'''


class SepsisDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)  # List of .psv files in the data directory
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = os.path.join(self.data_dir, self.file_list[idx])
        variable_names, time_series_data = process_psv_file(file_name)
        return variable_names, time_series_data

dataset = SepsisDataset(data_dir)

# Iterate through the dataset one sample at a time
for idx in range(len(dataset)):
    variable_names, time_series_data = dataset[idx]
    # Now you can work with variable_names and time_series_data for each sample
    print(f"Sample {idx + 1}:")
    print("Variable Names:", variable_names)
    print("Time Series Data:", time_series_data)
    if idx == 100:
        break


sys.exit()
data_loader = DataLoader(dataset, batch_size=batch_size)

data_loader = iter(data_loader)

features, labels = next(data_loader)



print(features, labels)