import numpy as np, os, sys
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import warnings

def list_files_recursive(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

def process_psv_file(file_path):
    '''
    Function preprocessing data for each of the 
    patients' .psv files.

    1. Reads .psv file
    2. Filles columns with only one value with NaN's
    3. Performs data imputation based on scipy's interp1d
    4. If step 3 does not work, performs mean imputation
    '''
    # Read the .psv file
    if '.psv' not in file_path:
        return
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


class SepsisDataset(Dataset):
    def __init__(self, data_dir, is_train=True, max_sequence_length=None):
        self.data_dir = data_dir
        self.file_list = [file for file in list_files_recursive(data_dir) if '.psv' in file]
        self.is_train = is_train
        self.max_sequence_length = max_sequence_length

        # Split the dataset into train and test sets
        self.train_files, self.test_files = train_test_split(self.file_list,
                                                             test_size=0.2,
                                                             random_state=42)

        # Use either train or test files based on is_train flag
        self.file_list = self.train_files if is_train else self.test_files

        self.real_max_seq = 0

        # Preload all data into a NumPy array
        self.data = np.empty((len(self.file_list),), dtype=object)
        for i, file_name in tqdm(enumerate(self.file_list)):
            #Ignore runtime warnings
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            variable_names, time_series_data = process_psv_file(file_name)
            self.data[i] = (variable_names, time_series_data)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        variable_names, time_series_data = self.data[idx]

        # Convert time_series_data to a PyTorch tensor
        time_series_data = torch.tensor(time_series_data, dtype=torch.float32)

        if self.max_sequence_length is not None:
            if time_series_data.shape[0] > self.max_sequence_length:
                # Truncate sequences to max_sequence_length
                time_series_data = time_series_data[:self.max_sequence_length, :]
            elif len(time_series_data) < self.max_sequence_length:
                # Pad sequences to max_sequence_length with the last row
                last_row = time_series_data[-1].unsqueeze(0)  # Get the last row and add an extra dimension
                padding = last_row.repeat(self.max_sequence_length - len(time_series_data), 1)  # Repeat the last row
                time_series_data = torch.cat((time_series_data, padding))

        if 1 in time_series_data[:, -1]:
            # Replace zeros in the last column with ones if it is a Sepsis case
            # NN will anticipate it
            time_series_data[:, -1] = 1

        return time_series_data[:, :-1], time_series_data[:, -1]

