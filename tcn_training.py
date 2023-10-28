import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import multiprocessing
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tcn import *
from data_processing import *
from torch.utils.tensorboard import SummaryWriter

# GPU support
def setup_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Suppresing warnings related to missing values
def worker_init_fn(worker_id):
    warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load and preprocess data
def load_and_preprocess_data(data_dir, max_sequence_length, batch_size, num_workers=0):
    train_dataset = SepsisDataset(data_dir, is_train=True, max_sequence_length=max_sequence_length)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn)

    # Fit and save StandardScaler
    scaler = StandardScaler()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for idx, [input, targets] in enumerate(train_loader):
        if len(input.shape) == 3:
            input = input.view(-1, input.size(2))
        scaler.partial_fit(input)
        if (idx+1) % 20 == 0:
            print(f'Training StandardScaler: Batch [{idx+1}/{len(train_loader)}]')

    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    return train_loader, scaler

# Scale input data
def scale_inputs(input, scaler):
    input_flat = input.view(-1, input.size(2))
    input_scaled = torch.tensor(scaler.transform(input_flat)).float()
    original_batch_size = input.size(0)
    new_batch_size = input_flat.size(0)
    num_time_steps = input.size(2)
    input_scaled = input_scaled.view(original_batch_size, new_batch_size // original_batch_size, num_time_steps)
    return input_scaled

# Train the TCN model
def train_model(writer_train, writer_val, model, train_loader, val_loader, scaler, device, num_epochs, initial_lr):
    print("Initialising training...")
    pos_weight = torch.tensor(10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    for i in range(num_epochs):
        running_loss = 0.0
        true_labels = []
        predicted_labels = []
        probs = []

        # Training loop
        model.train()  # Set the model to training mode
        for idx, [input, targets] in enumerate(train_loader):
            input_scaled = scale_inputs(input, scaler)
            padding_mask = ~torch.isnan(targets)
            nan_mask = torch.isnan(input_scaled)
            has_nan = torch.any(nan_mask)

            if has_nan:
                # Zeroing out remaining NaN values
                input_scaled[torch.isnan(input_scaled)] = 0
                outputs = model(input_scaled).float()
                outputs = outputs[padding_mask].to(device)
                targets = targets[padding_mask].to(device)
                loss = criterion(outputs, targets)
            else:
                outputs = model(input_scaled).float()
                outputs = outputs[padding_mask].to(device)
                targets = targets[padding_mask].to(device)

                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            labels = targets.view(-1, 1)
            threshold = 0.3
            logits = torch.sigmoid(outputs)
            predicted = logits.view(-1, 1)
            predicted_classes = (predicted >= threshold).float()

            true_labels.extend([lab.item() for lab in labels])
            predicted_labels.extend([pred_lab.item() for pred_lab in predicted_classes])
            probs.extend([prob.item() for prob in predicted])
            running_loss += loss.item()

            if (idx+1) % 20 == 0:
                print(f'Training Batch [{idx+1}/{len(train_loader)}], Mean Loss: {loss.item()}')

        epoch_loss = running_loss / len(train_loader)
        no_obs = len(true_labels)
        no_obs_sepsis = len([1 for lab in true_labels if lab == 1])

        balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, probs)

        writer_train.add_scalar('Balanced accuracy', balanced_accuracy, i)
        writer_train.add_scalar('AUC Score', roc_auc, i)
        writer_train.add_scalar('Loss', epoch_loss, i)

        print(f'Epoch [{i+1}/{num_epochs}], Balanced Accuracy (Train): {balanced_accuracy}, AUC Score (Train): {roc_auc}, Loss: {epoch_loss}, No. of Sepsis diagnoses out of all obs.: {no_obs_sepsis}/{no_obs}')

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        val_true_labels = []
        val_predicted_labels = []
        val_predicted_no_thresh = []

        with torch.no_grad():
            for idx, [val_input, val_targets] in enumerate(val_loader):
                val_input_scaled = scale_inputs(val_input, scaler)
                val_padding_mask = ~torch.isnan(val_targets)
                val_input_scaled[torch.isnan(val_input_scaled)] = 0

                val_outputs = model(val_input_scaled)
                val_outputs = val_outputs[val_padding_mask]
                val_targets = val_targets[val_padding_mask]

                val_loss = criterion(val_outputs, val_targets)
                val_labels = val_targets.view(-1, 1)
                val_logits = torch.sigmoid(val_outputs)
                val_predicted = val_logits.view(-1, 1)
                val_predicted_thresh = (val_predicted >= threshold).float() # Threshold equal to training

                val_true_labels.extend([lab.item() for lab in val_labels])
                val_predicted_no_thresh.extend([pred_lab.item() for pred_lab in val_predicted])
                val_predicted_labels.extend([pred_lab.item() for pred_lab in val_predicted_thresh])
                val_running_loss += loss.item()

            if (idx + 1) % 20 == 0:
                print(f'Validation Batch [{idx + 1}/{len(val_loader)}], Loss: {val_loss.item()}')

        val_epoch_loss = val_running_loss / len(val_loader)
        val_no_obs = len(val_true_labels)
        val_no_obs_sepsis = len([1 for lab in val_true_labels if lab == 1])

        val_balanced_accuracy = balanced_accuracy_score(val_true_labels, val_predicted_labels)
        val_roc_auc = roc_auc_score(val_true_labels, val_predicted_no_thresh)

        writer_val.add_scalar('Balanced accuracy', val_balanced_accuracy, i)
        writer_val.add_scalar('AUC Score', val_roc_auc, i)
        writer_val.add_scalar('Loss', val_epoch_loss, i)

        print(f'Epoch [{i+1}/{num_epochs}], Balanced Accuracy (Validation): {val_balanced_accuracy}, AUC Score (Validation): {val_roc_auc}, Loss (Validation): {val_epoch_loss}, No. of Sepsis diagnoses out of all obs. (Validation): {val_no_obs_sepsis}/{val_no_obs}')

    return model  # Return the trained model

# Save the trained model
def save_model(model):
    FILE = "Sepsis_TCN_model.pth"
    torch.save(model, FILE)

def main():
    # Neural Net
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Tensorboard writer
    writer_train, writer_val = SummaryWriter('runs/TCN/train'), SummaryWriter('runs/TCN/val')

    # GPU support (deprecated)
    device = setup_device()

    # Data Processing
    data_dir = '../files/challenge-2019/1.0.0/training'
    max_sequence_length = 100
    batch_size = 128
    num_workers = round(multiprocessing.cpu_count() / 2)

    train_loader, scaler = load_and_preprocess_data(data_dir, max_sequence_length, batch_size=batch_size, num_workers=num_workers)

    # Create val dataset/loader for validation
    val_dataset = SepsisDataset(data_dir, is_train=False, max_sequence_length=max_sequence_length)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn)

    # Training
    input_size = 100
    output_size = 100
    num_channels = [150, 150]
    kernel_size = 3

    model = TCNBC(input_size=input_size, output_size=output_size, num_channels=num_channels, kernel_size=kernel_size, dropout=0.2)
    model = model.to(device).float()

    initial_lr = 0.01
    num_epochs = 10

    model = train_model(writer_train, writer_val, model, train_loader, val_loader, scaler, device, num_epochs, initial_lr)

    # Close the SummaryWriters when done and save the model
    writer_train.flush()
    writer_val.flush()
    writer_train.close()
    writer_val.close()
    save_model(model)

if __name__ == '__main__':
    main()
