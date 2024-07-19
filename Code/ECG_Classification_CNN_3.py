import os
import pandas as pd
import scipy.io
import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                             f1_score, accuracy_score, roc_auc_score, average_precision_score)
from torch.utils.data import DataLoader, TensorDataset, random_split

def read_mat_files(folder_path, label):
    """Reads .mat files from a folder and returns data and labels."""
    data_list = []
    label_list = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".mat"):
                file_path = os.path.join(folder_path, filename)
                mat_data = scipy.io.loadmat(file_path)
                data_list.append(mat_data)
                label_list.append(label)
    except Exception as e:
        print(f"Error reading .mat files: {e}")
    return data_list, label_list

def combine_data(folder_paths):
    """Combines data from multiple folders into a single DataFrame and Series."""
    all_data, all_labels, signals_array = [], [], []
    for label, folder_path in enumerate(folder_paths):
        data_list, labels = read_mat_files(folder_path, label)
        all_data.extend(data_list)
        all_labels.extend(labels)
        for filename in os.listdir(folder_path):
            if filename.endswith(".mat"):
                file_path = os.path.join(folder_path, filename)
                try:
                    record = wfdb.rdrecord(file_path.replace(".mat", ""))
                    for signal in record.p_signal.T:
                        signals_array.append(signal)
                except Exception as e:
                    print(f"Error reading WFDB record: {e}")
    dataFrame = pd.DataFrame(signals_array)
    labelSeries = pd.Series(all_labels, name='label')
    return dataFrame, labelSeries

def save_dataframe_to_csv(dataFrame, filename='combined_ECG_Data.csv'):
    """Saves a DataFrame to a CSV file."""
    try:
        dataFrame.to_csv(filename, index=False)
        print("CSV file created successfully.")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def split_data(dataFrame, sub_timewindow=1000):
    """Splits data into training and testing sets."""
    train = dataFrame.iloc[:2000]
    test = dataFrame.iloc[2001:]
    X_train = train.iloc[:, :sub_timewindow].values
    X_test = test.iloc[:, :sub_timewindow].values
    Y_train = train['label'].values
    Y_test = test['label'].values
    return X_train, X_test, Y_train, Y_test

def create_dataloaders(X_train, Y_train, X_test, Y_test, batch_size=16):
    """Creates DataLoader objects for training, validation, and testing sets."""
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    train_size = int(0.8 * len(X_train_tensor))
    val_size = len(X_train_tensor) - train_size
    train_dataset, val_dataset = random_split(TensorDataset(X_train_tensor, Y_train_tensor), [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

class ECGModel(nn.Module):
    """CNN model for ECG classification."""
    def __init__(self):
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10)
        self.pool = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.global_avg_pool(x).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    """Trains and validates the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
    return train_losses, val_losses

def plot_losses(train_losses, val_losses, epochs):
    """Plots training and validation losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, test_loader, criterion):
    """Evaluates the model on the test set and prints performance metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    y_pred = np.array(y_pred).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_binary))
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUROC: {auroc:.4f}')
    print(f'AUPRC: {auprc:.4f}')

if __name__ == "__main__":
    folder_paths = [
        '/Users/atorN/Dropbox/ECGs_training2017/Class_A',
        '/Users/atorN/Dropbox/ECGs_training2017/Class_N'
    ]
    dataFrame, labelSeries = combine_data(folder_paths)
    dataFrame = pd.concat([labelSeries, dataFrame], axis=1)
    dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)
    save_dataframe_to_csv(dataFrame)
    
    sub_timewindow = 1000
    X_train, X_test, Y_train, Y_test = split_data(dataFrame, sub_timewindow)
    train_loader, val_loader, test_loader = create_dataloaders(X_train, Y_train, X_test, Y_test)

    model = ECGModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    train_losses, val_losses = train_and_validate(model)