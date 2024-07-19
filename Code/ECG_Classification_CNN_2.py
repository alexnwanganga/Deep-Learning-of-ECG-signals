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
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, average_precision_score
)
from torch.utils.data import DataLoader, TensorDataset, random_split

#--------------------------Read and Combine Functions----------------------------------------------------------

def read_mat_files(folder_path, label):
    """
    Reads .mat files from a specified folder and assigns a label to each.
    
    Args:
        folder_path (str): Path to the folder containing .mat files.
        label (int): Label to assign to each .mat file.
        
    Returns:
        tuple: Two lists, one with data and the other with labels.
    """
    data_list = [] 
    label_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".mat"):
            file_path = os.path.join(folder_path, filename)
            mat_data = scipy.io.loadmat(file_path)
            data_list.append(mat_data)
            label_list.append(label)

    return data_list, label_list

def combine_data(folder_paths):
    """
    Combines data from multiple folders and returns as a DataFrame.
    
    Args:
        folder_paths (list): List of folder paths to read data from.
        
    Returns:
        tuple: DataFrame of combined data and Series of labels.
    """
    all_data = []
    all_labels = []
    signals_array = []

    for label, folder_path in enumerate(folder_paths):
        data_list, labels = read_mat_files(folder_path, label)
        all_data.extend(data_list)
        all_labels.extend(labels)

        for filename in os.listdir(folder_path):
            if filename.endswith(".mat"):
                file_path = os.path.join(folder_path, filename)
                record = wfdb.rdrecord(file_path.replace(".mat", ""))
                
                for signal in record.p_signal.T:
                    signals_array.append(signal)

    dataFrame = pd.DataFrame(signals_array)
    labelSeries = pd.Series(all_labels, name='label')

    return dataFrame, labelSeries

#---------------------------------Defining Database Structure---------------------------------------------------

folder_paths = [
    '/Users/atorN/Dropbox/ECGs_training2017/Class_A',
    '/Users/atorN/Dropbox/ECGs_training2017/Class_N'  # Removed Folder O for binary classification
]

dataFrame, labelSeries = combine_data(folder_paths)

print(f"DataFrame shape: {dataFrame.shape}")
print(f"Label Series shape: {labelSeries.shape}")

dataFrame = pd.concat([labelSeries, dataFrame], axis=1)
dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

print(f"Combined and randomized DataFrame shape: {dataFrame.shape}")

try:
    dataFrame.to_csv('combined_ECG_Data.csv', index=False)
    print("CSV file created successfully.")
except Exception as e:
    print(f"Error saving CSV file: {e}")

#---------------------------------Designating Training and Testing Sets---------------------------------------------------

train = dataFrame.iloc[:2000] 
test = dataFrame.iloc[2001:]

sub_timewindow = 1000

print("Shape of train DataFrame:", train.shape)

X_train = train.iloc[:, :sub_timewindow].values
X_test = test.iloc[:, :sub_timewindow].values
Y_train = train['label'].values
Y_test = test['label'].values

print('Train Shape - voltages, label:')
print(X_train.shape, Y_train.shape)
print('Test Shape - voltages, label:')
print(X_test.shape, Y_test.shape)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

val_split = 0.2
train_size = int((1 - val_split) * len(X_train_tensor))
val_size = len(X_train_tensor) - train_size
train_dataset, val_dataset = random_split(TensorDataset(X_train_tensor, Y_train_tensor), [train_size, val_size])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=batch_size, shuffle=False)

#---------------------------------Defining Model---------------------------------------------------

class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 10)
        self.conv2 = nn.Conv1d(16, 32, 10)
        self.pool = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(32, 64, 10)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.global_avg_pool(x).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x

model = ECGModel()
print(model)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#---------------------------------Training and Validation Loop---------------------------------------------------

epochs = 100
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

# ---------------------------------- Plot Training and Validation Loss ----------------------------------

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss', fontsize=20)
plt.legend(fontsize=14)
plt.show()

# ------------------------------------- Evaluation on Test Set -------------------------------------

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
print(confusion_matrix(y_true, y_pred))
