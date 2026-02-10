# Code cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

# Code cell
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Warning: NVIDIA GPU detected.')
    else:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Warning: Apple MPS detected.')
        else:
            device = torch.device('cpu')
            print('Warning: No NVIDIA GPU detected. Performance may be suboptimal.')

    print(f'Selected device: {device}')

    return device

device = get_device()

# Code cell
def preprocess_data(df):
    # Remove bus 1 data (load_P_bus_1, load_Q_bus_1, voltage_bus_1, theta_bus_1)
    df = df.drop(columns=[col for col in df.columns if col.endswith('bus_1')])

    # Input: load_P and load_Q from all buses except bus 1
    load_P_columns = [col for col in df.columns if 'load_P_bus' in col]
    load_Q_columns = [col for col in df.columns if 'load_Q_bus' in col]
    
    # Output: voltage and theta from all buses except bus 1
    voltage_columns = [col for col in df.columns if 'voltage_bus' in col]
    theta_columns = [col for col in df.columns if 'theta_bus' in col]

    # Stack inputs (load_P, load_Q) and outputs (voltage, theta)
    X = np.hstack([df[load_P_columns].values, df[load_Q_columns].values])
    y = np.hstack([df[voltage_columns].values, df[theta_columns].values])
    
    return X, y

# Code cell
file_path = 'case8_results.csv'
data = pd.read_csv(file_path)
X, y = preprocess_data(data)

# Normalize the inputs and outputs
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

# Display the shape of the processed data to ensure no bus_11 columns were dropped
print(X_train.shape, y_train.shape)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Code cell
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through layers with activation
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Code cell
# Set input and output size based on the dataset
input_size = X_train.shape[1]
output_size = y_train.shape[1]

# Initialize the model, loss function, and optimizer
model = NN(input_size, output_size)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_history = []

# Training loop
epochs = 50000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if (epoch+1) % 1000 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.10f}')

# Code cell
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), loss_history, label='Training Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (Log Scale)')
plt.title('Log-Scaled Loss History During Training')
plt.legend()
plt.grid(True)
plt.show()

# Code cell
# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)

# Inverse transform the predictions to recover original values
y_pred_original = scaler_y.inverse_transform(y_pred.cpu().numpy())
y_test_original = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

mape = np.mean(np.abs((y_pred_original - y_test_original)/y_test_original), axis=0)

print(f'MAPE mean: {np.mean(mape)}')
print(f'MAPE max: {np.max(mape)}')

# Code cell


