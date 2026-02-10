# Code cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *

import json  # Add this import for handling JSON files
import matplotlib.pyplot as plt  # Ensure matplotlib is imported for plotting


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
file_path = 'case8_results.csv'
data = pd.read_csv(file_path)
X, y = preprocess_data(data)

# Normalize the inputs and outputs
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# scaler_X = MinMaxScaler()
# scaler_y = MinMaxScaler()

# X_normalized = scaler_X.fit_transform(X)
# y_normalized = scaler_y.fit_transform(y)

# X = scaler_X.fit_transform(X)
# y = scaler_y.fit_transform(y)


# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

X_test, y_test = X_train, y_train


# Display the shape of the processed data to ensure no bus_11 columns were dropped
print(X_train.shape, y_train.shape)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)





class SmallNetworkConv1D(nn.Module):
    def __init__(self, input_dim):
        super(SmallNetworkConv1D, self).__init__()
        # Using kernel_size=3 with padding=1 to maintain sequence length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, padding=0)
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 1, input_dim)
        Returns:
            Tensor of shape (batch_size, 1, input_dim)
        """
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x

# Define the modified ResidualBlock with Conv1D
class ResidualBlockConv1D(nn.Module):
    def __init__(self, input_dim, alpha=0.9):
        super(ResidualBlockConv1D, self).__init__()
        self.alpha = alpha  # Scaling coefficient
        self.residual_function = SmallNetworkConv1D(input_dim)

    def forward(self, x):
        return x + self.alpha * self.residual_function(x)
        # return self.residual_function(x)

    def inverse(self, y):
        # Use fixed-point iteration to find the inverse
        x = y
        for _ in range(100):
            x = y - self.alpha * self.residual_function(x)  # y = x + af(x) --> x = y - af(x)
        return x

# Define the modified ResNet with Conv1D
class ResNetConv1D(nn.Module):
    def __init__(self, input_dim, num_blocks):
        super(ResNetConv1D, self).__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlockConv1D(input_dim) for _ in range(num_blocks)]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 1, input_dim)
        Returns:
            Tensor of shape (batch_size, 1, input_dim)
        """
        x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        x = x.squeeze(1)
        return x

    def inverse(self, y):
        """
        Args:
            y: Tensor of shape (batch_size, 1, input_dim)
        Returns:
            Tensor of shape (batch_size, 1, input_dim)
        """
        y = y.unsqueeze(1)
        for block in reversed(self.blocks):
            y = block.inverse(y)
        y = y.squeeze(1)
        return y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Code cell
# Set input and output size based on the dataset
input_size = X_train.shape[1]
output_size = y_train.shape[1]
print("input_size", input_size)
print("output_size", output_size)
block_num = 3
epochs = 20000
model_name = "conv_resnet_kernel2"

# Initialize the model, loss function, and optimizer
model = ResNetConv1D(input_size, block_num)
print(model)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_history = []

# Training loop

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.10f}')

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
plt.savefig(f"./results/{model_name}_powerflow_loss.png")




# Evaluation, Inversion, MSE Calculations, and JSON Saving (Modified)
# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)  # [batch_size, output_size]
    x_inv = model.inverse(y_pred)  # Inverse from y_pred
    x_inv_real = model.inverse(y_test_tensor)  # Inverse from y

# Inverse transform the predictions to recover original values
# y_pred_original = scaler_y.inverse_transform(y_pred.cpu().numpy())  # for MaxMin scale
# y_test_original = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())
# x_inv_original = scaler_X.inverse_transform(x_inv.cpu().numpy())
# x_inv_real_original = scaler_X.inverse_transform(x_inv_real.cpu().numpy())
# X_test_original = scaler_X.inverse_transform(X_test)


y_pred_original = y_pred.cpu().numpy()  # for MaxMin scale
y_test_original = y_test_tensor.cpu().numpy()
x_inv_original = x_inv.cpu().numpy()
x_inv_real_original =x_inv_real.cpu().numpy()
X_test_original = X_test


# Calculate MSEs and convert them to native Python floats
mse_y = float(np.mean((y_pred_original - y_test_original) ** 2))
mse_x_inv = float(np.mean((x_inv_original - X_test) ** 2))
mse_x_inv_real = float(np.mean((x_inv_real_original - X_test) ** 2))

# Calculate MAPE and convert them to native Python floats
mape_y = float(np.mean(np.abs((y_pred_original - y_test_original) / y_test_original)) * 100)
mape_x_inv = float(np.mean(np.abs((x_inv_original - X_test) / X_test)) * 100)
mape_x_inv_real = float(np.mean(np.abs((x_inv_real_original - X_test) / X_test)) * 100)

# Print MSEs
print(f'MSE(y, y_pred): {mse_y}')
print(f'MSE(x_inv, x): {mse_x_inv}')
print(f'MSE(x_inv_real, x): {mse_x_inv_real}')

# Print MAPEs
print(f'MAPE(y, y_pred): {mape_y}%')
print(f'MAPE(x_inv, x): {mape_x_inv}%')
print(f'MAPE(x_inv_real, x): {mape_x_inv_real}%')


### store results #######################################################################################

# Prepare data to store in JSON
results = {
    "x_test_original": X_test_original.tolist(), # Original x_test data
    "y_test_original": y_test_original.tolist(), # Original y_test data
    "y_pred": y_pred_original.tolist(),
    "x_inv": x_inv_original.tolist(),
    "x_inv_real": x_inv_real_original.tolist(),
    "MSE": {
        "MSE_y_y_pred": mse_y,
        "MSE_x_inv_x": mse_x_inv,
        "MSE_x_inv_real_x": mse_x_inv_real,
        "MAPE_y_y_pred": mape_y,
        "MAPE_x_inv_x": mape_x_inv,
        "MAPE_x_inv_real_x": mape_x_inv_real

    }
}

# Save the results to a JSON file
json_file_path = f'./results/{model_name}_results.json'
with open(json_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {json_file_path}")



visualize(json_file_path, model_name)

