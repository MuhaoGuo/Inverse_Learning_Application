
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



class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, process_first_part=True):
        """
        Affine Coupling Layer as per NICE architecture.

        Args:
            input_dim (int): Dimension of the input.
            process_first_part (bool): Whether to process the first or second part of the input.
        """
        super(AffineCouplingLayer, self).__init__()
        self.process_first_part = process_first_part
        self.input_dim = input_dim
        self.split_dim1 = input_dim // 2
        self.split_dim2 = input_dim - self.split_dim1

        if self.process_first_part:
            network_input_dim = self.split_dim1
            network_output_dim = self.split_dim2
        else:
            network_input_dim = self.split_dim2
            network_output_dim = self.split_dim1

        self.network = nn.Sequential(
            nn.Linear(network_input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, network_output_dim)
        )

    def forward(self, x):
        """
        Forward pass of the Affine Coupling Layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Output tensor after coupling.
        """
        x1, x2 = x.split([self.split_dim1, self.split_dim2], dim=1)
        if self.process_first_part:
            shift = self.network(x1)
            y1 = x1
            y2 = x2 + shift
        else:
            shift = self.network(x2)
            y1 = x1 + shift
            y2 = x2
        y = torch.cat((y1, y2), dim=1)
        return y

    def inverse(self, y):
        """
        Inverse pass of the Affine Coupling Layer.

        Args:
            y (Tensor): Output tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Reconstructed input tensor.
        """
        y1, y2 = y.split([self.split_dim1, self.split_dim2], dim=1)
        if self.process_first_part:
            shift = self.network(y1)
            x1 = y1
            x2 = y2 - shift
        else:
            shift = self.network(y2)
            x1 = y1 - shift
            x2 = y2
        x = torch.cat((x1, x2), dim=1)
        return x



class NICEResNet(nn.Module):
    def __init__(self, input_dim, num_blocks):
        """
        Modified NICE model inspired by ResNet architecture.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Number of hidden units in the coupling layers.
            num_blocks (int): Number of Affine Coupling Layers (blocks) in the model.
        """
        super(NICEResNet, self).__init__()
        self.input_dim = input_dim
        self.num_blocks = num_blocks

        # Initialize Affine Coupling Layers
        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            process_first_part = (i % 2 == 0)  # Alternate processing parts for each block
            self.layers.append(AffineCouplingLayer(input_dim, process_first_part))

        # Learnable scaling factor
        self.scaling_factor = nn.Parameter(torch.ones(1, input_dim))

    def forward(self, x):
        """
        Forward pass through the NICEResNet model.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Transformed output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        x = x * self.scaling_factor
        return x

    def inverse(self, y):
        """
        Inverse pass through the NICEResNet model.

        Args:
            y (Tensor): Output tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Reconstructed input tensor.
        """
        y = y / self.scaling_factor
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

    def count_parameters(self):
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Code cell
# Set input and output size based on the dataset
input_size = X_train.shape[1]
output_size = y_train.shape[1]
print("input_size", input_size)
print("output_size", output_size)
block_num = 3
epochs = 20000
model_name = "NICE"

# Initialize the model, loss function, and optimizer
model = NICEResNet(input_size, block_num)
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






