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
    # X = np.hstack([df[load_P_columns].values, df[load_Q_columns].values])
    # y = np.hstack([df[voltage_columns].values, df[theta_columns].values])

    X = np.hstack([df[load_Q_columns].values])
    y = np.hstack([df[voltage_columns].values])


    X = X[::10, :] 
    y = y[::10, :]

    # X = X[:, [0,2]]
    # y = y[:, [0,2]]
    print(X.shape, y.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Subplot 1: Input data x
    axes[0].plot(X, label='X', alpha=0.6)
    axes[0].set_title('Input Data x')
    axes[0].legend()

    axes[1].plot(y, label='y', alpha=0.6)
    axes[1].set_title('Output Data y')
    axes[1].legend()
    
    plt.savefig("./results/powerflow_raw_data.png")

    return X, y




def visualize(json_file_path, model_name):

    with open(json_file_path, 'r') as json_file:
        results = json.load(json_file)

    # Extract data
    x_test_original = np.array(results["x_test_original"])
    y_test_original = np.array(results["y_test_original"])
    y_pred_original = np.array(results["y_pred"])
    x_inv_original = np.array(results["x_inv"])
    x_inv_real_original = np.array(results["x_inv_real"])
    mse = results["MSE"]
    
    # Visualization with 5 Subplots (Modified)
    # Select a subset of data for visualization (e.g., first 100 samples)
    num_samples = len(x_test_original)  #100
    indices = np.arange(num_samples)

    # Prepare data for plotting
    x_plot = x_test_original[:num_samples]
    y_plot = y_test_original[:num_samples]
    y_pred_plot = y_pred_original[:num_samples]
    x_inv_plot = x_inv_original[:num_samples]
    x_inv_real_plot = x_inv_real_original[:num_samples]


    # Create a figure with 5 subplots in one row
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    # Subplot 1: Input data x
    axes[0].plot(indices, x_plot[:, 0], label='P', alpha=0.6)
    axes[0].plot(indices, x_plot[:, -1], label='Q', alpha=0.6)
    axes[0].set_title('Input Data x')
    axes[0].legend()

    # Subplot 2: Target data y
    axes[1].plot(indices, y_plot[:, 0], label='voltage', alpha=0.6)
    axes[1].plot(indices, y_plot[:, -1], label='theta', alpha=0.6)
    axes[1].set_title('Target Data y')
    axes[1].legend()

    # Subplot 3: Forward predicted data y_pred
    axes[2].plot(indices, y_pred_plot[:, 0], label='voltage_pred', alpha=0.6)
    axes[2].plot(indices, y_pred_plot[:, -1], label='theta_pred', alpha=0.6)
    axes[2].set_title('Predicted Data y_pred')
    axes[2].legend()

    # Subplot 4: Inverse data x_inv from y_pred
    axes[3].plot(indices, x_inv_plot[:, 0], label='P_inv', alpha=0.6)
    axes[3].plot(indices, x_inv_plot[:, -1], label='Q_inv', alpha=0.6)
    axes[3].set_title('Inverse Data x_inv from y_pred')
    axes[3].legend()

    # Subplot 5: Real inverse data x_inv_real from y
    axes[4].plot(indices, x_inv_real_plot[:, 0], label='P_inv_real', alpha=0.6)
    axes[4].plot(indices, x_inv_real_plot[:, -1], label='Q_inv_real', alpha=0.6)
    axes[4].set_title('Real Inverse Data x_inv_real from y')
    axes[4].legend()

    # axes[5].plot(indices, y_pred_original[:, 0] - y_plot[:, 0], label='forward error', alpha=0.6)
    # axes[5].plot(indices, y_pred_original[:, 1] - y_plot[:, -1], label='forward error', alpha=0.6)
    # axes[5].set_title('error of y and f(x)')
    # axes[5].legend()

    plt.tight_layout()
    plt.savefig(f"./results/{model_name}_powerflow.png")
    plt.show()



import random

def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility in convolutional layers
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility
torch.manual_seed(0)

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



class SmallNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

        self.register_buffer('mask_tril', torch.tril(torch.ones_like(self.fc1.weight)))
        self.register_buffer('mask_triu', torch.triu(torch.ones_like(self.fc2.weight)))
        self.fc1.weight = nn.Parameter(torch.mul(self.fc1.weight, self.mask_tril))
        self.fc2.weight = nn.Parameter(torch.mul(self.fc2.weight, self.mask_triu))

        # Apply masks and ensure diagonal elements are ones
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

        # Register hooks to enforce masking after each update
        self.fc1.weight.register_hook(lambda grad: grad * self.mask_tril)
        self.fc2.weight.register_hook(lambda grad: grad * self.mask_triu)

        self.negative_slope = 0.5

    
    def inverse_leaky_relu(self, z, negative_slope):
        return torch.where(z >= 0, z, z / negative_slope)


    def forward(self, x):
        fc1_fwd = self.fc1(x)
        fc1_fwd = F.leaky_relu(fc1_fwd, negative_slope=self.negative_slope)
        fc2_fwd = self.fc2(fc1_fwd)
        fc2_fwd = F.leaky_relu(fc2_fwd, negative_slope=self.negative_slope)
        return fc2_fwd

    def inverse(self, y):
        batch_size = y.shape[0]
        # Inverse of the last activation
        # y1 = F.leaky_relu(y, negative_slope=1 / self.negative_slope)
        y1 = self.inverse_leaky_relu(y, self.negative_slope)

        # Subtract bias
        y1 = y1 - self.fc2.bias
        # Solve fc2.weight * x = y1 using triangular solve
        y1_unsqueezed = y1.unsqueeze(2)  # shape (batch_size, input_dim, 1)
        fc2_weight_expanded = self.fc2.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # fc2_inv, _ = torch.triangular_solve(y1_unsqueezed, fc2_weight_expanded, upper=True)
        fc2_inv = torch.linalg.solve_triangular(fc2_weight_expanded, y1_unsqueezed, upper=True)

        fc2_inv = fc2_inv.squeeze(2)

        # Inverse of the first activation
        # y2 = F.leaky_relu(fc2_inv, negative_slope=1 / self.negative_slope)
        y2 = self.inverse_leaky_relu(fc2_inv, negative_slope=self.negative_slope)


        # Subtract bias
        y2 = y2 - self.fc1.bias
        # Solve fc1.weight * x = y2 using triangular solve
        y2_unsqueezed = y2.unsqueeze(2)
        fc1_weight_expanded = self.fc1.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # fc1_inv, _ = torch.triangular_solve(y2_unsqueezed, fc1_weight_expanded, upper=False)
        fc1_inv = torch.linalg.solve_triangular(fc1_weight_expanded, y2_unsqueezed, upper=False)

        fc1_inv = fc1_inv.squeeze(2)
        return fc1_inv

    def apply_weight_masks(self):
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

class DipDNN_Block(nn.Module):
    def __init__(self, input_dim, alpha=0.9):
        super(DipDNN_Block, self).__init__()
        # self.alpha = alpha  # Scaling coefficient
        self.residual_function = SmallNetwork(input_dim)

    def forward(self, x):
        return self.residual_function(x)

    def inverse(self, y):
        return self.residual_function.inverse(y)

    def apply_weight_masks(self):
        self.residual_function.apply_weight_masks()

class DipDNN(nn.Module):
    def __init__(self, input_dim, num_blocks):
        super(DipDNN, self).__init__()
        self.blocks = nn.ModuleList(
            [DipDNN_Block(input_dim) for _ in range(num_blocks)])
        self.input_dim = input_dim

    def forward(self, x):

        # x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for block in self.blocks:
            x = block(x)

        # x = x.reshape(x.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        return x

    def inverse(self, y):
        # y = y.reshape(y.shape[0], -1)  # (batchsize, -1)

        for block in reversed(self.blocks):
            y = block.inverse(y)

        # y = y.reshape(y.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        return y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_weight_masks(self):
        for block in self.blocks:
            block.apply_weight_masks()


# Code cell
# Set input and output size based on the dataset
input_size = X_train.shape[1]
output_size = y_train.shape[1]
print("input_size", input_size)
print("output_size", output_size)
block_num = 3
epochs =  20000
model_name = "dipdnn"
apply_weight_masks = False

# Initialize the model, loss function, and optimizer
model = DipDNN(input_size, block_num)
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

    if apply_weight_masks:
        model.apply_weight_masks()

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


y_pred_original = y_pred.cpu().numpy()  
y_test_original = y_test_tensor.cpu().numpy()
x_inv_original = x_inv.cpu().numpy()
x_inv_real_original =x_inv_real.cpu().numpy()
X_test_original = X_test


x_inv_original = np.maximum(x_inv_original, 0)
x_inv_real_original = np.maximum(x_inv_real_original, 0)




# Calculate MSEs and convert them to native Python floats
mse_y = float(np.mean((y_pred_original - y_test_original) ** 2))
mse_x_inv = float(np.mean((x_inv_original - X_test_original) ** 2))
mse_x_inv_real = float(np.mean((x_inv_real_original - X_test_original) ** 2))

# Calculate MAPE and convert them to native Python floats
mape_y = float(np.mean(np.abs((y_pred_original - y_test_original) / y_test_original)) * 100)
mape_x_inv = float(np.mean(np.abs((x_inv_original - X_test_original) / X_test_original)) * 100)
mape_x_inv_real = float(np.mean(np.abs((x_inv_real_original - X_test_original) / X_test_original)) * 100)

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


