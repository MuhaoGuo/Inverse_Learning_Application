import pandas as pd
import numpy as np
import glob
import re
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
from get_data import get_medical_data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility in convolutional layers
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

# Set the seed before initializing the model or any random operations
set_seed(42)  # You can use any seed value of your choice

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

X, y = get_medical_data()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
# X, y = X.T, y.T
X, y = X.reshape(4, -1), y.reshape(4, -1)
dataset = TensorDataset(X, y)
print(X.shape, y.shape)
# exit()

batch_size = 100000
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


    def forward(self, x):
        fc1_fwd = self.fc1(x)
        fc1_fwd = F.leaky_relu(fc1_fwd, negative_slope=self.negative_slope)
        fc2_fwd = self.fc2(fc1_fwd)
        fc2_fwd = F.leaky_relu(fc2_fwd, negative_slope=self.negative_slope)
        return fc2_fwd

    def inverse(self, y):
        batch_size = y.shape[0]
        # Inverse of the last activation
        y1 = F.leaky_relu(y, negative_slope=1 / self.negative_slope)
        # Subtract bias
        y1 = y1 - self.fc2.bias
        # Solve fc2.weight * x = y1 using triangular solve
        y1_unsqueezed = y1.unsqueeze(2)  # shape (batch_size, input_dim, 1)
        fc2_weight_expanded = self.fc2.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # fc2_inv, _ = torch.triangular_solve(y1_unsqueezed, fc2_weight_expanded, upper=True)
        fc2_inv = torch.linalg.solve_triangular(fc2_weight_expanded, y1_unsqueezed, upper=True)

        fc2_inv = fc2_inv.squeeze(2)

        # Inverse of the first activation
        y2 = F.leaky_relu(fc2_inv, negative_slope=1 / self.negative_slope)
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

def train_model(model, train_loader, optimizer, criterion, epochs, device=torch.device('cpu')):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                real_inv = model.inverse(targets)

                loss = criterion(outputs, targets)
                loss_real = criterion(inputs, real_inv)
                loss = loss + loss_real

                loss.backward()
                optimizer.step()
                model.apply_weight_masks()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))


input_dim = X.shape[-1]
# print(input_dim)

num_blocks = 1
learning_rate = 1e-3
# num_epochs = 5000
epochs = 2000 #000 # 400

device = get_device()
# device = torch.device('cpu')
model_name = "dipdnn"
data_name = "ECG"
model = DipDNN(input_dim=input_dim, num_blocks=num_blocks).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

### train model #########################
train_model(model, train_loader, optimizer, criterion, epochs, device)

### test model #########################

model.eval()
with torch.no_grad():
    X = X.to(device)
    y = y.to(device)
    print(X.shape, y.shape)
    y_pred = model(X)  # [batch_size, output_size]
    x_inv_fake = model.inverse(y_pred)  # Inverse from y_pred
    x_inv_real = model.inverse(y)  # Inverse from y

    fwd_error = criterion(y_pred, y)
    inv_fake_error = criterion(x_inv_fake, X)
    inv_real_error = criterion(x_inv_real, X)
    print(f"{data_name, model_name}: fwd_error: {fwd_error}")
    print(f"{data_name, model_name}: inv_fake_error: {inv_fake_error}")
    print(f"{data_name, model_name}: inv_real_error: {inv_real_error}")


    X = X.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    x_inv_fake = x_inv_fake.cpu().detach().numpy()
    x_inv_real = x_inv_real.cpu().detach().numpy()

    # X, y, y_pred, x_inv_fake, x_inv_real = X.T, y.T, y_pred.T, x_inv_fake.T, x_inv_real.T
    X, y, y_pred, x_inv_fake, x_inv_real = X.reshape(-1, 2), y.reshape(-1, 2), y_pred.reshape(-1, 2), x_inv_fake.reshape(-1, 2), x_inv_real.reshape(-1, 2)


    # y_pred = target_scaler.inverse_transform(y_pred)
    # x_inv_fake = input_scaler.inverse_transform(x_inv_fake)
    # x_inv_real = input_scaler.inverse_transform(x_inv_real)

    indices = list(range(0, len(X), 1))

    fig, axes = plt.subplots(2, 5, figsize=(15, 3))
    # Subplot 1: Input data x
    axes[0, 0].plot(indices, X[:, 0], label='input1', alpha=0.6, color="blue")
    axes[1, 0].plot(indices, X[:, -1], label='input2', alpha=0.6, color="blue")
    axes[0, 0].set_title('Input x')

    # Subplot 2: Target data y
    axes[0, 1].plot(indices, y[:, 0], label='targets1', alpha=0.6, color="green")
    axes[1, 1].plot(indices, y[:, -1], label='targets2', alpha=0.6,  color="green")
    axes[0, 1].set_title('Target y')

    # Subplot 3: Forward predicted data y_pred
    axes[0, 2].plot(indices, y_pred[:, 0], label='pred1', alpha=0.6, color="green")
    axes[1, 2].plot(indices, y_pred[:, -1], label='pred2', alpha=0.6, color="green")
    axes[0, 2].set_title('y_pred')

    # Subplot 4: Inverse data x_inv from y_pred
    axes[0, 3].plot(indices, x_inv_fake[:, 0], label='inv_fake1', alpha=0.6, color="blue")
    axes[1, 3].plot(indices, x_inv_fake[:, -1], label='inv_fake2', alpha=0.6, color="blue")
    axes[0, 3].set_title('x_inv_fake from y_pred')

    # Subplot 5: Real inverse data x_inv_real from y
    axes[0, 4].plot(indices, x_inv_real[:, 0], label='inv_real1', alpha=0.6, color="blue")
    axes[1, 4].plot(indices, x_inv_real[:, -1], label='inv_real2', alpha=0.6, color="blue")
    axes[0, 4].set_title('x_inv_real from y')


    plt.tight_layout()
    plt.savefig(f"./results/{data_name}_{model_name}.png")
    plt.show()


# def visualize(json_file_path, model_name):
#     with open(json_file_path, 'r') as json_file:
#         results = json.load(json_file)
#
#     # Extract data
#     x_test_original = np.array(results["x_test_original"])
#     y_test_original = np.array(results["y_test_original"])
#     y_pred_original = np.array(results["y_pred"])
#     x_inv_original = np.array(results["x_inv"])
#     x_inv_real_original = np.array(results["x_inv_real"])
#     mse = results["MSE"]
#
#     # Visualization with 5 Subplots (Modified)
#     # Select a subset of data for visualization (e.g., first 100 samples)
#     num_samples = len(x_test_original)  # 100
#     indices = np.arange(num_samples)
#
#     # Prepare data for plotting
#     x_plot = x_test_original[:num_samples]
#     y_plot = y_test_original[:num_samples]
#     y_pred_plot = y_pred_original[:num_samples]
#     x_inv_plot = x_inv_original[:num_samples]
#     x_inv_real_plot = x_inv_real_original[:num_samples]
#
#     # Create a figure with 5 subplots in one row
#     fig, axes = plt.subplots(1, 5, figsize=(25, 5))
#     # Subplot 1: Input data x
#     axes[0].plot(indices, x_plot[:, 0], label='Q1', alpha=0.6)
#     axes[0].plot(indices, x_plot[:, -1], label='Q2', alpha=0.6)
#     axes[0].set_title('Input Data x')
#     axes[0].legend()
#
#     # Subplot 2: Target data y
#     axes[1].plot(indices, y_plot[:, 0], label='voltage1', alpha=0.6)
#     axes[1].plot(indices, y_plot[:, -1], label='voltage2', alpha=0.6)
#     axes[1].set_title('Target Data y')
#     axes[1].legend()
#
#     # Subplot 3: Forward predicted data y_pred
#     axes[2].plot(indices, y_pred_plot[:, 0], label='voltage_pred1', alpha=0.6)
#     axes[2].plot(indices, y_pred_plot[:, -1], label='voltage_pred2', alpha=0.6)
#     axes[2].set_title('Predicted Data y_pred')
#     axes[2].legend()
#
#     # Subplot 4: Inverse data x_inv from y_pred
#     axes[3].plot(indices, x_inv_plot[:, 0], label='Q1_inv', alpha=0.6)
#     axes[3].plot(indices, x_inv_plot[:, -1], label='Q2_inv', alpha=0.6)
#     axes[3].set_title('Inverse Data x_inv from y_pred')
#     axes[3].legend()
#
#     # Subplot 5: Real inverse data x_inv_real from y
#     axes[4].plot(indices, x_inv_real_plot[:, 0], label='Q1_inv_real', alpha=0.6)
#     axes[4].plot(indices, x_inv_real_plot[:, -1], label='Q2_inv_real', alpha=0.6)
#     axes[4].set_title('Real Inverse Data x_inv_real from y')
#     axes[4].legend()
#
#     # axes[5].plot(indices, y_pred_original[:, 0] - y_plot[:, 0], label='forward error', alpha=0.6)
#     # axes[5].plot(indices, y_pred_original[:, 1] - y_plot[:, -1], label='forward error', alpha=0.6)
#     # axes[5].set_title('error of y and f(x)')
#     # axes[5].legend()
#
#     plt.tight_layout()
#     plt.savefig(f"./results/{model_name}_powerflow.png")
#     plt.show()