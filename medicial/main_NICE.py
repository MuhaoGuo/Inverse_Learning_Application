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

batch_size = 100000
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
                # loss_real = criterion(inputs, real_inv)
                # loss = loss + loss_real

                loss.backward()
                optimizer.step()
                # model.apply_weight_masks()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))


input_dim = X.shape[-1]
num_blocks = 1
learning_rate = 1e-3
# num_epochs = 5000
epochs = 2000 #000 # 400

device = get_device()
# device = torch.device('cpu')
model_name = "NICE"
data_name = "ECG"
model = NICEResNet(input_dim=input_dim, num_blocks=num_blocks).to(device)
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
