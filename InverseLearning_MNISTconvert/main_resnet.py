from MNIST_negative_data import *
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
# from utils import *
from enum import Enum, auto
import torch
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.modules import Module
from torch.nn import functional as F
from typing import Optional
import argparse
import os
from utils import *

parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
parser.add_argument('--save_dir', default="./MNIST_resnet", type=str, help='directory to save results') # mnist cifar10
parser.add_argument('--dataset', default='Cifar10', type=str, help='dataset')  # MNIST  FashionMNIST Cifar10

# parser.add_argument('--epochs', default=200, type=int, help='number of epochs')  # 200
# parser.add_argument('--batch', default=128, type=int, help='batch size')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--optimizer', default="sgd", type=str, help="optimizer", choices=["adam", "adamax", "sgd"])
# parser.add_argument('-nesterov', '--nesterov', dest='nesterov', action='store_true',
#                     help='nesterov momentum')
# parser.add_argument('--weight_decay', default=5e-4, type=float, help='coefficient for weight decay')
# parser.add_argument('-densityEstimation', '--densityEstimation', dest='densityEstimation', action='store_true', help='perform density estimation', default=False)
# parser.add_argument('--nBlocks', nargs='+', type=int, default=[13, 13, 13])   # 4, 4, 4   / 1, 0, 0
# parser.add_argument('--nStrides', nargs='+', type=int, default=[1, 2, 2])   # 1, 2, 2  / 1, 0, 0
# parser.add_argument('--nChannels', nargs='+', type=int, default=[12, 48, 192])  #16, 64, 256 12, 48, 192
# parser.add_argument('--coeff', default=0.9, type=float, help='contraction coefficient for linear layers')
# parser.add_argument('--numTraceSamples', default=1, type=int, help='number of samples used for trace estimation')
# parser.add_argument('--numSeriesTerms', default=1, type=int, help='number of terms used in power series for matrix log')
# parser.add_argument('--powerIterSpectralNorm', default=5, type=int, help='number of power iterations used for spectral norm')
# parser.add_argument('--init_batch', default=1024, type=int, help='init batch size')
# parser.add_argument('--init_ds', default=2, type=int, help='initial downsampling')
# parser.add_argument('--inj_pad', default=0, type=int, help='initial inj padding')
# parser.add_argument('--warmup_epochs', default=10, type=int, help='epochs for warmup')
args = parser.parse_args()



def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility in convolutional layers
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

# Set the seed before initializing the model or any random operations
set_seed(42)  # You can use any seed value of your choice

# Detect device: CUDA for Ubuntu GPU, MPS for MacBook with M1/M2/M3, or CPU fallback
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA (GPU) for training.")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Metal (MPS) for training on MacBook Pro with M1/M2/M3.")
else:
    device = torch.device('cpu')
    print("Using CPU for training.")
# device = torch.device('cpu')


if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)




##################################################################################
if args.dataset == "MNIST":
    train_loader, test_loader = load_MNIST_convert_data(batch_size = 128, samples= 30)
if args.dataset == "FashionMNIST":
    train_loader, test_loader = load_FashionMNIST_convert_data(batch_size = 128, samples= 20)
if args.dataset == "Cifar10":
    from utils_color import *
    train_loader, test_loader = load_cifar10_convert_data(batch_size = 128, samples= 10)
##################################################################################

class SmallNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        # self.fc3 = nn.Linear(input_dim, input_dim)
        # self.activation = nn.ReLU()
        # self.activation = nn.Sigmoid()
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # x = self.activation(self.fc2(x))
        # x = self.fc3(x)
        return x


# Define the invertible residual block
# with options for Lipschitz enforcement
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, alpha=0.9):
        super(ResidualBlock, self).__init__()
        self.alpha = alpha  # Scaling coefficient
        self.residual_function = SmallNetwork(input_dim)

    def forward(self, x):
        return x + self.alpha * self.residual_function(x)
        # return self.residual_function(x)

    def inverse(self, y):
        # Use fixed-point iteration to find the inverse
        x = y
        for _ in range(100):
            x = y - self.alpha * self.residual_function(x)  # y = x + af(x) --> x = y
        return x


# Define the i-ResNet
class ResNet(nn.Module):
    def __init__(self, input_dim, num_blocks):
        super(ResNet, self).__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(input_dim) for _ in range(num_blocks)])

    def forward(self, x):

        x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for block in self.blocks:
            x = block(x)

        if args.dataset == "Cifar10":
            x = x.reshape(x.shape[0], 3, 32, 32)
        else:
            x = x.reshape(x.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        return x

    def inverse(self, y):
        y = y.reshape(y.shape[0], -1)  # (batchsize, -1)

        for block in reversed(self.blocks):
            y = block.inverse(y)

        
        if args.dataset == "Cifar10":
            y = y.reshape(y.shape[0], 3, 32, 32)
        else:
            y = y.reshape(y.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        return y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


input_dim = 28 * 28  # Flattened MNIST images
hidden_dim = 128
num_blocks = 3
enforce_lipz = 'power_iteration'  # or 'power_iteration'
learning_rate = 1e-3
num_epochs = 1000

if args.dataset == "Cifar10":
    input_dim = 32 * 32 * 3 
    learning_rate = 1e-5
    num_epochs = 2000

# Initialize the iResNet model
model = ResNet(input_dim=input_dim, num_blocks=num_blocks).to(device)
print(f'Total trainable parameters: {model.count_parameters()}')

print(model)

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Define the optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)

    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch}/{num_epochs}], Forward Pass Loss: {avg_loss:.6f}')

    # Optionally, check Lipschitz continuity
    # model.check_lipz_continuity()

# Evaluate inverse error after training
model.eval()
with torch.no_grad():
    inverse_error = 0.0
    inverse_error_real = 0.0 
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass: input -> inverted
        outputs = model(inputs)

        # Inverse pass: f(x) -> input
        reconstructed_inputs = model.inverse(outputs)
        reconstructed_inputs_real = model.inverse(targets)
        # print("reconstructed_inputs", reconstructed_inputs)
        
        # Compute inverse error (MSE between original inputs and reconstructed inputs)
        loss = criterion(reconstructed_inputs, inputs)
        loss_real = criterion(reconstructed_inputs_real, inputs)

        inverse_error += loss.item() * inputs.size(0)
        inverse_error_real += loss_real.item() * inputs.size(0)

    avg_inverse_error = inverse_error / len(test_loader.dataset)
    avg_inverse_error_real = inverse_error_real / len(test_loader.dataset)
    print(f'Inverse Pass Error (MSE) after training: {avg_inverse_error}')
    print(f'Inverse Pass Error (MSE) after training real: {avg_inverse_error_real}')


if args.dataset == "Cifar10":
    visualize_results_color(model, train_loader, device, num_samples=3, save_path=f"{args.save_dir}/{args.dataset}_Samples_train.png")
    visualize_results_color(model, test_loader, device, num_samples=3, save_path=f"{args.save_dir}/{args.dataset}_Samples_test.png")
else:
    visualize_results(model, train_loader, device, num_samples=3, save_path=f"{args.save_dir}/{args.dataset}_Samples_train.png")
    visualize_results(model, test_loader, device, num_samples=3, save_path=f"{args.save_dir}/{args.dataset}_Samples_test.png")






