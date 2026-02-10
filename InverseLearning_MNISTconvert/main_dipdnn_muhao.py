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

parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on MNIST convert')
parser.add_argument('--save_dir', default="./MNIST_dipdnn_muhao", type=str, help='directory to save results') # mnist cifar10
parser.add_argument('--dataset', default='FashionMNIST', type=str, help='dataset')  # FashionMNIST MNIST Cifar10

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

        self.negative_slope = 0.9

    # def forward(self, x):
    #     fc1_fwd = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)  # 0.01
    #     fc2_fwd = F.leaky_relu(self.fc2(fc1_fwd), negative_slope=self.negative_slope)
    #     return fc2_fwd
    
    # def inverse(self, y):
    #     fc2_W_T = torch.linalg.inv(torch.mul(self.fc2.weight, self.mask_triu))
    #     fc2_inv = F.linear(F.leaky_relu(y, negative_slope=1/self.negative_slope) - self.fc2.bias, fc2_W_T)
    #     fc1_W_T = torch.linalg.inv(torch.mul(self.fc1.weight, self.mask_tril))
    #     fc1_inv = F.linear(F.leaky_relu(fc2_inv, negative_slope=1/self.negative_slope) - self.fc1.bias, fc1_W_T)
    #     return fc1_inv

    def forward(self, x):
        fc1_fwd = self.fc1(x)
        fc1_fwd = F.leaky_relu(fc1_fwd, negative_slope = self.negative_slope)
        fc2_fwd = self.fc2(fc1_fwd)
        fc2_fwd = F.leaky_relu(fc2_fwd, negative_slope = self.negative_slope)
        return fc2_fwd
    
    # def inverse(self, y):
    #     y1 = F.leaky_relu(y, negative_slope=1 / self.negative_slope) - self.fc2.bias
    #     y1 = y - self.fc2.bias
    #     fc2_W_T = torch.linalg.inv(self.fc2.weight)
    #     fc2_inv = F.linear(y1, fc2_W_T)

    #     y2 = F.leaky_relu(fc2_inv, negative_slope=1 / self.negative_slope) - self.fc1.bias
    #     fc1_W_T = torch.linalg.inv(self.fc1.weight)
    #     fc1_inv = F.linear(y2, fc1_W_T)
    #     return fc1_inv
    
    def inverse(self, y):
        batch_size = y.shape[0]
        # Inverse of the last activation
        y1 = F.leaky_relu(y, negative_slope=1 / self.negative_slope)
        # Subtract bias
        y1 = y1 - self.fc2.bias
        # Solve fc2.weight * x = y1 using triangular solve
        y1_unsqueezed = y1.unsqueeze(2)  # shape (batch_size, input_dim, 1)
        fc2_weight_expanded = self.fc2.weight.unsqueeze(0).expand(batch_size, -1, -1)
        fc2_inv, _ = torch.triangular_solve(y1_unsqueezed, fc2_weight_expanded, upper=True)
        fc2_inv = fc2_inv.squeeze(2)

        # Inverse of the first activation
        y2 = F.leaky_relu(fc2_inv, negative_slope=1 / self.negative_slope)
        # Subtract bias
        y2 = y2 - self.fc1.bias
        # Solve fc1.weight * x = y2 using triangular solve
        y2_unsqueezed = y2.unsqueeze(2)
        fc1_weight_expanded = self.fc1.weight.unsqueeze(0).expand(batch_size, -1, -1)
        fc1_inv, _ = torch.triangular_solve(y2_unsqueezed, fc1_weight_expanded, upper=False)
        fc1_inv = fc1_inv.squeeze(2)
        return fc1_inv

    def apply_weight_masks(self):
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

    # def forward(self, x):
    #     # Forward pass with masked linear layers and Tanh activation
    #     z1 = self.fc1(x)
    #     z2 = torch.tanh(z1)
    #     z3 = self.fc2(z2)
    #     # y = torch.tanh(z3)
    #     return z3


    # def inverse(self, y):
    #     epsilon = 1e-6  # For numerical stability
        
    #     # Step 1: Inverse of second Linear Transformation
    #     z2_inv = F.linear(y - self.fc2.bias, torch.linalg.inv(self.fc2.weight))
        
    #     # Step 2: Inverse of Tanh Activation
    #     z1_inv_activation = 0.5 * torch.log((1 + z2_inv) / (1 - z2_inv))
        
    #     # Step 3: Inverse of first Linear Transformation
    #     x_inv = F.linear(z1_inv_activation - self.fc1.bias, torch.linalg.inv(self.fc1.weight))
        
    #     return x_inv
    

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

        # y = y.reshape(y.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        if args.dataset == "Cifar10":
            y = y.reshape(y.shape[0], 3, 32, 32)
        else:
            y = y.reshape(y.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        return y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    def apply_weight_masks(self):
        for block in self.blocks:
            block.apply_weight_masks()




image_l = 28
input_dim = image_l * image_l  # Flattened MNIST images 28 * 28
num_blocks = 3
learning_rate = 1e-3
num_epochs = 1000

if args.dataset == "Cifar10":
    input_dim = 32 * 32 * 3 
    learning_rate = 1e-5
    num_epochs = 2000


# Initialize the iResNet model
model = DipDNN(input_dim=input_dim, num_blocks=num_blocks).to(device)
print(f'Total trainable parameters: {model.count_parameters()}')
print(model)

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()

# Define the optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=1e-5)


reg_count = 0
# Training loop
mask_tril = torch.tril(torch.ones_like(model.blocks[0].residual_function.fc1.weight))
mask_triu = torch.triu(torch.ones_like(model.blocks[0].residual_function.fc2.weight))

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        # inverse pass
        input_est = model.inverse(targets)
        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.apply_weight_masks()

        epoch_loss += loss.item() * inputs.size(0)

        # print(model.blocks[0].residual_function.fc1.weight)
        ############################################################
        with torch.no_grad():
            # Regularize weights to prevent near-singular matrices
            for name, param in model.named_parameters():
                if 'weight' in name:
                    diag_elements = torch.diag(param)
                    threshold = 1e-3 # 1e-2
                    mask = torch.abs(diag_elements) < threshold
                    non_zero_value = 1e-1 # 1e-1
                    if torch.any(mask):
                        param[torch.eye(param.size(0)).bool()] = torch.where(
                            mask, torch.tensor(non_zero_value, device=param.device), diag_elements
                        )
                        reg_count += 1

            # Re-apply masks to ensure triangular structure
            # for block in model.blocks:
            #     block.residual_function.fc1.weight.mul_(mask_tril)
            #     block.residual_function.fc2.weight.mul_(mask_triu)
        #########################################################################

        # print(model.blocks[0].residual_function.fc1.weight)
        # exit()

    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch}/{num_epochs}], Forward Pass Loss: {avg_loss}')

    # Optionally, check Lipschitz continuity
    # model.check_lipz_continuity()
print("reg_count", reg_count)


# print(model.blocks[0].residual_function.fc1.weight)
# print(model.blocks[0].residual_function.fc2.weight)
# exit()

# Evaluate inverse error after training
model.eval()
with torch.no_grad():
    inverse_error = 0.0
    inverse_error_real = 0.0 
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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

    avg_inverse_error = inverse_error / len(train_loader.dataset)
    avg_inverse_error_real = inverse_error_real / len(train_loader.dataset)
    print(f'Inverse Pass Error (MSE) after training: {avg_inverse_error}')
    print(f'Inverse Pass Error (MSE) after training real: {avg_inverse_error_real}')


if args.dataset == "Cifar10":
    visualize_results_color(model, train_loader, device, num_samples=3, save_path=f"{args.save_dir}/{args.dataset}_Samples_train.png")
    visualize_results_color(model, test_loader, device, num_samples=3, save_path=f"{args.save_dir}/{args.dataset}_Samples_test.png")
else:
    visualize_results(model, train_loader, device, num_samples=3, save_path=f"{args.save_dir}/{args.dataset}_Samples_train.png")
    visualize_results(model, test_loader, device, num_samples=3, save_path=f"{args.save_dir}/{args.dataset}_Samples_test.png")






