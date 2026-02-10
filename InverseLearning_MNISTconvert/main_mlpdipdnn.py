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

parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
parser.add_argument('--save_dir', default="./MNIST_mlpdipdnn_400epoch_0001_1relu_1", type=str, help='directory to save results') # mnist cifar10
parser.add_argument('--dataset', default='MNIST', type=str, help='dataset')  # mnist cifar10

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

# Load MNIST dataset
data_path = '../data/'
mnist_data_train = torchvision.datasets.MNIST(root=data_path, train=True, download=False, )
mnist_data_test = torchvision.datasets.MNIST(root=data_path, train=False, download=False, )

##################################################
# section: Create a subset of the dataset
from torch.utils.data import Subset

subset_indices = list(range(30))
mnist_data_train = Subset(mnist_data_train, subset_indices)
mnist_data_test = Subset(mnist_data_test, subset_indices)
##################################################

# Example: Generate dataset with color inversion
transformed_dataset_train = generate_dataset(transform_type='invert_colors', data=mnist_data_train)
transformed_dataset_test = generate_dataset(transform_type='invert_colors', data=mnist_data_test)

# Visualize the transformed examples (color inversion in this case)
visualize_transformed_examples(transformed_dataset_train)
visualize_transformed_examples(transformed_dataset_test)

# Prepare the dataset for training the neural network
inputs_train, outputs_train = prepare_for_nn(transformed_dataset_train)
inputs_test, outputs_test = prepare_for_nn(transformed_dataset_test)

# inputs and outputs are now ready to be fed into a neural network
print(f"inputs_train: {inputs_train.shape}")
print(f"outputs_train: {outputs_train.shape}")

print(f"inputs_test: {inputs_test.shape}")
print(f"outputs_test: {outputs_test.shape}")

batch_size = 128
train_tensor = TensorDataset(inputs_train, outputs_train)
train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

test_tensor = TensorDataset(inputs_test, outputs_test)
test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=True)


##################################################################################
##################################################################################

class Inverse_Net_maskBlock(nn.Module):
    def __init__(self, in_features):
        super(Inverse_Net_maskBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.fc2 = nn.Linear(in_features, in_features, bias=True)

        # self.register_buffer('mask_tril', torch.tril(torch.ones_like(self.fc1.weight)))
        # self.register_buffer('mask_triu', torch.triu(torch.ones_like(self.fc2.weight)))
        # self.fc1.weight = nn.Parameter(torch.mul(self.fc1.weight, self.mask_tril))
        # self.fc2.weight = nn.Parameter(torch.mul(self.fc2.weight, self.mask_triu))
        self.act = nn.ELU()
    def forward(self, x):
        # fc1_fwd = F.leaky_relu(self.fc1(x), negative_slope=0.1)  # 0.01
        # fc2_fwd = F.leaky_relu(self.fc2(fc1_fwd), negative_slope=0.1)

        x = self.fc1(x)
        x = self.act(x)
        fc2_fwd = self.fc2(x)
        # fc2_fwd += x
        return fc2_fwd

    def inverse(self, y):
        # fc2_W_T = torch.linalg.inv(torch.mul(self.fc2.weight, self.mask_triu))
        # fc2_inv = F.linear(F.leaky_relu(y, negative_slope=1/0.1) - self.fc2.bias, fc2_W_T)
        # fc1_W_T = torch.linalg.inv(torch.mul(self.fc1.weight, self.mask_tril))
        # fc1_inv = F.linear(F.leaky_relu(fc2_inv, negative_slope=1/0.1) - self.fc1.bias, fc1_W_T
        
        return y

# Define the Invertible Neural Network for feature representation
class Inverse_DNN_with_Features(nn.Module):
    def __init__(self, input_dim, num_blocks):
        super(Inverse_DNN_with_Features, self).__init__()
        self.layers = nn.ModuleList([Inverse_Net_maskBlock(input_dim) for _ in range(num_blocks)])
        # self.feature_fc = nn.Linear(input_dim, input_dim)
        # self.classifier = nn.Linear(feature_dim, num_classes)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for layer in self.layers:
            x = layer(x)
        # features = F.leaky_relu(self.feature_fc(x))
        # x = self.feature_fc(x)
        # features = self.dropout(features)
        # out = self.classifier(features)

        x = x.reshape(x.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))

        return x

    def inverse(self, x):
        x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for layer in reversed(self.layers):
            x = layer.inverse(x)

        x = x.reshape(x.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



input_dim = 28 * 28  # Flattened MNIST images
hidden_dim = 128
num_blocks = 3
# enforce_lipz = 'power_iteration'  # or 'power_iteration'
learning_rate = 1e-3
num_epochs = 400

# Initialize the iResNet model
model = Inverse_DNN_with_Features(input_dim=input_dim, num_blocks=num_blocks).to(device)
print(f'Total trainable parameters: {model.count_parameters()}')
print(model)

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Define the optimizer (Adam)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass: input -> inverted
        outputs  = model(inputs)

        # Inverse pass: inverted -> input
        reconstructed_inputs = model.inverse(outputs)

        # Compute inverse error (MSE between original inputs and reconstructed inputs)
        loss = criterion(reconstructed_inputs, inputs)
        inverse_error += loss.item() * inputs.size(0)

    avg_inverse_error = inverse_error / len(test_loader.dataset)
    print(f'Inverse Pass Error (MSE) after training: {avg_inverse_error:.6f}')




def visualize_results(model, test_loader, device, num_samples=3, save_path="./res.png"):
    model.eval()
    samples_visualized = 0
    plt.figure(figsize=(12, num_samples * 3))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass: input -> inverted
            outputs = model(inputs)

            # Inverse pass: inverted -> input
            reconstructed_inputs = model.inverse(outputs)

            # Inverse pass: input -> some output
            inverse_of_y = model.inverse(targets)

            for i in range(inputs.size(0)):
                if samples_visualized >= num_samples:
                    break

                # Original Input x
                original = inputs[i].cpu().numpy().reshape(28, 28)

                # target y 
                target = targets[i].cpu().numpy().reshape(28, 28)

                # Forward Output  f(x)
                forward_output = outputs[i].cpu().numpy().reshape(28, 28)

                # Inverse of f(x)
                inverse_forward = reconstructed_inputs[i].cpu().numpy().reshape(28, 28)

                # Inverse of y
                inverse_input = inverse_of_y[i].cpu().numpy().reshape(28, 28)

                # Plotting
                titles = ['Input x', "Target y", 'model f(x)', 'Inverse of f(x)', 'Inverse of y']
                images = [original, target, forward_output, inverse_forward, inverse_input]

                for j in range(5):
                    plt.subplot(num_samples, 5, samples_visualized * 5 + j + 1)
                    plt.imshow(images[j], cmap='gray')
                    plt.title(titles[j])
                    plt.axis('off')

                samples_visualized += 1

            if samples_visualized >= num_samples:
                break

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



visualize_results(model, train_loader, device, num_samples=3, save_path=f"{args.save_dir}/MNIST_Samples_train.png")
visualize_results(model, test_loader, device, num_samples=3, save_path=f"{args.save_dir}/MNIST_Samples_test.png")














