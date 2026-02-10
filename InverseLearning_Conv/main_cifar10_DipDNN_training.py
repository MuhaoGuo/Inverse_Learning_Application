import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import torchvision.transforms as transforms
# import visdom
import os
import sys
import time
import argparse
import pdb
import torch.nn as nn
import random
import json
from utils_cifar import train, test, std, mean, get_hms, interpolate
# from conv_iResNet import conv_iResNet as iResNet
# from conv_iResNet import conv_ResNet as ResNet
# from conv_iResNet import conv_ResNet_specific
# from models.conv_iResNet import multiscale_conv_iResNet as multiscale_iResNet
from conv_iResNet import conv_ResNet_specific
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import platform
import scipy.linalg
import copy

parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
# parser.add_argument('--save_dir', default="./resnet_specific_cifar10_12block_200epoch_12channels", type=str, help='directory to save results') # mnist cifar10
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')  # mnist cifar10
parser.add_argument('--epochs', default=20, type=int, help='number of epochs')  # 200
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--optimizer', default="sgd", type=str, help="optimizer", choices=["adam", "adamax", "sgd"])
parser.add_argument('-nesterov', '--nesterov', dest='nesterov', action='store_true',
                    help='nesterov momentum')

parser.add_argument('-densityEstimation', '--densityEstimation', dest='densityEstimation', action='store_true', help='perform density estimation', default=False)
parser.add_argument('--nBlocks', nargs='+', type=int, default=[13, 13, 13])   # 4, 4, 4   
parser.add_argument('--nStrides', nargs='+', type=int, default=[1, 2, 2])   # (12, 48, 192): (1, 2, 2)  / (12, 12, 12): (1, 1, 1)
parser.add_argument('--nChannels', nargs='+', type=int, default=[12, 48, 192])  #16, 64, 256 , (12, 48, 192), (12, 12, 12)
parser.add_argument('--coeff', default=0.9, type=float, help='contraction coefficient for linear layers')
parser.add_argument('--numTraceSamples', default=1, type=int, help='number of samples used for trace estimation')
parser.add_argument('--numSeriesTerms', default=1, type=int, help='number of terms used in power series for matrix log')
parser.add_argument('--powerIterSpectralNorm', default=5, type=int, help='number of power iterations used for spectral norm')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='coefficient for weight decay')
parser.add_argument('--init_batch', default=1024, type=int, help='init batch size')
parser.add_argument('--init_ds', default=2, type=int, help='initial downsampling')  # 2 
parser.add_argument('--inj_pad', default=0, type=int, help='initial inj padding')
parser.add_argument('--warmup_epochs', default=10, type=int, help='epochs for warmup')
args = parser.parse_args()


torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True


train_chain = [
               transforms.Pad(4, padding_mode="symmetric"),
               transforms.RandomCrop(32),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor()]
test_chain = [transforms.ToTensor()]


print("task", "classification")
clf_chain = [transforms.Normalize(mean[args.dataset], std[args.dataset])]
transform_train = transforms.Compose(train_chain + clf_chain)
transform_test = transforms.Compose(test_chain + clf_chain)


# if args.dataset  == 'mnist':
    # assert args.densityEstimation, "Currently mnist is only supported for density estimation"
    # mnist_transforms = [transforms.Pad(2, 0), transforms.ToTensor(), lambda x: x.repeat((3, 1, 1))]
    # transform_train_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
    # transform_test_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
    # trainset = torchvision.datasets.MNIST(
    #     root='./data', train=True, download=True, transform=transform_train_mnist)
    # testset = torchvision.datasets.MNIST(
    #     root='./data', train=False, download=False, transform=transform_test_mnist)
    # nClasses = 10
    # in_shape = (1, 28, 28)  # NOTE (3, 32, 32) --> (1, 28, 28)

if args.dataset == 'cifar10':
    print("dataset", "cifar10")
    if platform.system() == "Darwin":  # MacOS
        root = '/Users/muhaoguo/Documents/study/Paper_Projects/Datasets/cifar10_data'
    elif platform.system() == "Windows":
        root = 'D:\Projects\DATASETS\cifar10_data'
    root = 'cifar10_data'  # todo server
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
    nClasses = 10
    in_shape = (3, 32, 32)


# section: Create a subset of the dataset
##################################################
# subset_indices = list(range(100))
# trainset = Subset(trainset, subset_indices)
# testset = Subset(testset, subset_indices)
##################################################


# section: Model and data Setting
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                          shuffle=True, num_workers=0,
                                          worker_init_fn=np.random.seed(1234))  # num_workers=0
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                         shuffle=False, num_workers=0,
                                         worker_init_fn=np.random.seed(1234))  # num_workers=0


use_cuda = torch.cuda.is_available()
print("use_cuda", use_cuda)





class Conv2d_to_ToeplitzLinear(nn.Module):
    def __init__(self, conv2d, input_shape, device=None):
        super(Conv2d_to_ToeplitzLinear, self).__init__()
        self.conv2d = conv2d
        self.input_shape = input_shape  # (C_in, H_in, W_in)

        # Get convolution parameters
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.kernel_size = conv2d.kernel_size  # (kH, kW)
        self.stride = conv2d.stride  # (sH, sW)
        self.padding = conv2d.padding  # (pH, pW)
        self.dilation = conv2d.dilation  # (dH, dW)
        self.bias = conv2d.bias is not None

        # Compute output dimensions
        self.output_shape = self.compute_output_shape()

        # Construct the Linear layer with Toeplitz weights
        self.construct_linear_layer(device=device)

    def compute_output_shape(self):
        C_in, H_in, W_in = self.input_shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        return (self.out_channels, H_out, W_out)

    def construct_linear_layer(self, device=None):
        C_in, H_in, W_in = self.input_shape
        C_out, H_out, W_out = self.output_shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        # Total number of input and output elements
        input_size = C_in * H_in * W_in
        output_size = C_out * H_out * W_out

        # Initialize the Linear layer
        self.linear = nn.Linear(input_size, output_size, bias=self.bias).to(device)

        # Create the Toeplitz matrix
        # The weight of the linear layer will be the Toeplitz matrix
        toeplitz_matrix = torch.zeros((output_size, input_size), device=device)

        # Get the convolution kernel weights
        weight = self.conv2d.weight.data.to(device)  # Shape: (C_out, C_in, kH, kW)
        if self.bias:
            bias = self.conv2d.bias.data.to(device)  # Shape: (C_out,)
        else:
            bias = torch.zeros(C_out, device=device)

        # For each output position, compute the corresponding input positions
        out_idx = 0
        for oc in range(C_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    # Compute the input region for this output element
                    for ic in range(C_in):
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh * sH - pH + kh * dH
                                iw = ow * sW - pW + kw * dW
                                if 0 <= ih < H_in and 0 <= iw < W_in:
                                    in_idx = ic * H_in * W_in + ih * W_in + iw
                                    weight_val = weight[oc, ic, kh, kw]
                                    toeplitz_matrix[out_idx, in_idx] = weight_val
                    out_idx += 1

        # Assign the Toeplitz matrix as the weight of the linear layer
        self.linear.weight.data = toeplitz_matrix

        if self.bias:
            # Expand the bias to match each output element
            expanded_bias = bias.repeat_interleave(H_out * W_out)
            self.linear.bias.data = expanded_bias
        else:
            self.linear.bias = None

    def forward(self, x):
        # x shape: (N, C_in, H_in, W_in)
        N = x.size(0)
        C_in, H_in, W_in = self.input_shape
        C_out, H_out, W_out = self.output_shape

        # Flatten the input
        x_flat = x.view(N, -1)  # Shape: (N, input_size)

        # Apply the linear layer
        out = self.linear(x_flat)  # Shape: (N, output_size)

        # Reshape the output
        out = out.view(N, C_out, H_out, W_out)
        return out

class LUDecomposedLinear(nn.Module):
    '''
    with mask
    '''
    def __init__(self, linear):
        super(LUDecomposedLinear, self).__init__()
        W = linear.weight.data  # Shape: (out_features, in_features)
        b = linear.bias.data if linear.bias is not None else None

        # Perform LU decomposition on W
        W_numpy = W.cpu().numpy()
        P_numpy, L_numpy, U_numpy = scipy.linalg.lu(W_numpy)
        L = torch.from_numpy(L_numpy).to(W.device).float()
        U = torch.from_numpy(U_numpy).to(W.device).float()
        P = torch.from_numpy(P_numpy).to(W.device).float()

        # Create two Linear layers without bias
        self.linear_U = nn.Linear(U.size(1), U.size(0), bias=False)
        self.linear_L = nn.Linear(L.size(1), L.size(0), bias=False)

        # Assign the decomposed weights
        self.linear_U.weight.data = U
        self.linear_L.weight.data = L

        # Register the triangular masks as buffers
        self.register_buffer('mask_triu', torch.triu(torch.ones_like(U)))  # Upper triangular mask
        self.register_buffer('mask_tril', torch.tril(torch.ones_like(L)))  # Lower triangular mask

        # Store the original bias
        self.bias = b

        # Store permutation matrix P
        self.P = P

        # Ensure gradients are masked
        self.register_hooks()

    def forward(self, x):
        # Apply the first linear layer (U)
        out = self.linear_U(x)  # Shape: [batch_size, mid_features]

        # Apply the second linear layer (L)
        out = self.linear_L(out)  # Shape: [batch_size, out_features]

        # Apply the permutation matrix P
        out = torch.matmul(out, self.P.T)  # Shape: [batch_size, out_features]

        # Add the original bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def register_hooks(self):
        # Hook to apply masks to the gradients after each backward pass
        self.linear_U.weight.register_hook(lambda grad: grad * self.mask_triu)
        self.linear_L.weight.register_hook(lambda grad: grad * self.mask_tril)




# Path where the model was saved
# load_path = './DipDNN_base_models/DipDNN_base_resnet_specific_cifar10_12block_200epoch_12channels.pth'
# load_path = './DipDNN_base_models/DipDNN_base_resnet_specific_cifar10_39block_200epoch_12channels.pth'
# load_path = './DipDNN_base_models/DipDNN_base_resnet_specific_cifar10_39block_200epoch_1248192channels.pth'


# Load the entire model
model = torch.load(load_path)

# If you want to use it for inference
model.eval()  # Set the model to evaluation mode
print(model)


test_objective = -np.inf
loss_epoch, acc = test(test_objective, None, model, None, testloader, use_cuda, None)


print("test loss_epoch mlp_ResNet_step1", loss_epoch)
print("test acc mlp_ResNet_step1", acc)


if args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == "adamax":
    optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)



for epoch in range(1, 1 + args.epochs):
    loss_epoch, acc = train(args, model, optimizer, epoch, trainloader, trainset, use_cuda, None)


test_objective = -np.inf
loss_epoch, acc = test(test_objective, None, model, None, testloader, use_cuda, None)

print("test loss_epoch mlp_ResNet_step1", loss_epoch)
print("test acc mlp_ResNet_step1", acc)