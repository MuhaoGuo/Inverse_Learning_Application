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
from conv_iResNet import conv_ResNet as ResNet
from conv_iResNet import conv_ResNet_specific
# from models.conv_iResNet import multiscale_conv_iResNet as multiscale_iResNet
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import platform
import scipy.linalg
import copy


parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
parser.add_argument('--save_dir', default="./resnet_specific_cifar10_39block_200epoch_1248192channels", type=str, help='directory to save results') # mnist cifar10
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')  # mnist cifar10

parser.add_argument('--epochs', default=200, type=int, help='number of epochs')  # 200
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
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


noActnorm = False
fixedPrior = False
nonlin = 'elu'
model = conv_ResNet_specific(nBlocks=args.nBlocks,
                nStrides=args.nStrides,
                nChannels=args.nChannels,
                nClasses=nClasses,
                init_ds=args.init_ds,
                inj_pad=args.inj_pad,
                in_shape=in_shape,
                coeff=args.coeff,
                numTraceSamples=args.numTraceSamples,
                numSeriesTerms=args.numSeriesTerms,
                n_power_iter=args.powerIterSpectralNorm,
                density_estimation=args.densityEstimation,
                actnorm=(not noActnorm),
                learn_prior=(not fixedPrior),
                nonlin=nonlin)

# print(model)
# exit()



if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

test_objective = -np.inf

def get_init_batch(dataloader, batch_size):
    """
    gets a batch to use for init
    """
    batches = []
    seen = 0
    for x, y in dataloader:
        batches.append(x)
        seen += x.size(0)
        if seen >= batch_size:
            break
    batch = torch.cat(batches)
    return batch

# init actnrom parameters
init_batch = get_init_batch(trainloader, args.init_batch)
print("initializing actnorm parameters...")
with torch.no_grad():
    model(init_batch, ignore_logdet=True)
print("initialized")

use_cuda = torch.cuda.is_available()
print("use_cuda", use_cuda)
if use_cuda:
    model.cuda()
    # model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
    # cudnn.benchmark = True
    # in_shapes = model.module.get_in_shapes()
    in_shapes = model.get_in_shapes()
else:
    in_shapes = model.get_in_shapes()



###############################################################################################################
###############################################################################################################
# load model  -----------------------------------
state_dict = torch.load(args.save_dir + '/model_state_dict.pth')
model.load_state_dict(state_dict)

# section testing 
# ###################################### ###################################### #####################################
print('------------- original model--------------------------')

test_objective = -np.inf
test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
loss_epoch, acc = test(test_objective, args, model, args.epochs, testloader, use_cuda, test_log)

print(model)
print("test loss_epoch", loss_epoch)
print("test acc", acc)

# ###################################### ###################################### #####################################

# Define a custom module to replace Conv2d with Linear layers
# class Conv2d_to_Linear(nn.Module):
#     def __init__(self, conv2d):
#         super(Conv2d_to_Linear, self).__init__()

#         # Get parameters from the Conv2d layer
#         self.in_channels = conv2d.in_channels
#         self.out_channels = conv2d.out_channels
#         self.kernel_size = conv2d.kernel_size
#         self.stride = conv2d.stride
#         self.padding = conv2d.padding
#         self.dilation = conv2d.dilation
#         self.bias = conv2d.bias is not None

#         # Create an unfold operation to extract sliding local blocks from the input
#         self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=self.dilation,
#                                 padding=self.padding, stride=self.stride)

#         # Reshape the Conv2d weights to match the Linear layer
#         weight = conv2d.weight.data
#         weight = weight.view(self.out_channels, -1)

#         # Initialize the Linear layer
#         self.linear = nn.Linear(weight.shape[1], self.out_channels, bias=self.bias)
#         self.linear.weight.data = weight
#         if self.bias:
#             self.linear.bias.data = conv2d.bias.data

#     def forward(self, x):
#         # Unfold the input to match the Linear layer input
#         x_unfold = self.unfold(x)  # Shape: (N, C_in * K_h * K_w, L)
#         x_unfold = x_unfold.transpose(1, 2)  # Shape: (N, L, C_in * K_h * K_w)

#         # Apply the Linear layer
#         out = self.linear(x_unfold)  # Shape: (N, L, out_channels)

#         # Compute the output dimensions
#         H_out = (x.size(2) + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
#         W_out = (x.size(3) + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

#         # Reshape the output to match Conv2d output shape
#         out = out.transpose(1, 2).contiguous()
#         out = out.view(x.size(0), self.out_channels, H_out, W_out)
#         return out


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

# Original Conv2d layer
# conv_layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
# conv_layer.weight.data = torch.randn_like(conv_layer.weight.data)
# conv_layer.bias.data = torch.randn_like(conv_layer.bias.data)

# # Input tensor
# x = torch.randn(1, 3, 32, 32)

# # Output from original Conv2d layer
# output_conv = conv_layer(x)

# # Toeplitz-based Linear layer
# toeplitz_layer = Conv2d_to_ToeplitzLinear(conv_layer, input_shape=(3, 32, 32))

# # Output from Toeplitz-based Linear layer
# output_toeplitz = toeplitz_layer(x)

# # Compute the difference
# difference = torch.norm(output_conv - output_toeplitz)
# print("Difference between outputs:", difference.item())
# exit()


in_shapes = [(12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16), (12, 16, 16),
              (48, 8, 8), (48, 8, 8), (48, 8, 8), (48, 8, 8), (48, 8, 8), (48, 8, 8), (48, 8, 8), (48, 8, 8), (48, 8, 8), (48, 8, 8), (48, 8, 8), (48, 8, 8), (48, 8, 8), 
              (192, 4, 4), (192, 4, 4), (192, 4, 4), (192, 4, 4), (192, 4, 4), (192, 4, 4), (192, 4, 4), (192, 4, 4), (192, 4, 4), (192, 4, 4), (192, 4, 4), (192, 4, 4)]

# Function to recursively replace Conv2d layers with Conv2d_to_ToeplitzLinear
def replace_conv2d_with_linear(model):
    i = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            input_shape=(module.in_channels, int(np.sqrt(3* 32 * 32/ module.in_channels)), int(np.sqrt(3* 32 * 32/ module.in_channels))) 
            print(f"{i}, {name}, converting: {input_shape}")
            i += 1
            setattr(model, name, Conv2d_to_ToeplitzLinear(conv2d = module, input_shape=input_shape, device="cuda"))
        else:
            replace_conv2d_with_linear(module)  # Recursively apply to child modules

mlp_ResNet_step1 = copy.deepcopy(model) 

replace_conv2d_with_linear(mlp_ResNet_step1)

print("----------- mlp_ResNet_step1 -------------------")
print('Testing mlp_ResNet_step1 model (replace conv to linear) -------------------------')
test_objective = -np.inf
test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
loss_epoch, acc = test(test_objective, args, mlp_ResNet_step1, args.epochs, testloader, use_cuda, test_log)


print(mlp_ResNet_step1)
print("test loss_epoch mlp_ResNet_step1", loss_epoch)
print("test acc mlp_ResNet_step1", acc)

# ###################################### ###################################### #####################################
# Define a custom module to split a Linear layer into two Linear layers
class SplitLinear_plus(nn.Module):
    def __init__(self, linear):
        super(SplitLinear_plus, self).__init__()
        W = linear.weight.data  # Shape: (out_features, in_features)
        b = linear.bias.data if linear.bias is not None else None

        # Decompose W into lower and upper triangular matrices
        L = torch.tril(W, diagonal=-1)  # Exclude the diagonal
        U = torch.triu(W)               # Include the diagonal

        # Create two Linear layers
        self.linear_L = nn.Linear(W.shape[1], W.shape[0], bias=False)
        self.linear_U = nn.Linear(W.shape[1], W.shape[0], bias=(b is not None))

        # Assign the decomposed weights
        self.linear_L.weight.data = L
        self.linear_U.weight.data = U

        # Assign the bias to one of the layers (e.g., upper triangular part)
        if b is not None:
            self.linear_U.bias.data = b
            # Set the other bias to zero if needed
            # self.linear_L.bias.data = torch.zeros_like(b)
        else:
            self.linear_U.bias = None

    def forward(self, x):
        # Compute outputs from both linear layers and sum them
        out_L = self.linear_L(x)
        out_U = self.linear_U(x)
        return out_L + out_U
    
        # out_L = self.linear_L(x)
        # out_U = self.linear_U(out_L)
        # return out_U


def replace_conv_linear_with_splitlinear_plus(model):
    for name, module in model.named_children():
        if isinstance(module, Conv2d_to_ToeplitzLinear):
            # Replace the internal Linear layer with SplitLinear
            module.linear = SplitLinear_plus(module.linear)
        else:
            # Recursively apply to child modules
            replace_conv_linear_with_splitlinear_plus(module)


class SplitLinear(nn.Module):
    def __init__(self, linear):
        super(SplitLinear, self).__init__()
        W = linear.weight.data.clone()  # Shape: (out_features, in_features)
        b = linear.bias.data.clone() if linear.bias is not None else None

        # Ensure W is a square matrix by padding if necessary
        out_features, in_features = W.shape
        max_features = max(out_features, in_features)

        # Pad W to make it square
        padded_W = torch.zeros((max_features, max_features), device=W.device, dtype=W.dtype)
        padded_W[:out_features, :in_features] = W

        # Decompose W into lower and upper triangular matrices
        L = torch.tril(padded_W)
        U = torch.triu(padded_W)

        # Create two Linear layers
        self.linear_L = nn.Linear(max_features, max_features, bias=False)
        self.linear_U = nn.Linear(max_features, max_features, bias=(b is not None))

        # Assign the decomposed weights
        self.linear_L.weight.data = L
        self.linear_U.weight.data = U

        # Assign the bias to the second layer
        if b is not None:
            # Pad bias to match the max_features size
            padded_b = torch.zeros(max_features, device=b.device, dtype=b.dtype)
            padded_b[:out_features] = b
            self.linear_U.bias.data = padded_b
        else:
            self.linear_U.bias = None

        # Store original output size
        self.out_features = out_features

    def forward(self, x):
        # Pad input x to match max_features
        batch_size = x.size(0)
        in_features = x.size(1)
        max_features = self.linear_L.in_features

        padded_x = torch.zeros(batch_size, max_features, device=x.device, dtype=x.dtype)
        padded_x[:, :in_features] = x

        # Apply the two linear layers sequentially
        out_L = self.linear_L(padded_x)
        out_U = self.linear_U(out_L)

        # Trim the output to the original out_features
        out = out_U[:, :self.out_features]
        return out

# Function to replace linear layers within Conv2d_to_Linear modules
def replace_conv_linear_with_splitlinear(model):
    for name, module in model.named_children():
        if isinstance(module, Conv2d_to_ToeplitzLinear):
            # Replace the internal Linear layer with SplitLinear
            module.linear = SplitLinear(module.linear)
        else:
            # Recursively apply to child modules
            replace_conv_linear_with_splitlinear(module)

# ---------------------
# Define a custom module to replace a Linear layer with two Linear layers using LU decomposition

# class LUDecomposedLinear(nn.Module):
#     def __init__(self, linear):
#         super(LUDecomposedLinear, self).__init__()
#         W = linear.weight.data  # Shape: (out_features, in_features)
#         b = linear.bias.data if linear.bias is not None else None

#         # Perform LU decomposition on W
#         W_numpy = W.cpu().numpy()
#         P_numpy, L_numpy, U_numpy = scipy.linalg.lu(W_numpy)
#         L = torch.from_numpy(L_numpy).to(W.device).float()
#         U = torch.from_numpy(U_numpy).to(W.device).float()
#         P = torch.from_numpy(P_numpy).to(W.device).float()

#         # Create two Linear layers without bias
#         self.linear_U = nn.Linear(U.size(1), U.size(0), bias=False)
#         self.linear_L = nn.Linear(L.size(1), L.size(0), bias=False)

#         # Assign the decomposed weights
#         self.linear_U.weight.data = U
#         self.linear_L.weight.data = L

#         # Store the original bias
#         self.bias = b

#         # Store permutation matrix P
#         self.P = P


#     def forward(self, x):
#         # print("Input shape:", x.shape)
#         # x has shape [batch_size, features] (features = in_features)
#         batch_size, in_features = x.shape

#         # No need to reshape x since it's already in [batch_size, features] format
#         # Apply the first linear layer (U)
#         out = self.linear_U(x)  # Shape: [batch_size, mid_features]

#         # Apply the second linear layer (L)
#         out = self.linear_L(out)  # Shape: [batch_size, out_features]

#         # Apply the permutation matrix P
#         # P has shape [out_features, out_features], we can apply it directly
#         out = torch.matmul(out, self.P.T)  # Shape: [batch_size, out_features]

#         # Add the original bias
#         if self.bias is not None:
#             out = out + self.bias

#         return out


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



### verify LUDecomposedLinear: 
# -------------------------------------------------------------------------
# Define the original Linear layer
linear_layer = nn.Linear(10, 10)  # Example: in_features=10, out_features=10

# Create the LUDecomposedLinear layer from the original Linear layer
decomposed_layer = LUDecomposedLinear(linear_layer)

# Generate some random input data
batch_size, features = 32, 10  # Example input dimensions
input_data = torch.randn(batch_size, features)

# Pass the input through the original Linear layer
original_output = linear_layer(input_data)

# Pass the input through the LUDecomposedLinear layer
decomposed_output = decomposed_layer(input_data)

# Compare the outputs
are_equal = torch.allclose(original_output, decomposed_output, atol=1e-6)
print(f"Are the original and decomposed layers equivalent? {are_equal}")
# -------------------------------------------------------------------------


# Function to replace linear layers within Conv2d_to_Linear modules with LUDecomposedLinear
def replace_conv_linear_with_lu_linear(model):
    for name, module in model.named_children():
        if isinstance(module, Conv2d_to_ToeplitzLinear):
            # Replace the internal Linear layer with LUDecomposedLinear
            module.linear = LUDecomposedLinear(module.linear)
        else:
            # Recursively apply to child modules
            replace_conv_linear_with_lu_linear(module)



print("----------- DipDNN -------------------")


method = "LUDecomposedLinear"
if method =="LUDecomposedLinear":
    replace_conv_linear_with_lu_linear(mlp_ResNet_step1)
elif method =="SplitLinear":
    replace_conv_linear_with_splitlinear(mlp_ResNet_step1)
elif method == "SplitLinear_plus":
    replace_conv_linear_with_splitlinear_plus(mlp_ResNet_step1)

print(mlp_ResNet_step1)




test_objective = -np.inf
test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
loss_epoch, acc = test(test_objective, args, mlp_ResNet_step1, args.epochs, testloader, use_cuda, test_log)


print("test loss_epoch mlp_ResNet_step1", loss_epoch)
print("test acc mlp_ResNet_step1", acc)




## saving DipDNN init models =========================================
DipDNN_name = args.save_dir[2:]
save_path = './DipDNN_base_models/DipDNN_base_{}.pth'.format(DipDNN_name)
# torch.save(mlp_ResNet_step1, save_path)
torch.save(mlp_ResNet_step1.state_dict(), save_path)
## saving DipDNN init models dict=========================================



