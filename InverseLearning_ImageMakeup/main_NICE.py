from get_data import *
from get_data_MNIST2Fashion import *
from get_data_MNIST2Fashion1class import get_MNIST2MNISTFashion1class_data, get_MNIST2MNISTFashion1sample_data
from get_data_MNIST2halfMNISThalfFashion import get_halfMNISThalfFashion, get_halfMNISThalfFashion_1over4, get_halfMNISThalfFashion_1over16, get_halfMNISThalfFashion_veritcal

from utils import *
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

parser = argparse.ArgumentParser(description='Train model on MNIST convert')
parser.add_argument('--save_dir', default="./halfMNISThalfFashion_1over16_nice", type=str, help='directory to save results') # MNISTmakeup_nice  MNIST2MNISTFashion_resnet MNIST2MNISTFashion1class_resnet
parser.add_argument('--dataset', default='halfMNISThalfFashion_1over16', type=str, help='dataset')  # MNISTmakeup MNIST2MNISTFashion MNIST2MNISTFashion1class

# halfMNISThalfFashion_resnet halfMNISThalfFashion_1over4 halfMNISThalfFashion_1over16  halfMNISThalfFashion_veritcal

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
#################################################################
if args.dataset  == "MNISTmakeup":
    # Load MNIST dataset
    data_path = '../data/'
    mnist_train = torchvision.datasets.MNIST(root=data_path, train=True, download=False, )
    mnist_test = torchvision.datasets.MNIST(root=data_path, train=False, download=False, )

    #----------------------------------------------------------
    # section: Create a subset of the dataset
    from torch.utils.data import Subset
    subset_indices = list(range(30))
    mnist_train = Subset(mnist_train, subset_indices)
    mnist_test = Subset(mnist_test, subset_indices)
    # --------------------------------------------------------

    # Define any additional transforms (e.g., normalization)
    additional_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create the paired datasets
    paired_mnist_train = PairedMNISTDataset(mnist_train, transform=additional_transforms)
    paired_mnist_test = PairedMNISTDataset(mnist_test, transform=additional_transforms)
    print(paired_mnist_train.original_dataset[0][0].size)
    print(paired_mnist_train.original_dataset[0][1])


    # Define batch size
    batch_size = 128

    # Create DataLoaders for training and testing
    train_loader = DataLoader(paired_mnist_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(paired_mnist_test, batch_size=batch_size, shuffle=False)

elif args.dataset  == "MNIST2MNISTFashion":
    mnist_to_fashionmnist_mapping = {
        0: 0,  # MNIST (0) -> FashionMNIST (T-shirt/top)
        1: 1,  # MNIST (1) -> FashionMNIST (Trouser)
        2: 2,  # MNIST (2) -> FashionMNIST (Pullover)
        3: 3,  # MNIST (3) -> FashionMNIST (Dress)
        4: 4,  # MNIST (4) -> FashionMNIST (Coat)
        5: 5,  # MNIST (5) -> FashionMNIST (Sandal)
        6: 6,  # MNIST (6) -> FashionMNIST (Shirt)
        7: 7,  # MNIST (7) -> FashionMNIST (Sneaker)
        8: 8,  # MNIST (8) -> FashionMNIST (Bag)
        9: 9   # MNIST (9) -> FashionMNIST (Ankle boot)
    }
    # 2. Define the transformation to convert PIL Images to Tensors
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts PIL Image to Tensor and scales pixel values to [0, 1]
    ])

    data_path = '../data'
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    fashionmnist_train = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    fashionmnist_test = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)

    # 6. Function to group FashionMNIST indices by label
    def group_fashionmnist_by_label(fashionmnist_dataset):
        label_to_indices = defaultdict(list)
        for idx, (image, label) in enumerate(fashionmnist_dataset):
            label_to_indices[label].append(idx)
        return label_to_indices

    # 7. Create the label to indices mapping for FashionMNIST
    fashion_label_to_indices_train = group_fashionmnist_by_label(fashionmnist_train)
    fashion_label_to_indices_test = group_fashionmnist_by_label(fashionmnist_test)


    # 9. Create the paired datasets
    paired_train_dataset = PairedMNISTFashionMNIST(
        mnist_dataset=mnist_train,
        fashionmnist_dataset=fashionmnist_train,
        label_mapping=mnist_to_fashionmnist_mapping,
        fashion_label_to_indices=fashion_label_to_indices_train,
        transform_input=None,  # Already transformed with ToTensor()
        transform_target=None, # Already transformed with ToTensor()
        shuffle=True
    )

    paired_test_dataset = PairedMNISTFashionMNIST(
        mnist_dataset=mnist_test,
        fashionmnist_dataset=fashionmnist_test,
        label_mapping=mnist_to_fashionmnist_mapping,
        fashion_label_to_indices=fashion_label_to_indices_test,
        transform_input=None,
        transform_target=None,
        shuffle=True
    )


    #----------------------------------------------------------
    # section: Create a subset of the dataset
    from torch.utils.data import Subset
    subset_indices = list(range(30))
    paired_train_dataset = Subset(paired_train_dataset, subset_indices)
    paired_test_dataset = Subset(paired_test_dataset, subset_indices)
    # --------------------------------------------------------


    # 10. Create DataLoaders for the paired datasets
    batch_size = 128

    train_loader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False)

elif args.dataset  == "MNIST2MNISTFashion1class":
    # train_loader, test_loader = get_MNIST2MNISTFashion1class_data(samples = 30, batch_size = 128)
    train_loader, test_loader = get_MNIST2MNISTFashion1sample_data(samples = 10, batch_size = 64)
elif args.dataset  == "halfMNISThalfFashion":
    train_loader, test_loader = get_halfMNISThalfFashion(samples = 30, batch_size = 64)

elif args.dataset  == "halfMNISThalfFashion_1over4":
    train_loader, test_loader = get_halfMNISThalfFashion_1over4(samples = 30, batch_size = 64)

elif args.dataset  == "halfMNISThalfFashion_1over16":
    train_loader, test_loader = get_halfMNISThalfFashion_1over16(samples = 30, batch_size = 64)

elif args.dataset  == "halfMNISThalfFashion_veritcal":
    train_loader, test_loader = get_halfMNISThalfFashion_veritcal(samples = 30, batch_size = 64)

#################################################################
##################################################################################



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
        x = x.reshape(x.shape[0], -1)  # (batchsize, -1)
        for layer in self.layers:
            x = layer(x)
        x = x * self.scaling_factor

        if args.dataset == "Cifar10":
            x = x.reshape(x.shape[0], 3, 32, 32)
        else:
            x = x.reshape(x.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
            
        return x

    def inverse(self, y):
        """
        Inverse pass through the NICEResNet model.

        Args:
            y (Tensor): Output tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Reconstructed input tensor.
        """
        y = y.reshape(y.shape[0], -1)  # (batchsize, -1)
        y = y / self.scaling_factor
        for layer in reversed(self.layers):
            y = layer.inverse(y)
                
        if args.dataset == "Cifar10":
            y = y.reshape(y.shape[0], 3, 32, 32)
        else:
            y = y.reshape(y.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        return y

    def count_parameters(self):
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




image_l = 28 # paired_mnist_train.original_dataset[0][0].size[0]
input_dim = image_l * image_l  # Flattened MNIST images 28 * 28
hidden_dim = 128
num_blocks = 3
enforce_lipz = 'power_iteration'  # or 'power_iteration'
learning_rate = 1e-3
num_epochs = 2000

# Initialize the iResNet model
model = NICEResNet(input_dim=input_dim, num_blocks=num_blocks).to(device)
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



visualize_results(model, train_loader, device, num_samples=30, save_path=f"{args.save_dir}/{args.dataset}_Samples_train.png")
visualize_results(model, test_loader, device, num_samples=30, save_path=f"{args.save_dir}/{args.dataset}_Samples_test.png")


