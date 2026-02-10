from get_data import *
from get_data_MNIST2Fashion import *
from get_data_MNIST2Fashion1class import get_MNIST2MNISTFashion1class_data, get_MNIST2MNISTFashion1sample_data
from get_data_MNIST2halfMNISThalfFashion import get_halfMNISThalfFashion, get_halfMNISThalfFashion_1over4, get_halfMNISThalfFashion_1over16, get_halfMNISThalfFashion_veritcal

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
from utils import *
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
parser.add_argument('--save_dir', default="./get_halfMNISThalfFashion_veritcal_iresnet", type=str, help='directory to save results') # MNISTmakeup_iresnet  MNIST2MNISTFashion_iresnet MNIST2MNISTFashion1class_iresnet
parser.add_argument('--dataset', default='halfMNISThalfFashion_veritcal', type=str, help='dataset')  # MNISTmakeup MNIST2MNISTFashion MNIST2MNISTFashion1class

# halfMNISThalfFashion_resnet halfMNISThalfFashion_1over4 halfMNISThalfFashion_1over16  get_halfMNISThalfFashion_veritcal
 
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


#################################################################
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
#################################################################

##################################################################################

class _SpectralNorm(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        item: float = 0 # todo
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")

        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             f'got n_power_iterations={n_power_iterations}')
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        self.item = item # todo
        if ndim > 1:
            # For ndim == 1 we do not need to approximate anything (see _SpectralNorm.forward)
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer('_u', F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer('_v', F.normalize(v, dim=0, eps=self.eps))

            # Start with u, v initialized to some reasonable values by performing a number
            # of iterations of the power method
            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # Precondition
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        # See original note at torch/nn/utils/spectral_norm.py
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallelized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.

        # Precondition
        assert weight_mat.ndim > 1

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            self._u = F.normalize(torch.mv(weight_mat, self._v),      # type: ignore[has-type]
                                  dim=0, eps=self.eps, out=self._u)   # type: ignore[has-type]
            self._v = F.normalize(torch.mv(weight_mat.H, self._u),
                                  dim=0, eps=self.eps, out=self._v)   # type: ignore[has-type]

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.vdot(u, torch.mv(weight_mat, v))
            sigma += self.item
            return weight / sigma

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value


def spectral_norm_customize(module: Module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None,
                  item = 0 # todo
                  ) -> Module:
    r"""Apply spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    When applied on a vector, it simplifies to

    .. math::
        \mathbf{x}_{SN} = \dfrac{\mathbf{x}}{\|\mathbf{x}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by reducing the Lipschitz constant
    of the model. :math:`\sigma` is approximated performing one iteration of the
    `power method`_ every time the weight is accessed. If the dimension of the
    weight tensor is greater than 2, it is reshaped to 2D in power iteration
    method to get spectral norm.


    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`power method`: https://en.wikipedia.org/wiki/Power_iteration
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~torch.nn.utils.parametrize.register_parametrization`. It is a
        reimplementation of :func:`torch.nn.utils.spectral_norm`.

    .. note::
        When this constraint is registered, the singular vectors associated to the largest
        singular value are estimated rather than sampled at random. These are then updated
        performing :attr:`n_power_iterations` of the `power method`_ whenever the tensor
        is accessed with the module on `training` mode.

    .. note::
        If the `_SpectralNorm` module, i.e., `module.parametrization.weight[idx]`,
        is in training mode on removal, it will perform another power iteration.
        If you'd like to avoid this iteration, set the module to eval mode
        before its removal.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter. Default: ``"weight"``.
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm. Default: ``1``.
        eps (float, optional): epsilon for numerical stability in
            calculating norms. Default: ``1e-12``.
        dim (int, optional): dimension corresponding to number of outputs.
            Default: ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with a new parametrization registered to the specified
        weight

    Example::
        # >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        # >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        # >>> snm = spectral_norm(nn.Linear(20, 40))
        # >>> snm
        ParametrizedLinear(
          in_features=20, out_features=40, bias=True
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _SpectralNorm()
            )
          )
        )
        # >>> torch.linalg.matrix_norm(snm.weight, 2)
        tensor(1.0081, grad_fn=<AmaxBackward0>)
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(module, name, _SpectralNorm(weight, n_power_iterations, dim, eps, item))  # todo item
    return module



# Define a small neural network for the residual function
class SmallNetwork(nn.Module):
    def __init__(self, input_dim, use_power_iteration=False):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        # self.fc3 = nn.Linear(input_dim, input_dim)

        # self.activation = nn.ReLU()
        # self.activation = nn.Sigmoid()
        self.activation = nn.ELU()

        if use_power_iteration:
            self.apply_spectral_normalization()

    def apply_spectral_normalization(self):
        # Apply spectral normalization to each linear layer
        # for layer in [self.fc1, self.fc2, self.fc3]:
        for layer in [self.fc1, self.fc2]:
            nn.utils.parametrizations.spectral_norm(layer)
            self.scale_weights_below_one(layer)

    def scale_weights_below_one(self, layer):
        with torch.no_grad():
            u, s, v = torch.svd(layer.weight)
            max_singular_value = s.max().item()

            if max_singular_value >= 0.99:
                spectral_norm_customize(layer, item=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # x = self.activation(self.fc2(x))
        # x = self.fc3(x)
        return x

# Define the invertible residual block
# with options for Lipschitz enforcement
class InvertibleResidualBlock(nn.Module):
    def __init__(self, input_dim, enforce_lipz='orthogonal', alpha=0.9):
        # defaulted as using orthogonal, change with enforce_lipz = 'power_iteration'
        super(InvertibleResidualBlock, self).__init__()
        self.alpha = alpha  # Scaling coefficient
        use_power_iteration = enforce_lipz == 'power_iteration'
        self.residual_function = SmallNetwork(input_dim, use_power_iteration=use_power_iteration)

        if enforce_lipz == 'orthogonal':
            # Ensure Lipschitz continuity using orthogonal initialization with small gain
            for layer in self.residual_function.modules():
                if isinstance(layer, nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, gain=1e-2)

    def forward(self, x):
        return x + self.alpha * self.residual_function(x)

    def inverse(self, y):
        # Use fixed-point iteration to find the inverse
        x = y
        for _ in range(100):
            x = y - self.alpha * self.residual_function(x)  # y = x + af(x) --> x = y
        return x


    # Check if Lipschitz continuty is satisified using spectral norm
    def check_lipz_continuity(self):
        # Check if the Lipschitz constant is < 1 for all layers
        lipz_values = []
        for layer in self.residual_function.modules():
            if isinstance(layer, nn.Linear):
                u, s, v = torch.svd(layer.weight)
                max_singular_value = s.max().item()
                lipz_values.append(max_singular_value)

        all_satisfied = all(lipz < 1 for lipz in lipz_values)
        # print(f"Overall Lipschitz Continuity Satisfied: {all_satisfied}")
        # For entire model, yes or no
        return all_satisfied, lipz_values

# Define the i-ResNet
class iResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, enforce_lipz='orthogonal'):
        super(iResNet, self).__init__()
        self.blocks = nn.ModuleList(
            [InvertibleResidualBlock(input_dim, enforce_lipz=enforce_lipz) for _ in range(num_blocks)])

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for block in self.blocks:
            x = block(x)

        x = x.reshape(x.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))
        return x

    def inverse(self, y):
        y = y.reshape(y.shape[0], -1)  # (batchsize, -1)

        for block in reversed(self.blocks):
            y = block.inverse(y)
        
        y = y.reshape(y.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))
        return y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def check_lipz_continuity(self):
        # Check if Lipschitz continuity is satisfied for all blocks
        all_satisfied = True
        for i, block in enumerate(self.blocks):
            # print(f"Checking Lipschitz continuity for block {i + 1}:")
            satisfied, lipz_values = block.check_lipz_continuity()
            all_satisfied = all_satisfied and satisfied
            # print(f"Lipz values for block {i + 1}: {lipz_values}")
        print(f"Overall Network Lipschitz Continuity Satisfied: {all_satisfied}")
        return all_satisfied



image_l = 28 # paired_mnist_train.original_dataset[0][0].size[0]
input_dim = image_l * image_l  # Flattened MNIST images 28 * 28
hidden_dim = 128
num_blocks = 6
enforce_lipz = 'power_iteration'  # or 'power_iteration'
learning_rate = 1e-3
num_epochs = 2000

# Initialize the iResNet model
model = iResNet(input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=num_blocks, enforce_lipz=enforce_lipz).to(device)
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




visualize_results(model, train_loader, device, num_samples=3, save_path=f"{args.save_dir}/{args.dataset}_Samples_train.png")
visualize_results(model, test_loader, device, num_samples=3, save_path=f"{args.save_dir}/{args.dataset}_Samples_test.png")












