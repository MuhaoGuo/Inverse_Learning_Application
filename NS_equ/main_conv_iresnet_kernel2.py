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
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.modules import Module
from torch.nn import functional as F
from typing import Optional

def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility in convolutional layers
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility


# Set the seed before initializing the model or any random operations
set_seed(42)  # You can use any seed value of your choice

datapath = "/Users/muhaoguo/Documents/study/Paper_Projects/Datasets/Data_Jiaqi/data"

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


file_pattern = '{}/chemical_*.csv'.format(datapath)
files = glob.glob(file_pattern)
files = sorted(files)

data_all = pd.DataFrame()
pattern = r'{}/chemical_(\d+)\.csv'.format(datapath)

for file in files:
    match = re.search(pattern, file)
    if match:
        time_step = int(match.group(1))
    else:
        raise ValueError('Missing time step: ' + file)
    data = pd.read_csv(file)
    data.drop(columns=['z', 'wz'], inplace=True)
    data['t'] = time_step
    data_all = pd.concat([data_all, data], ignore_index=True)


inputs = data_all[['x', 'y', 't', 'wx', 'wy']].values
targets = data_all[['uA', 'uB', 'uC', 'uA', 'uB']].values


inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)
dataset = TensorDataset(inputs, targets)

batch_size = 24460
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



##########################


class _SpectralNorm(Module):
    def __init__(
            self,
            weight: torch.Tensor,
            n_power_iterations: int = 1,
            dim: int = 0,
            eps: float = 1e-12,
            item: float = 0  # todo
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
        self.item = item  # todo
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
            self._u = F.normalize(torch.mv(weight_mat, self._v),  # type: ignore[has-type]
                                  dim=0, eps=self.eps, out=self._u)  # type: ignore[has-type]
            self._v = F.normalize(torch.mv(weight_mat.H, self._u),
                                  dim=0, eps=self.eps, out=self._v)  # type: ignore[has-type]

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
                            item=0  # todo
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
    parametrize.register_parametrization(module, name,
                                         _SpectralNorm(weight, n_power_iterations, dim, eps, item))  # todo item
    return module


# Define a small neural network for the residual function
class SmallNetworkConv1D(nn.Module):
    def __init__(self, input_dim, use_power_iteration=False):
        super(SmallNetworkConv1D, self).__init__()
        # self.fc1 = nn.Linear(input_dim, input_dim)
        # self.fc2 = nn.Linear(input_dim, input_dim)
        # # self.fc3 = nn.Linear(input_dim, input_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, padding=0)

        # self.activation = nn.ReLU()
        # self.activation = nn.Sigmoid()
        self.activation = nn.ELU()

        if use_power_iteration:
            self.apply_spectral_normalization()

    def apply_spectral_normalization(self):
        # Apply spectral normalization to each linear layer
        # for layer in [self.fc1, self.fc2, self.fc3]:
        for layer in [self.conv1, self.conv2]:
            nn.utils.parametrizations.spectral_norm(layer)
            self.scale_weights_below_one(layer)

    def scale_weights_below_one(self, layer):
        with torch.no_grad():
            u, s, v = torch.svd(layer.weight)
            max_singular_value = s.max().item()

            if max_singular_value >= 0.99:
                spectral_norm_customize(layer, item=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        # x = self.activation(self.fc2(x))
        # x = self.fc3(x)
        return x


# Define the invertible residual block
# with options for Lipschitz enforcement
class InvertibleResidualBlockConv1D(nn.Module):
    def __init__(self, input_dim, enforce_lipz='orthogonal', alpha=0.9):
        # defaulted as using orthogonal, change with enforce_lipz = 'power_iteration'
        super(InvertibleResidualBlockConv1D, self).__init__()
        self.alpha = alpha  # Scaling coefficient
        use_power_iteration = enforce_lipz == 'power_iteration'
        self.residual_function = SmallNetworkConv1D(input_dim, use_power_iteration=use_power_iteration)

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
class iResNetConv1D(nn.Module):
    def __init__(self, input_dim, num_blocks, enforce_lipz='power_iteration'):
        super(iResNetConv1D, self).__init__()
        self.blocks = nn.ModuleList(
            [InvertibleResidualBlockConv1D(input_dim, enforce_lipz=enforce_lipz) for _ in range(num_blocks)])

    def forward(self, x):
        # x = x.reshape(x.shape[0], -1)  # (batchsize, -1)
        x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        x = x.squeeze(1)
        # x = x.reshape(x.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))
        return x

    def inverse(self, y):
        # y = y.reshape(y.shape[0], -1)  # (batchsize, -1)
        y = y.unsqueeze(1)
        for block in reversed(self.blocks):
            y = block.inverse(y)
        y = y.squeeze(1)
        # y = y.reshape(y.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))
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
##########################


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
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # model.apply_weight_masks()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))
torch.manual_seed(1234)


input_dim = 5
num_blocks = 3
learning_rate = 1e-3
# num_epochs = 5000
epochs = 400
time = 60

device = get_device()
device = torch.device('cpu')
model_name = "conv_iresnet_kernel2"
model = iResNetConv1D(input_dim=input_dim, num_blocks=num_blocks).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

### train model #########################
train_model(model, train_loader, optimizer, criterion, epochs, device)

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


### test model #########################

data_selected = data_all[data_all['t'] == time]
test_input = data_selected[['x', 'y', 't', 'wx', 'wy']].values
test_target = data_selected[['uA', 'uB', 'uC', 'uA', 'uB']].values

test_input = torch.tensor(test_input, dtype=torch.float32).to(device)
test_target = torch.tensor(test_target, dtype=torch.float32).to(device)

test_pred = model(test_input)
inverse_of_fx = model.inverse(test_pred)
inverse_of_y = model.inverse(test_target.to(device))

print(test_pred.shape)
print(inverse_of_fx.shape)
print(inverse_of_y.shape)

test_pred = test_pred.cpu().detach().numpy()
inverse_of_fx = inverse_of_fx.cpu().detach().numpy()
inverse_of_y = inverse_of_y.cpu().detach().numpy()


# Concatenate arrays along the columns (axis=1)
combined_data = np.concatenate((test_pred, inverse_of_fx, inverse_of_y), axis=1)
# print(combined_data.shape)
# Define column names for each array
columns = ['uA_pred', 'uB_pred', 'uC_pred', 'uA_notuse', 'uB_notuse',
           'x_inv', 'y_inv', 't_inv', 'wx_inv', 'wy_inv',
           'x_inv_real', 'y_inv_real', 't_inv_real', 'wx_inv_real', 'wy_inv_real']

data_pred = pd.DataFrame(combined_data, columns=columns)


x = data_selected['x'].values
y = data_selected['y'].values
wx = data_selected['wx'].values
wy = data_selected['wy'].values
t = data_selected['t'].values

uA_true = data_selected['uA'].values
uA_pred = data_pred['uA_pred'].values
uA_error = (uA_true - uA_pred) ** 2
uB_true = data_selected['uB'].values
uB_pred = data_pred['uB_pred'].values
uB_error = (uB_true - uB_pred) ** 2
uC_true = data_selected['uC'].values
uC_pred = data_pred['uC_pred'].values
uC_error = (uC_true - uC_pred) ** 2


x_inv = data_pred['x_inv'].values
y_inv = data_pred['y_inv'].values
t_inv = data_pred['t_inv'].values
wx_inv = data_pred['wx_inv'].values
wy_inv = data_pred['wy_inv'].values

x_inv_real = data_pred['x_inv_real'].values
y_inv_real = data_pred['y_inv_real'].values
t_inv_real = data_pred['t_inv_real'].values
wx_inv_real = data_pred['wx_inv_real'].values
wy_inv_real = data_pred['wy_inv_real'].values

x_error = (x - x_inv) ** 2
x_error_real = (x - x_inv_real) ** 2
y_error = (y - y_inv) ** 2
y_error_real = (y - y_inv_real) ** 2
t_error = (t - t_inv) ** 2
t_error_real = (t - t_inv_real) ** 2
wx_error = (wx - wx_inv) ** 2
wx_error_real = (wx - wx_inv_real) ** 2
wy_error = (wy - wy_inv) ** 2
wy_error_real = (wy - wy_inv_real) ** 2


# Add 'x' and 'y' as new columns to the existing DataFrame
data_pred['uA_true'] = uA_true
data_pred['uB_true'] = uB_true
data_pred['uC_true'] = uC_true
data_pred['x'] = x
data_pred['y'] = y
data_pred['wx'] = wx
data_pred['wy'] = wy
data_pred['t'] = t

data_pred['uA_error'] = uA_error
data_pred['uB_error'] = uB_error
data_pred['uC_error'] = uC_error

data_pred['x_error'] = x_error
data_pred['y_error'] = y_error
data_pred['t_error'] = t_error
data_pred['wx_error'] = wx_error
data_pred['wy_error'] = wy_error

data_pred['x_error_real'] = x_error_real
data_pred['y_error_real'] = y_error_real
data_pred['t_error_real'] = t_error_real
data_pred['wx_error_real'] = wx_error_real
data_pred['wy_error_real'] = wy_error_real

csv_file_path = f'./results/{model_name}_data_selected_{time}.csv'
data_pred.to_csv(csv_file_path, index=False)

print(data_pred.columns)
