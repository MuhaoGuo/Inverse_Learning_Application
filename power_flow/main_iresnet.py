# Code cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *
import json  # Add this import for handling JSON files
import matplotlib.pyplot as plt  # Ensure matplotlib is imported for plotting
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


# Code cell
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


device = get_device()

# Code cell
file_path = 'case8_results.csv'
data = pd.read_csv(file_path)
X, y = preprocess_data(data)

# Normalize the inputs and outputs
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# scaler_X = MinMaxScaler()
# scaler_y = MinMaxScaler()

# X_normalized = scaler_X.fit_transform(X)
# y_normalized = scaler_y.fit_transform(y)

# X = scaler_X.fit_transform(X)
# y = scaler_y.fit_transform(y)


# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

X_test, y_test = X_train, y_train

# Display the shape of the processed data to ensure no bus_11 columns were dropped
print(X_train.shape, y_train.shape)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)



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
    def __init__(self, input_dim, num_blocks, enforce_lipz='power_iteration'):
        super(iResNet, self).__init__()
        self.blocks = nn.ModuleList(
            [InvertibleResidualBlock(input_dim, enforce_lipz=enforce_lipz) for _ in range(num_blocks)])

    def forward(self, x):
        # x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for block in self.blocks:
            x = block(x)

        # x = x.reshape(x.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))
        return x

    def inverse(self, y):
        # y = y.reshape(y.shape[0], -1)  # (batchsize, -1)

        for block in reversed(self.blocks):
            y = block.inverse(y)

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


# Code cell
# Set input and output size based on the dataset
input_size = X_train.shape[1]
output_size = y_train.shape[1]
print("input_size", input_size)
print("output_size", output_size)
block_num = 3
epochs = 20000
model_name = "iresnet"

# Initialize the model, loss function, and optimizer
model = iResNet(input_size, block_num)
print(model)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_history = []

# Training loop

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.10f}')

# Code cell
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), loss_history, label='Training Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (Log Scale)')
plt.title('Log-Scaled Loss History During Training')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f"./results/{model_name}_powerflow_loss.png")




# Evaluation, Inversion, MSE Calculations, and JSON Saving (Modified)
# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)  # [batch_size, output_size]
    x_inv = model.inverse(y_pred)  # Inverse from y_pred
    x_inv_real = model.inverse(y_test_tensor)  # Inverse from y

# Inverse transform the predictions to recover original values
# y_pred_original = scaler_y.inverse_transform(y_pred.cpu().numpy())  # for MaxMin scale
# y_test_original = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())
# x_inv_original = scaler_X.inverse_transform(x_inv.cpu().numpy())
# x_inv_real_original = scaler_X.inverse_transform(x_inv_real.cpu().numpy())
# X_test_original = scaler_X.inverse_transform(X_test)

y_pred_original = y_pred.cpu().numpy()  # for MaxMin scale
y_test_original = y_test_tensor.cpu().numpy()
x_inv_original = x_inv.cpu().numpy()
x_inv_real_original =x_inv_real.cpu().numpy()
X_test_original = X_test


# Calculate MSEs and convert them to native Python floats
mse_y = float(np.mean((y_pred_original - y_test_original) ** 2))
mse_x_inv = float(np.mean((x_inv_original - X_test) ** 2))
mse_x_inv_real = float(np.mean((x_inv_real_original - X_test) ** 2))

# Calculate MAPE and convert them to native Python floats
mape_y = float(np.mean(np.abs((y_pred_original - y_test_original) / y_test_original)) * 100)
mape_x_inv = float(np.mean(np.abs((x_inv_original - X_test) / X_test)) * 100)
mape_x_inv_real = float(np.mean(np.abs((x_inv_real_original - X_test) / X_test)) * 100)

# Print MSEs
print(f'MSE(y, y_pred): {mse_y}')
print(f'MSE(x_inv, x): {mse_x_inv}')
print(f'MSE(x_inv_real, x): {mse_x_inv_real}')

# Print MAPEs
print(f'MAPE(y, y_pred): {mape_y}%')
print(f'MAPE(x_inv, x): {mape_x_inv}%')
print(f'MAPE(x_inv_real, x): {mape_x_inv_real}%')


### store results #######################################################################################

# Prepare data to store in JSON
results = {
    "x_test_original": X_test_original.tolist(), # Original x_test data
    "y_test_original": y_test_original.tolist(), # Original y_test data
    "y_pred": y_pred_original.tolist(),
    "x_inv": x_inv_original.tolist(),
    "x_inv_real": x_inv_real_original.tolist(),
    "MSE": {
        "MSE_y_y_pred": mse_y,
        "MSE_x_inv_x": mse_x_inv,
        "MSE_x_inv_real_x": mse_x_inv_real,
        "MAPE_y_y_pred": mape_y,
        "MAPE_x_inv_x": mape_x_inv,
        "MAPE_x_inv_real_x": mape_x_inv_real

    }
}
# Save the results to a JSON file
json_file_path = f'./results/{model_name}_results.json'
with open(json_file_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {json_file_path}")



visualize(json_file_path, model_name)

