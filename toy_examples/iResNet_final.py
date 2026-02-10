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


# torch.manual_seed(0)

class _SpectralNorm(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        item: float = 0.2# todo
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
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.fc2 = nn.Linear(input_dim * 2, input_dim * 2)
        self.fc3 = nn.Linear(input_dim * 2, input_dim)

        # self.activation = nn.ReLU()
        # self.activation = nn.Sigmoid()
        self.activation = nn.LeakyReLU(0.1)

        if use_power_iteration:
            self.apply_spectral_normalization()

    def apply_spectral_normalization(self):
        # Apply spectral normalization to each linear layer
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.utils.parametrizations.spectral_norm(layer)
            self.scale_weights_below_one(layer)

    def scale_weights_below_one(self, layer):
        checking = False
        if checking:
            with torch.no_grad():
                u, s, v = torch.svd(layer.weight)
                max_singular_value = s.max().item()
                if max_singular_value >= 0.99:  #0.99
                    spectral_norm_customize(layer, item=0.2)  #0.2
        else:
            spectral_norm_customize(layer, item=0.2)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the invertible residual block
# with options for Lipschitz enforcement
class InvertibleResidualBlock(nn.Module):
    def __init__(self, input_dim, enforce_lipz='orthogonal', alpha=0.7):
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

    def log_det_jacobian(self, x):
        # Compute the Jacobian
        jacobian = self.compute_jacobian(x)
        # Compute the singular values
        singular_values = torch.linalg.svdvals(jacobian)
        # Compute the log determinant of the Jacobian
        log_det_jacobian = torch.sum(torch.log(singular_values), dim=-1)
        return log_det_jacobian

    def compute_jacobian(self, x):
        # Use PyTorch's autograd to compute the Jacobian
        x = x.requires_grad_(True)
        y = self.residual_function(x)
        jacobian = []
        for i in range(y.size(1)):
            jacobian_row = torch.autograd.grad(y[:, i], x, retain_graph=True, grad_outputs=torch.ones_like(y[:, i]))[0]
            jacobian.append(jacobian_row)
        jacobian = torch.stack(jacobian, dim=1)
        return jacobian

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
        # log_det_jacobian = torch.zeros(x.size(0))
        for block in self.blocks:
            x = block(x)
            # log_det_jacobian += block.log_det_jacobian(x)
        return x, None

    def inverse(self, y):
        for block in reversed(self.blocks):
            y = block.inverse(y)
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

def train_model(model, data_loader, optimizer, num_epochs=100):
    model.train()
    iresnet_lips = []
    for epoch in range(num_epochs):
        total_loss = 0

        # ###################################################
        # estimated_lipschitz = estimate_lipschitz(model, data_loader, num_samples=100)
        # print("Epoch: {} | estimated_lipschitz: {}".format(epoch, estimated_lipschitz))
        # iresnet_lips.append(estimated_lipschitz)
        ###################################################

        for x, y in data_loader:
            optimizer.zero_grad()

            # Forward pass: compute model output
            y_pred, _ = model(x)

            loss = criterion(y, y_pred)  # MSE loss

            # Backward pass: compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')

        # todo: make sure the Lipschitz continuity
        check_after_opt =False
        if check_after_opt:
            for block in model.blocks:
                block.residual_function.apply_spectral_normalization()

        # Check Lipschitz continuity
        # if (epoch + 1) % 100 == 0:
        #     model.check_lipz_continuity()
    print(iresnet_lips)
    return iresnet_lips



# Example usage:
if __name__ == "__main__":
    input_dim = 4 # x has 2 features
    hidden_dim = 10  # Not used
    num_blocks = 2 # for bad case  1
    batch_size = 64
    # num_epochs = 400
    num_epochs = 1000
    learning_rate = 0.01
    # num_gen_samples = 1000  # Generate the same number of samples as the original data for comparison

    criterion = nn.MSELoss()

    # Generate time series data
    from utils import * # deterministic_data_high_dim_case3

    # X, y, t = deterministic_data_high_dim_case3(dims=input_dim)
    # X, y, t = deterministic_data_case4()
    # X, y, t = deterministic_data_case1()
    # X, y, t = deterministic_data_case2()
    # X, y, t = deterministic_data_case3()
    # X, y, t = deterministic_data_case4()
    # X, y, t = deterministic_data_case4_iid()
    # X, y, t = deterministic_data_case4_1()
    X, y, t = deterministic_data_case4_2()
    # X, y, t = deterministic_data_case4_3()

    # X, y, t = deterministic_data_case13()
    # X, y, t = deterministic_data_case_bad1()
    # X, y, t = deterministic_data_case_bad2()



    # Create DataLoader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model with either orthogonal initialization or power iteration
    model = iResNet(input_dim, hidden_dim, num_blocks,
                    enforce_lipz='power_iteration')  # Change 'power_iteration' to 'orthogonal' as needed

    print(model)
    print("iResNet Number of Parameters:", model.count_parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, data_loader, optimizer, num_epochs=num_epochs)

    # Evaluation
    model.eval()
    y_pred, _ = model(X)

    #section predict forward
    loss_pred = criterion(y, y_pred)
    print("iResNet｜ predict(y) MSE", loss_pred.item())
    mse_per_column = calculate_mse_per_column(y.detach().numpy(), y_pred.detach().numpy())
    print("iResNet｜ predict(y) MSE per col", mse_per_column)

    #section inverse part: from x hidden -> x_reconstruct
    x_reconstruct = model.inverse(y_pred)
    loss_reconstruct = criterion(X, x_reconstruct)
    print("iResNet｜ reconstruct ｜ MSE:", loss_reconstruct.item())
    mse_per_column = calculate_mse_per_column(X.detach().numpy(), x_reconstruct.detach().numpy())
    print("iResNet｜ reconstruct ｜ MSE per col", mse_per_column)

    #section inverse part: from y -> x_reconstruct
    x_reconstruct_from_y = model.inverse(y)
    loss_reconstruct = criterion(X, x_reconstruct_from_y)
    print("iResNet｜ reconstruct from y ｜ MSE:", loss_reconstruct.item())
    mse_per_column = calculate_mse_per_column(X.detach().numpy(), x_reconstruct_from_y.detach().numpy())
    print("iResNet｜ reconstruct from y ｜ MSE per col", mse_per_column)


    ##########################################################################
    import json
    model_name = "iResNet"
    data_name = "case2"
    results_dict = {}
    results_dict['t'] = t.tolist()
    results_dict['y'] = y.detach().numpy().tolist()
    results_dict['X'] = X.detach().numpy().tolist()
    results_dict['y_pred'] = y_pred.detach().numpy().tolist()
    results_dict['x_inv'] = x_reconstruct.detach().numpy().tolist()
    results_dict['x_inv_real'] = x_reconstruct_from_y.detach().numpy().tolist()

    # print(results_dict)
    json_file_path = f'./{data_name}_{model_name}_results.json'
    print(json_file_path)
    with open(json_file_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    print(f"Results saved to {json_file_path}")
    #########################################################################

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(input_dim):
        axs[0].scatter(t, y.detach().numpy()[:, i], alpha=0.8, c="black", label='True y {}'.format(i))
        # axs[0].plot(t, y.detach().numpy()[:, i], alpha=0.8, label='True y {}'.format(i))
        axs[0].scatter(t, y_pred.detach().numpy()[:, i], alpha=0.8, s=10, label='Predict y {}'.format(i))
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y')
    axs[0].set_title('iResNet: Followed Pass')
    axs[0].legend()

    for i in range(input_dim):
        axs[1].scatter(t, X.detach().numpy()[:, i], alpha=1, label='True X {}'.format(i), c="black")
        axs[1].scatter(t, x_reconstruct.detach().numpy()[:, i], alpha=0.8, s=10, label='Reconstruct X {}'.format(i))
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('X value')
    axs[1].set_title('iResNet: Reconstructed X vs X')
    axs[1].legend()

    for i in range(input_dim):
        axs[2].scatter(t, X.detach().numpy()[:, i], alpha=1, label='True X {}'.format(i), c="black")
        axs[2].scatter(t, x_reconstruct_from_y.detach().numpy()[:, i], alpha=0.8, s=10, label='Reconstruct X from y{}'.format(i))
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('X value')
    axs[2].set_title('iResNet: Reconstructed X (from y) vs X')
    axs[2].legend()
    plt.show()