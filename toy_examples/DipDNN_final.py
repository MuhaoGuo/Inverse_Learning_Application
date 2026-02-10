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
import torch.nn.utils as nn_utils
import random

def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility in convolutional layers
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility


torch.manual_seed(10) # 0 or 10

#
class SmallNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

        self.register_buffer('mask_tril', torch.tril(torch.ones_like(self.fc1.weight)))
        self.register_buffer('mask_triu', torch.triu(torch.ones_like(self.fc2.weight)))
        self.fc1.weight = nn.Parameter(torch.mul(self.fc1.weight, self.mask_tril))
        self.fc2.weight = nn.Parameter(torch.mul(self.fc2.weight, self.mask_triu))
        self.negative_slope = 0.8 # 0.8

        # # Apply masks and ensure diagonal elements are ones
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

        # Register hooks to enforce masking after each update
        self.fc1.weight.register_hook(lambda grad: grad * self.mask_tril)
        self.fc2.weight.register_hook(lambda grad: grad * self.mask_triu)


    def forward(self, x):
        fc1_fwd = self.fc1(x)
        fc1_fwd = F.leaky_relu(fc1_fwd, negative_slope = self.negative_slope)
        fc2_fwd = self.fc2(fc1_fwd)
        fc2_fwd = F.leaky_relu(fc2_fwd, negative_slope = self.negative_slope)
        return fc2_fwd

    def apply_weight_masks(self):
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

    def inverse(self, y):
        batch_size = y.shape[0]
        # Inverse of the last activation
        y1 = F.leaky_relu(y, negative_slope=1 / self.negative_slope)
        # Subtract bias
        y1 = y1 - self.fc2.bias
        # Solve fc2.weight * x = y1 using triangular solve
        y1_unsqueezed = y1.unsqueeze(2)  # shape (batch_size, input_dim, 1)
        fc2_weight_expanded = self.fc2.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # fc2_inv, _ = torch.triangular_solve(y1_unsqueezed, fc2_weight_expanded, upper=True)
        fc2_inv = torch.linalg.solve_triangular(fc2_weight_expanded, y1_unsqueezed, upper=True)

        fc2_inv = fc2_inv.squeeze(2)

        # Inverse of the first activation
        y2 = F.leaky_relu(fc2_inv, negative_slope=1 / self.negative_slope)
        # Subtract bias
        y2 = y2 - self.fc1.bias
        # Solve fc1.weight * x = y2 using triangular solve
        y2_unsqueezed = y2.unsqueeze(2)
        fc1_weight_expanded = self.fc1.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # fc1_inv, _ = torch.triangular_solve(y2_unsqueezed, fc1_weight_expanded, upper=False)
        fc1_inv = torch.linalg.solve_triangular(fc1_weight_expanded, y2_unsqueezed, upper=False)

        fc1_inv = fc1_inv.squeeze(2)
        return fc1_inv



class DipDNN_Block(nn.Module):
    def __init__(self, input_dim, alpha=0.9):
        super(DipDNN_Block, self).__init__()
        self.alpha = alpha  # Scaling coefficient
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

    def forward(self, x):
        forward_activations = []
        forward_activations.append(x.clone())

        # x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for block in self.blocks:
            x = block(x)
            forward_activations.append(x.clone())

            # x = x.reshape(x.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))
        return x, forward_activations

    # def inverse(self, y):
    #     # y = y.reshape(y.shape[0], -1)  # (batchsize, -1)
    #
    #     for block in reversed(self.blocks):
    #         y = block.inverse(y)
    #
    #     # y = y.reshape(y.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))
    #     return y


    def inverse(self, y):
        inverse_activations = []
        inverse_activations.append(y)
        for block in reversed(self.blocks):
            y = block.inverse(y)
            inverse_activations.append(y.clone())  # Store inverse pass output
        # inverse_activations = list(reversed(inverse_activations))  # Align with forward order
        inverse_activations = list(inverse_activations)
        return y, inverse_activations

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_weight_masks(self):
        for block in self.blocks:
            block.apply_weight_masks()

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE)

    Parameters:
    y_true (numpy.ndarray): Ground truth (actual) values.
    y_pred (numpy.ndarray): Predicted values.

    Returns:
    float: MAPE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero by adding a small epsilon where y_true is zero
    epsilon = np.finfo(np.float64).eps
    denominator = np.where(y_true == 0, epsilon, y_true)

    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
    return mape

def l2_regularization(model, lambda_l2):
    l2_norm = sum(torch.norm(param, 2)**2 for param in model.parameters() if param.requires_grad)
    return lambda_l2 * l2_norm


def train_model(model, data_loader, optimizer, num_epochs=100):
    model.train()
    iresnet_lips = []
    reg_count = 0
    apply_weight_masks = False

    mask_tril = torch.tril(torch.ones_like(model.blocks[0].residual_function.fc1.weight))
    mask_triu = torch.triu(torch.ones_like(model.blocks[0].residual_function.fc2.weight))

    for epoch in range(num_epochs):
        total_loss = 0

        # ###################################################
        # estimated_lipschitz = estimate_lipschitz(model, data_loader, num_samples=100)
        # print("Epoch: {} | estimated_lipschitz: {}".format(epoch, estimated_lipschitz))
        # iresnet_lips.append(estimated_lipschitz)
        ###################################################

        for x, y in data_loader:
            optimizer.zero_grad()

            # print(model.blocks[0].residual_function.fc1.weight)
            # Forward pass: compute model output
            y_pred, log_det_jacobian = model(x)

            loss = criterion(y, y_pred)  # MSE loss

            # l2 = l2_regularization(model, 0.00000)
            # loss += l2

            # Backward pass: compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            total_loss += loss.item()

            # print(model.blocks[0].residual_function.fc1.weight)
            if apply_weight_masks:
                model.apply_weight_masks()

            with torch.no_grad():
                # Regularize weights to prevent near-singular matrices
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        diag_elements = torch.diag(param)
                        threshold = 1e-2  # 1e-2
                        mask = torch.abs(diag_elements) < threshold
                        non_zero_value = 1e-1 # 1e-1
                        if torch.any(mask):
                            param[torch.eye(param.size(0)).bool()] = torch.where(
                                mask, torch.tensor(non_zero_value, device=param.device), diag_elements
                            )
                            reg_count += 1

                # Re-apply masks to ensure triangular structure
                for block in model.blocks:
                    block.residual_function.fc1.weight.mul_(mask_tril)
                    block.residual_function.fc2.weight.mul_(mask_triu)

            # print(model.blocks[0].residual_function.fc1.weight)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')

    print(iresnet_lips)

    ################
    # Evaluation on Training and Test Sets
    X_tensor = x
    Y_tensor = y
    with torch.no_grad():
        # Forward pass
        Y_est, forward_train_activations = model.forward(X_tensor)
        # test_pred, forward_test_activations = model.forward(X_test_tensor)

        # Inverse pass
        X_est, inverse_train_activations = model.inverse(Y_tensor)
        X_reconstr, inverse_reconstr_activations = model.inverse(Y_est)

        # Compute losses
        train_loss = F.mse_loss(Y_est, Y_tensor).item()
        # test_loss = F.mse_loss(test_pred, Y_test_tensor).item()
        Train_MAPE = mean_absolute_percentage_error(Y_tensor.numpy(), Y_est.numpy())
        # Test_MAPE = mean_absolute_percentage_error(Y_test, test_pred.numpy())

        # Reconstruction errors
        yx_err = F.mse_loss(X_est, X_tensor).item()
        xyx_err = F.mse_loss(X_reconstr, X_tensor).item()

        print("Model Parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")
        print(f"# of weights regularization: {reg_count}")
        # print(f"Forward Pass MSE - Train: {train_loss}, Test: {test_loss}")
        # print(f"Forward Pass MAPE - Train: {Train_MAPE:.4f}%, Test: {Test_MAPE:.4f}%")

        print("\nInverse Reconstruction Errors:")
        print(f"One-direction learning (YX Error): {yx_err}")
        print(f"Reconstruction (XYX Error): {xyx_err}")

        # Detailed per-block inverse performance
        print("\nOne-to-one Performance by Block:")

        forward_train_activations_np = np.array([activation.detach().cpu().numpy() for activation in forward_train_activations ])
        inverse_reconstr_activations_np = np.array([activation.detach().cpu().numpy() for activation in inverse_reconstr_activations ])

        print(forward_train_activations_np[:, 0])
        print(inverse_reconstr_activations_np[:, 0])

        for layer in range(len(model.blocks)):

            fwd_input = forward_train_activations[-1 - (layer + 1)]  # forward layer inputs
            inv_reconstr = inverse_reconstr_activations[layer + 1]

            # Compute reconstruction error at this block
            zyz_err = F.mse_loss(inv_reconstr, fwd_input).item()

            # Verify forward outputs consistency
            fwd_output_forward = forward_train_activations[-1-(layer)]
            fwd_output_recomputed = model.blocks[-1-(layer)](fwd_input)


            inv_output = model.blocks[-1-(layer)].inverse(fwd_output_recomputed)
            # zz_est_err = F.mse_loss(inv_output, fwd_input).item()

            zz_est_err = np.mean((inv_output.detach().numpy() - fwd_input.detach().numpy()) ** 2)  # reconstruction of z in the middle layers
            print("Block " + str(layer) + " Propagation", zyz_err, "Ablation", zz_est_err)

    return iresnet_lips




# Example usage:
if __name__ == "__main__":
    input_dim = 4 # x has 2 features
    batch_size = 64
    num_epochs = 1000
    learning_rate = 0.01
    num_blocks = 8 # 6

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
    X, y, t = deterministic_data_case4_1()
    # X, y, t = deterministic_data_case4_2()
    # X, y, t = deterministic_data_case4_3()
    # X, y, t = deterministic_data_case13()
    # X, y, t = deterministic_data_case_bad1()
    # X, y, t = deterministic_data_case_bad2()


    # Create DataLoader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model with either orthogonal initialization or power iteration
    model = DipDNN(input_dim, num_blocks=num_blocks)  # Change 'power_iteration' to 'orthogonal' as needed

    print(model)

    print("DipDNN Number of Parameters:", model.count_parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, )
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-10, amsgrad=False)

    # Train the model
    train_model(model, data_loader, optimizer, num_epochs=num_epochs)


    # Evaluatio
    model.eval()
    y_pred, _ = model(X)

    #section predict forward
    loss_pred = criterion(y, y_pred)
    print("DipDNN｜ predict(y) MSE", loss_pred.item())
    mse_per_column = calculate_mse_per_column(y.detach().numpy(), y_pred.detach().numpy())
    print("DipDNN｜ predict(y) MSE per col", mse_per_column)

    #section inverse part: from x hidden -> x_reconstruct
    x_reconstruct, _ = model.inverse(y_pred)
    loss_reconstruct = criterion(X, x_reconstruct)
    print("DipDNN｜ reconstruct ｜ MSE:", loss_reconstruct.item())
    mse_per_column = calculate_mse_per_column(X.detach().numpy(), x_reconstruct.detach().numpy())
    print("DipDNN｜ reconstruct ｜ MSE per col", mse_per_column)

    #section inverse part: from y -> x_reconstruct
    x_reconstruct_from_y, _ = model.inverse(y)
    loss_reconstruct = criterion(X, x_reconstruct_from_y)
    print("DipDNN｜ reconstruct from y ｜ MSE:", loss_reconstruct.item())
    mse_per_column = calculate_mse_per_column(X.detach().numpy(), x_reconstruct_from_y.detach().numpy())
    print("DipDNN｜ reconstruct from y ｜ MSE per col", mse_per_column)

    ##########################################################################
    import json
    model_name = "DipDNN"
    data_name = "case2"
    results_dict = {}
    results_dict['t'] = t.tolist()
    results_dict['y'] = y.detach().numpy().tolist()
    results_dict['X'] = X.detach().numpy().tolist()
    results_dict['y_pred'] = y_pred.detach().numpy().tolist()
    results_dict['x_inv'] = x_reconstruct.detach().numpy().tolist()
    results_dict['x_inv_real'] = x_reconstruct_from_y.detach().numpy().tolist()

    json_file_path = f'./results/{data_name}_{model_name}_results.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    print(f"Results saved to {json_file_path}")
    ##########################################################################

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(input_dim):
        axs[0].scatter(t, y.detach().numpy()[:, i], alpha=0.8, c="black", label='True y {}'.format(i))
        # axs[0].plot(t, y.detach().numpy()[:, i], alpha=0.8, label='True y {}'.format(i))
        axs[0].scatter(t, y_pred.detach().numpy()[:, i], alpha=0.8, s=10, label='Predict y {}'.format(i))

    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y')
    axs[0].set_title('DipDNN: Followed Pass')
    axs[0].legend()

    for i in range(input_dim):
        axs[1].scatter(t, X.detach().numpy()[:, i], alpha=1, label='True X {}'.format(i), c="black")
        axs[1].scatter(t, x_reconstruct.detach().numpy()[:, i], alpha=0.8, s=10, label='Reconstruct X {}'.format(i))
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('X value')
    axs[1].set_title('DipDNN: Reconstructed X vs X')
    axs[1].legend()

    for i in range(input_dim):
        axs[2].scatter(t, X.detach().numpy()[:, i], alpha=1, label='True X {}'.format(i), c="black")
        axs[2].scatter(t, x_reconstruct_from_y.detach().numpy()[:, i], alpha=0.8, s=10, label='Reconstruct X from y{}'.format(i))
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('X value')
    axs[2].set_title('DipDNN: Reconstructed X (from y) vs X')
    axs[2].legend()
    plt.show()