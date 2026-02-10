import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from utils import *

torch.manual_seed(0)


# Define a small neural network for the residual function
class SmallNetwork(nn.Module):
    def __init__(self, input_dim, use_power_iteration=False):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim , input_dim)
        self.fc3 = nn.Linear(input_dim , input_dim)

        # self.activation = nn.ReLU()
        # self.activation = nn.Sigmoid()
        self.activation = nn.LeakyReLU(0.1)

        if use_power_iteration:
            # Apply spectral normalization using power iteration
            self.apply_spectral_normalization()

    # Lipschitz continuty of NN: compute the spectral norm of weight matrix for each layer
    # def apply_spectral_normalization(self):
    #     # Apply spectral normalization to each linear layer
    #     nn.utils.parametrizations.spectral_norm(self.fc1)
    #     nn.utils.parametrizations.spectral_norm(self.fc2)
    #     nn.utils.parametrizations.spectral_norm(self.fc3)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the invertible residual block
# with options for Lipschitz enforcement

class InvertibleResidualBlock(nn.Module):
    def __init__(self, input_dim, enforce_lipz='orthogonal', alpha=1):
        # defaulted as using orthogonal, change with enforce_lipz = 'power_iteration'
        super(InvertibleResidualBlock, self).__init__()
        self.alpha = alpha  # Scaling coefficient
        use_power_iteration = enforce_lipz == 'power_iteration'
        # print(use_power_iteration)
        self.residual_function = SmallNetwork(input_dim, use_power_iteration=use_power_iteration)

        # if enforce_lipz == 'orthogonal':
        #     # Ensure Lipschitz continuity using orthogonal initialization with small gain
        #     for layer in self.residual_function.modules():
        #         if isinstance(layer, nn.Linear):
        #             torch.nn.init.orthogonal_(layer.weight, gain=1e-2)

    def forward(self, x):
        return x + self.alpha * self.residual_function(x)
    # alpha = 0.9 is a scaling coeff to maintain Lipschitz constraint, can be changed


    def inverse(self, y):
        # Use fixed-point iteration to find the inverse
        x = y
        for _ in range(100):
            x = y - self.residual_function(x)
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
                # print(f"Lipz Constant for layer {layer}: {max_singular_value:.4f}")
                # print(f"{max_singular_value:.4f}")
                # for each layer, quantify Lipz values
        # Check if all values are < 1
        all_satisfied = all(lipz < 1 for lipz in lipz_values)
        print(f"Overall Lipschitz Continuity Satisfied: {all_satisfied}")
        # For entire model, yes or no
        return all_satisfied, lipz_values



# Define the i-ResNet
class iResNet(nn.Module):
    def __init__(self, input_dim, num_blocks, enforce_lipz='orthogonal'):
        super(iResNet, self).__init__()
        self.blocks = nn.ModuleList(
            [InvertibleResidualBlock(input_dim, enforce_lipz=enforce_lipz) for _ in range(num_blocks)])

    def forward(self, x):
        log_det_jacobian = torch.zeros(x.size(0))
        for block in self.blocks:
            x = block(x)
            log_det_jacobian += block.log_det_jacobian(x)
        return x, log_det_jacobian

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
            print(f"Checking Lipschitz continuity for block {i + 1}:")
            satisfied, lipz_values = block.check_lipz_continuity()
            all_satisfied = all_satisfied and satisfied
            print(f"Lipz values for block {i + 1}: {lipz_values}")
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
        # ###################################################

        for x, y in data_loader:
            optimizer.zero_grad()

            # Forward pass: compute model output
            y_pred, log_det_jacobian = model(x)

            loss = criterion(y, y_pred)  # MSE loss

            # Backward pass: compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')

        # Check Lipschitz continuity
        if (epoch + 1) % 100 == 0:
            model.check_lipz_continuity()

    print(iresnet_lips)
    return iresnet_lips

# Example usage:
if __name__ == "__main__":
    input_dim = 4 # x has 2 features
    hidden_dim = 10  # Not used
    # num_layers = 3 # for bad case
    batch_size = 64
    num_epochs = 1000
    learning_rate = 0.01
    num_blocks = 3
    # num_gen_samples = 1000  # Generate the same number of samples as the original data for comparison

    criterion = nn.MSELoss()

    # Generate time series data
    from utils import *

    # X, y, t = deterministic_data_high_dim_case3(dims=input_dim)
    # X, y, t = deterministic_data_case4()
    # X, y, t = deterministic_data_case1()
    # X, y, t = deterministic_data_case2()

    # X, y, t = deterministic_data_case3()
    # X, y, t = deterministic_data_case4()
    # X, y, t = deterministic_data_case4_1()
    # X, y, t = deterministic_data_case4_2()
    # X, y, t = deterministic_data_case4_3()

    # X, y, t = deterministic_data_case13()
    # X, y, t = deterministic_data_case_bad1()
    # X, y, t = deterministic_data_case_bad2()

    # Create DataLoader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model with either orthogonal initialization or power iteration
    model = iResNet(input_dim, num_blocks, enforce_lipz='None')  # Change 'power_iteration' to 'orthogonal' as needed
    print("ResNet Number of Parameters:", model.count_parameters())
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, data_loader, optimizer, num_epochs=num_epochs)

    # Evaluation
    model.eval()
    y_pred, log_det_jacobian = model(X)

    #section predict forward
    loss_pred = criterion(y, y_pred)
    print("ResNet｜ predict(y) MSE", loss_pred.item())
    mse_per_column = calculate_mse_per_column(y.detach().numpy(), y_pred.detach().numpy())
    print("ResNet｜ predict(y) MSE per col", mse_per_column)

    #section inverse part: from x hidden -> x_reconstruct
    x_reconstruct = model.inverse(y_pred)
    loss_reconstruct = criterion(X, x_reconstruct)
    print("ResNet｜ reconstruct ｜ MSE:", loss_reconstruct.item())
    mse_per_column = calculate_mse_per_column(X.detach().numpy(), x_reconstruct.detach().numpy())
    print("ResNet｜ reconstruct ｜ MSE per col", mse_per_column)


    #section inverse part: from y -> x_reconstruct
    x_reconstruct_from_y = model.inverse(y)
    loss_reconstruct = criterion(X, x_reconstruct_from_y)
    print("ResNet｜ reconstruct from y ｜ MSE:", loss_reconstruct.item())
    mse_per_column = calculate_mse_per_column(X.detach().numpy(), x_reconstruct_from_y.detach().numpy())
    print("ResNet｜ reconstruct from y ｜ MSE per col", mse_per_column)

    ##########################################################################
    import json
    model_name = "ResNet"
    data_name = "case2"
    results_dict = {}
    results_dict['t'] = t.tolist()
    results_dict['y'] = y.detach().numpy().tolist()
    results_dict['X'] = X.detach().numpy().tolist()
    results_dict['y_pred'] = y_pred.detach().numpy().tolist()
    results_dict['x_inv'] = x_reconstruct.detach().numpy().tolist()
    results_dict['x_inv_real'] = x_reconstruct_from_y.detach().numpy().tolist()

    json_file_path = f'./results/{data_name}_{model_name}_results.json'
    print(json_file_path)
    with open(json_file_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    print(f"Results saved to {json_file_path}")
    ##########################################################################

    # Plot results
    # Create a figure with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    # Plot on each subplot
    for i in range(input_dim):
        axs[0].scatter(t, y.detach().numpy()[:, i], alpha=0.8, c="black", label='True y {}'.format(i))
        # axs[0].plot(t, y.detach().numpy()[:, i], alpha=0.8, label='True y {}'.format(i))
        axs[0].scatter(t, y_pred.detach().numpy()[:, i], alpha=0.8, s=10, label='Predict y {}'.format(i))

    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y')
    axs[0].set_title('ResNet: Followed Pass')
    axs[0].legend()

    for i in range(input_dim):
        axs[1].scatter(t, X.detach().numpy()[:, i], alpha=1, label='True X {}'.format(i), c="black")
        axs[1].scatter(t, x_reconstruct.detach().numpy()[:, i], alpha=0.8, s=10, label='Reconstruct X {}'.format(i))
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('X value')
    axs[1].set_title('ResNet: Reconstructed X vs X')
    axs[1].legend()

    for i in range(input_dim):
        axs[2].scatter(t, X.detach().numpy()[:, i], alpha=1, label='True X {}'.format(i), c="black")
        axs[2].scatter(t, x_reconstruct_from_y.detach().numpy()[:, i], alpha=0.8, s=10, label='Reconstruct X from y{}'.format(i))
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('X value')
    axs[2].set_title('iResNet: Reconstructed X (from y) vs X')
    axs[2].legend()
    plt.show()

