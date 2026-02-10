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


torch.manual_seed(0)


class Inverse_Net_maskBlock(nn.Module):
    def __init__(self, in_features):
        super(Inverse_Net_maskBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.fc2 = nn.Linear(in_features, in_features, bias=True)

        self.register_buffer('mask_tril', torch.tril(torch.ones_like(self.fc1.weight)))
        self.register_buffer('mask_triu', torch.triu(torch.ones_like(self.fc2.weight)))
        self.fc1.weight = nn.Parameter(torch.mul(self.fc1.weight, self.mask_tril))
        self.fc2.weight = nn.Parameter(torch.mul(self.fc2.weight, self.mask_triu))

        # self.activation = nn.Sigmoid()

    def forward(self, x):
        fc1_fwd = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        fc2_fwd = F.leaky_relu(self.fc2(fc1_fwd), negative_slope=0.1)
        return fc2_fwd

    def inverse(self, y):
        fc2_W_T = torch.linalg.inv(torch.mul(self.fc2.weight, self.mask_triu))
        fc2_inv = F.linear(F.leaky_relu(y, negative_slope=1/0.1) - self.fc2.bias, fc2_W_T)
        fc1_W_T = torch.linalg.inv(torch.mul(self.fc1.weight, self.mask_tril))
        fc1_inv = F.linear(F.leaky_relu(fc2_inv, negative_slope=1/0.1) - self.fc1.bias, fc1_W_T)
        return fc1_inv

# Define the Invertible Neural Network for feature representation
class Inverse_DNN_with_Features(nn.Module):
    def __init__(self, input_dim, num_blocks):
        super(Inverse_DNN_with_Features, self).__init__()
        self.layers = nn.ModuleList([Inverse_Net_maskBlock(input_dim) for _ in range(num_blocks)])
        # self.feature_fc = nn.Linear(input_dim, input_dim)
        # self.classifier = nn.Linear(feature_dim, num_classes)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x = x.reshape(x.shape[0], -1)   # (batchsize, -1)

        for layer in self.layers:
            x = layer(x)
        # features = F.leaky_relu(self.feature_fc(x))
        # x = self.feature_fc(x)
        # features = self.dropout(features)
        # out = self.classifier(features)

        # x = x.reshape(x.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))

        return x, None

    def inverse(self, x):
        # x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for layer in reversed(self.layers):
            x = layer.inverse(x)

        # x = x.reshape(x.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_model(model, data_loader, optimizer, num_epochs=100):
    model.train()
    iresnet_lips = []
    for epoch in range(num_epochs):
        total_loss = 0

        # ###################################################
        estimated_lipschitz = estimate_lipschitz(model, data_loader, num_samples=100)
        print("Epoch: {} | estimated_lipschitz: {}".format(epoch, estimated_lipschitz))
        iresnet_lips.append(estimated_lipschitz)
        ###################################################

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

    print(iresnet_lips)
    return iresnet_lips




# Example usage:
if __name__ == "__main__":
    input_dim = 2 # x has 2 features
    hidden_dim = 10  # Not used
    num_layers = 1 # for bad case
    batch_size = 64
    num_epochs = 400
    learning_rate = 0.01
    # num_gen_samples = 1000  # Generate the same number of samples as the original data for comparison
    num_blocks = 3

    criterion = nn.MSELoss()

    # Generate time series data
    from utils import * # deterministic_data_high_dim_case3

    # X, y, t = deterministic_data_high_dim_case3(dims=input_dim)
    # X, y, t = deterministic_data_case4()
    # X, y, t = deterministic_data_case1()
    # X, y, t = deterministic_data_case2()
    # X, y, t = deterministic_data_case3()
    # X, y, t = deterministic_data_case4()
    # X, y, t = deterministic_data_case13()
    # X, y, t = deterministic_data_case_bad1()
    X, y, t = deterministic_data_case_bad2()


    # Create DataLoader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model with either orthogonal initialization or power iteration
    model = Inverse_DNN_with_Features(input_dim, num_blocks=num_blocks)  # Change 'power_iteration' to 'orthogonal' as needed

    print(model)

    print("DipDNN Number of Parameters:", model.count_parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, data_loader, optimizer, num_epochs=num_epochs)

    # Evaluation
    model.eval()
    y_pred, log_det_jacobian = model(X)

    #section predict forward
    loss_pred = criterion(y, y_pred)
    print("DipDNN｜ predict(y) MSE", loss_pred.item())
    mse_per_column = calculate_mse_per_column(y.detach().numpy(), y_pred.detach().numpy())
    print("DipDNN｜ predict(y) MSE per col", mse_per_column)

    #section inverse part: from x hidden -> x_reconstruct
    x_reconstruct = model.inverse(y_pred)
    loss_reconstruct = criterion(X, x_reconstruct)
    print("DipDNN｜ reconstruct ｜ MSE:", loss_reconstruct.item())
    mse_per_column = calculate_mse_per_column(X.detach().numpy(), x_reconstruct.detach().numpy())
    print("DipDNN｜ reconstruct ｜ MSE per col", mse_per_column)

    #section inverse part: from y -> x_reconstruct
    x_reconstruct_from_y = model.inverse(y)
    loss_reconstruct = criterion(X, x_reconstruct_from_y)
    print("DipDNN｜ reconstruct from y ｜ MSE:", loss_reconstruct.item())
    mse_per_column = calculate_mse_per_column(X.detach().numpy(), x_reconstruct_from_y.detach().numpy())
    print("DipDNN｜ reconstruct from y ｜ MSE per col", mse_per_column)


    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(input_dim):
        axs[0].scatter(t, y_pred.detach().numpy()[:, i], alpha=0.8, s=10, label='Predict y {}'.format(i))
        axs[0].plot(t, y.detach().numpy()[:, i], alpha=0.8, label='True y {}'.format(i))
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