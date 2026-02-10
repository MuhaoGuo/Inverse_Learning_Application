import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, process_first_part=True):
        super(AffineCouplingLayer, self).__init__()
        self.process_first_part = process_first_part
        self.network = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2)
        )
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
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
        y1, y2 = y.chunk(2, dim=1)
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


class NICE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(NICE, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            process_first_part = (i % 2 == 0)
            self.layers.append(AffineCouplingLayer(input_dim, hidden_dim, process_first_part))

        self.scaling_factor = nn.Parameter(torch.ones(1, input_dim))
        # self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x * self.scaling_factor

        # out = self.output_layer(x)
        return x

    def inverse(self, y):
        y = y / self.scaling_factor
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_nice(model, data_loader, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in data_loader:
            optimizer.zero_grad()

            # Forward pass: compute model output
            y_pred = model(x)

            loss = criterion(y, y_pred)  # MSE loss

            # Backward pass: compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')


# Example usage:
if __name__ == "__main__":
    input_dim = 4 # x has 2 features
    hidden_dim = 10 # hidden = 10
    batch_size = 64
    num_epochs = 1000
    learning_rate = 0.01
    num_layers = 2   # 2 * 3 = 6 layers
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    # Generate time series data
    from utils import *
    # X, y, t = deterministic_data_case1()
    # X, y, t = deterministic_data_case2()
    # X, y, t = deterministic_data_case3()
    # X, y, t = deterministic_data_case4()
    # X, y, t = deterministic_data_case4_iid()
    # X, y, t = deterministic_data_case4_1()
    # X, y, t = deterministic_data_case4_2()
    X, y, t = deterministic_data_case4_3()
    # X, y, t = deterministic_data_high_dim_case3(dims = input_dim)
    # X, y, t = deterministic_data_case7()
    # X, y, t = deterministic_data_case_bad1()
    # X, y, t = deterministic_data_case_bad2()

    # Create DataLoader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer
    model = NICE(input_dim, hidden_dim, num_layers)
    print("NICE Number of Parameters:", model.count_parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # section Train the NICE model
    train_nice(model, data_loader, optimizer, num_epochs=num_epochs)

    # section evaluation
    model.eval()
    y_pred = model(X)

    #section predict forward
    loss_pred = criterion(y, y_pred)
    print("NICE｜ predict(y) MSE", loss_pred.item())
    mse_per_column = calculate_mse_per_column(y.detach().numpy(), y_pred.detach().numpy())
    print("NICE｜ predict(y) MSE per col", mse_per_column)

    #section inverse part: from x hidden -> x_reconstruct
    x_reconstruct = model.inverse(y_pred)
    loss_reconstruct = criterion(X, x_reconstruct)
    print("NICE｜ reconstruct ｜ MSE:", loss_reconstruct.item())
    mse_per_column = calculate_mse_per_column(X.detach().numpy(), x_reconstruct.detach().numpy())
    print("NICE｜ reconstruct ｜ MSE per col", mse_per_column)

    #section inverse part: from y -> x_reconstruct
    x_reconstruct_from_y = model.inverse(y)
    loss_reconstruct = criterion(X, x_reconstruct_from_y)
    print("NICE｜ reconstruct from y ｜ MSE:", loss_reconstruct.item())
    mse_per_column = calculate_mse_per_column(X.detach().numpy(), x_reconstruct_from_y.detach().numpy())
    print("NICE｜ reconstruct from y ｜ MSE per col", mse_per_column)


    ##########################################################################
    import json
    model_name = "NICE"
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
        # axs[0].plot(t, y.detach().numpy()[:, i], alpha=0.8, label='True y {}'.format(i))
        axs[0].scatter(t, y.detach().numpy()[:, i], alpha=0.8, c="black", label='True y {}'.format(i))
        axs[0].scatter(t, y_pred.detach().numpy()[:, i], alpha=0.8, s=10, label='Predict y {}'.format(i))
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y')
    axs[0].set_title('NICE: Followed Pass')
    axs[0].legend()

    for i in range(input_dim):
        axs[1].scatter(t, X.detach().numpy()[:, i], alpha=1, label='True X {}'.format(i), c="black")
        axs[1].scatter(t, x_reconstruct.detach().numpy()[:, i], alpha=0.8, s=10, label='Reconstruct X {}'.format(i))
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('X value')
    axs[1].set_title('NICE: Reconstructed X vs X')
    axs[1].legend()

    for i in range(input_dim):
        axs[2].scatter(t, X.detach().numpy()[:, i], alpha=1, label='True X {}'.format(i), c="black")
        axs[2].scatter(t, x_reconstruct_from_y.detach().numpy()[:, i], alpha=0.8, s=10, label='Reconstruct X from y{}'.format(i))
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('X value')
    axs[2].set_title('NICE: Reconstructed X (from y) vs X')
    axs[2].legend()
    plt.show()
