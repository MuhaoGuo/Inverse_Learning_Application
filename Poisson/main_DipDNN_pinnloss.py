import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import tqdm


# Add this helper function before you call train_model:

def compute_pde_loss(u_pred, f_vals, Nx=512, pde_weight=1e-3):
    # u_pred and f_vals shape: (batch_size, Nx+1)
    # Approximate second derivative w.r.t. x using central differences
    # For simplicity, assume dx = 1/Nx:
    dx = 1.0 / Nx

    # Pad u_pred for boundary differences:
    # u shape: (B, Nx+1)
    u_left = u_pred[:, :-2]    # u(x_{i-1})
    u_mid = u_pred[:, 1:-1]    # u(x_i)
    u_right = u_pred[:, 2:]    # u(x_{i+1})

    # Compute second derivative in x:
    # (u(x+1) - 2u(x) + u(x-1)) / dx^2
    u_xx = (u_right - 2*u_mid + u_left) / (dx**2)

    # Also trim f_vals to match dimensions:
    f_mid = f_vals[:, 1:-1]

    # PDE residual: u_xx + f = 0  -> want to minimize (u_xx + f)^2
    residual = u_xx + f_mid
    pde_loss = torch.mean(residual**2) * pde_weight
    return pde_loss

# Now modify the train_model function by adding PDE loss:
def train_model(model, train_loader, optimizer, criterion, epochs, device=torch.device('cpu')):
    model = model.to(device)
    model.train()
    Nx = 512  # we know Nx=512 from the data
    pde_weight = 1e-3  # You can adjust this weight as needed

    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
                inputs = inputs.to(device)   # f
                targets = targets.to(device) # u
                optimizer.zero_grad()
                outputs = model(inputs)

                # Original supervised loss between outputs and targets:
                loss = criterion(outputs, targets)

                # Compute PDE loss from outputs (u) and inputs (f):
                # Since PDE: -u_xx = f -> u_xx + f = 0
                # Note: outputs ~ u, inputs ~ f
                pde_loss = compute_pde_loss(outputs, inputs, Nx=Nx, pde_weight=pde_weight)

                # Add PDE loss to total loss:
                total_loss = loss + pde_loss

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))

# All other logic (data loading, model definition, visualization) remains unchanged.
# Just run the code as before.


from utils import *


class SmallNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

        self.register_buffer('mask_tril', torch.tril(torch.ones_like(self.fc1.weight)))
        self.register_buffer('mask_triu', torch.triu(torch.ones_like(self.fc2.weight)))
        self.fc1.weight = nn.Parameter(torch.mul(self.fc1.weight, self.mask_tril))
        self.fc2.weight = nn.Parameter(torch.mul(self.fc2.weight, self.mask_triu))

        # Apply masks and ensure diagonal elements are ones
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

        # Register hooks to enforce masking after each update
        self.fc1.weight.register_hook(lambda grad: grad * self.mask_tril)
        self.fc2.weight.register_hook(lambda grad: grad * self.mask_triu)

        self.negative_slope = 0.5

    def inverse_leaky_relu(self, z, negative_slope):
        return torch.where(z >= 0, z, z / negative_slope)

    def forward(self, x):
        fc1_fwd = self.fc1(x)
        fc1_fwd = F.leaky_relu(fc1_fwd, negative_slope=self.negative_slope)
        fc2_fwd = self.fc2(fc1_fwd)
        fc2_fwd = F.leaky_relu(fc2_fwd, negative_slope=self.negative_slope)
        return fc2_fwd

    def inverse(self, y):
        batch_size = y.shape[0]
        # Inverse of the last activation
        # y1 = F.leaky_relu(y, negative_slope=1 / self.negative_slope)
        y1 = self.inverse_leaky_relu(y, self.negative_slope)

        # Subtract bias
        y1 = y1 - self.fc2.bias
        # Solve fc2.weight * x = y1 using triangular solve
        y1_unsqueezed = y1.unsqueeze(2)  # shape (batch_size, input_dim, 1)
        fc2_weight_expanded = self.fc2.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # fc2_inv, _ = torch.triangular_solve(y1_unsqueezed, fc2_weight_expanded, upper=True)
        fc2_inv = torch.linalg.solve_triangular(fc2_weight_expanded, y1_unsqueezed, upper=True)

        fc2_inv = fc2_inv.squeeze(2)

        # Inverse of the first activation
        # y2 = F.leaky_relu(fc2_inv, negative_slope=1 / self.negative_slope)
        y2 = self.inverse_leaky_relu(fc2_inv, negative_slope=self.negative_slope)

        # Subtract bias
        y2 = y2 - self.fc1.bias
        # Solve fc1.weight * x = y2 using triangular solve
        y2_unsqueezed = y2.unsqueeze(2)
        fc1_weight_expanded = self.fc1.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # fc1_inv, _ = torch.triangular_solve(y2_unsqueezed, fc1_weight_expanded, upper=False)
        fc1_inv = torch.linalg.solve_triangular(fc1_weight_expanded, y2_unsqueezed, upper=False)

        fc1_inv = fc1_inv.squeeze(2)
        return fc1_inv

    def apply_weight_masks(self):
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

class DipDNN_Block(nn.Module):
    def __init__(self, input_dim, alpha=0.9):
        super(DipDNN_Block, self).__init__()
        # self.alpha = alpha  # Scaling coefficient
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
        self.input_dim = input_dim

    def forward(self, x):

        # x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for block in self.blocks:
            x = block(x)

        # x = x.reshape(x.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        return x

    def inverse(self, y):
        # y = y.reshape(y.shape[0], -1)  # (batchsize, -1)

        for block in reversed(self.blocks):
            y = block.inverse(y)

        # y = y.reshape(y.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        return y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_weight_masks(self):
        for block in self.blocks:
            block.apply_weight_masks()


device = get_device()
print("device", device)
# Set the seed before initializing the model or any random operations
set_seed(42)  # You can use any seed value of your choice
train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_poisson_data_sampling(device)


input_dim = 513
num_blocks = 3
learning_rate = 1e-4
epochs = 1000

model_name = "DipDNNpinn3_4"
data_name = "Poisson"
model = DipDNN(input_dim=input_dim, num_blocks=num_blocks).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()

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


### Test model on Train #########################
train_pred = model(X_train_tensor)
inverse_of_fx_train = model.inverse(train_pred)
inverse_of_y_train = model.inverse(y_train_tensor)



fig, axis = plt.subplots(1, 8, figsize=(20, 2))
Nx = 512
xmin, xmax, ymin, ymax = 0, 1, 0, 1
im0 = axis[0].imshow(y_train_tensor.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "True y (u)")
axis[0].set_title("True y")
im1 = axis[1].imshow(train_pred.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "Pred y")
axis[1].set_title("Pred y")
im2 = axis[2].imshow(y_train_tensor.cpu().detach().numpy().reshape(Nx + 1, Nx + 1) - train_pred.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "Fwd error")
axis[2].set_title("Fwd error")
im3 = axis[3].imshow(X_train_tensor.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "True X (f)")
axis[3].set_title("True X")
im4 = axis[4].imshow(inverse_of_fx_train.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "fake inv X (from fx)")
axis[4].set_title("fake inv X")
im5 = axis[5].imshow(inverse_of_y_train.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "real inv X (from y)")
axis[5].set_title("real inv X")
im6 = axis[6].imshow(inverse_of_fx_train.cpu().detach().numpy().reshape(Nx + 1, Nx + 1) -X_train_tensor.cpu().detach().numpy().reshape(Nx + 1, Nx + 1) , extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "fake inv error")
axis[6].set_title("fake inv error")
im7 = axis[7].imshow(inverse_of_y_train.cpu().detach().numpy().reshape(Nx + 1, Nx + 1)- X_train_tensor.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "real inv error")
axis[7].set_title("real inv error")
plt.savefig(f"./results/{data_name}_{model_name}.png")


# Calculate MSE
fwd_error = mean_squared_error(y_train_tensor.cpu().detach().numpy(), train_pred.cpu().detach().numpy())
inv_fake_error = mean_squared_error(X_train_tensor.cpu().detach().numpy(), inverse_of_fx_train.cpu().detach().numpy())
inv_real_error = mean_squared_error(X_train_tensor.cpu().detach().numpy(), inverse_of_y_train.cpu().detach().numpy())
print("Training Data ..... ")
print(f"{model_name} MSE train fwd_error: {fwd_error}")
print(f"{model_name} MSE train inv_fake_error: {inv_fake_error}")
print(f"{model_name} MSE train inv_real_error: {inv_real_error}")
plt.show()
