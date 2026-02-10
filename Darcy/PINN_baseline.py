import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# 1. Device selection (similar to Poisson snippet)
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# 2. Simple MLP (QRES style) for f and u
#    We'll name them "model_f" and "model_u" to parallel your Poisson code
# ------------------------------------------------------------------------------
class QRES(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(36, 36, 36)):
        super(QRES, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        # Final layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# Create two models: model_f and model_u
# They approximate the coefficient f(x) and the solution u(x).
model_f = QRES(input_dim=2, output_dim=1, hidden_dims=(30, 40, 40,20,20)).to(device)
model_u = QRES(input_dim=2, output_dim=1, hidden_dims=(50, 50, 20)).to(device)


# ------------------------------------------------------------------------------
# 3. Load data and create DataLoaders
#    We assume the file "darcy_data_f_u.npy" has shape ((Nx+1)^2, 2):
#       - data[:, 0] = true_f
#       - data[:, 1] = true_u
# ------------------------------------------------------------------------------
scaling = 100.0
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
Nx, Ny = 512, 512

# PDE domain points (xx)
xx_1d = np.linspace(xmin, xmax, Nx + 1).astype(np.float32)
xx_2d = np.array(np.meshgrid(xx_1d, xx_1d)).T.reshape(-1, 2)  # shape: ((Nx+1)*(Ny+1), 2)
xx_torch = torch.from_numpy(xx_2d).to(device)

# Load the Darcy data
#   data[:, 0]: f(x)
#   data[:, 1]: u(x)
loaded_data = np.load('darcy_data_f_u.npy').astype(np.float32)
loaded_data = scaling * loaded_data
true_f = loaded_data[:, 0:1]  # shape ((Nx+1)^2, 1)
true_u = loaded_data[:, 1:2]  # shape ((Nx+1)^2, 1)

data_f_torch = torch.from_numpy(true_f).to(device)
data_u_torch = torch.from_numpy(true_u).to(device)

# Build PyTorch Datasets and DataLoaders
pde_dataset = torch.utils.data.TensorDataset(xx_torch)  # PDE points
data_dataset = torch.utils.data.TensorDataset(xx_torch, data_u_torch)  # for solution data

pde_loader = torch.utils.data.DataLoader(pde_dataset, batch_size=50000, shuffle=True, drop_last=False)
data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=50000, shuffle=True, drop_last=False)


# ------------------------------------------------------------------------------
# 4. PDE residual: div(f(x) * grad(u(x))) + 1 = 0
# ------------------------------------------------------------------------------
def divergence_of_f_grad_u(x, model_f, model_u):
    """
    Compute div( f(x) * grad( u(x) ) ).
    """
    x = x.detach()
    x.requires_grad_(True)

    # forward pass
    f_val = model_f(x)  # shape (batch_size, 1)
    u_val = model_u(x)  # shape (batch_size, 1)

    # gradient of u wrt x
    grad_u = torch.autograd.grad(
        outputs=u_val,
        inputs=x,
        grad_outputs=torch.ones_like(u_val),
        create_graph=True
    )[0]  # shape (batch_size, 2)

    # f_val * grad_u => shape (batch_size, 2)
    f_grad_u = f_val * grad_u

    # divergence
    div_val = torch.zeros_like(u_val)
    for i in range(x.shape[1]):
        partial = torch.autograd.grad(
            outputs=f_grad_u[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(f_grad_u[:, i]),
            create_graph=True
        )[0][:, i].unsqueeze(-1)
        div_val = div_val + partial

    return div_val


def darcy_pde_residual(x, model_f, model_u):
    """
    PDE residual = div(f * grad(u)) + 1
    We want this to be zero => residual = div(...) + 1
    """
    div_term = divergence_of_f_grad_u(x, model_f, model_u)
    return div_term + 1.0


# ------------------------------------------------------------------------------
# 5. Training Loop
#    Combine PDE residual loss and data mismatch (u_pred vs u_data).
# ------------------------------------------------------------------------------
optimizer = optim.Adam(list(model_f.parameters()) + list(model_u.parameters()), lr=1e-3)

num_steps = 5000
pde_weight = 1.0
data_weight = 10000.0
model_name = "PINN_Darcy_baseline_dw10000_hiddenmatch_5000epoch"
t0 = time.time()

step_count = 0
for epoch in range(num_steps):
    pde_iter = iter(pde_loader)
    data_iter = iter(data_loader)

    while True:
        try:
            pde_batch = next(pde_iter)[0]            # (batch_size, 2)
            data_batch_x, data_batch_u = next(data_iter)  # (batch_size, 2), (batch_size, 1)
        except StopIteration:
            break

        # PDE loss
        residual = darcy_pde_residual(pde_batch, model_f, model_u)  # (batch_size, 1)
        pde_loss = torch.mean(residual**2)

        # Data loss: match predicted u to data
        u_pred = model_u(data_batch_x)
        data_loss = torch.mean((u_pred - data_batch_u)**2)

        loss = pde_weight * pde_loss + data_weight * data_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_count += 1

    # Print progress every N epochs
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{num_steps} | PDE loss={pde_loss.item():.4e} | Data loss={data_loss.item():.4e}")

t1 = time.time()
print(f"Training finished after {num_steps} epochs, in {t1 - t0:.2f} seconds.")


# ------------------------------------------------------------------------------
# 6. Evaluate and plot predicted f and u (strictly following Poisson snippet style)
# ------------------------------------------------------------------------------
xx_eval = xx_torch.detach()  # shape: ((Nx+1)^2, 2)
with torch.no_grad():
    f_out = model_f(xx_eval).cpu().reshape(Nx + 1, Nx + 1) / scaling
    u_out = model_u(xx_eval).cpu().reshape(Nx + 1, Nx + 1) / scaling

# Prepare true f, u for comparison
true_f_2d = (torch.from_numpy(true_f.reshape(Nx + 1, Nx + 1)) / scaling).cpu()
true_u_2d = (torch.from_numpy(true_u.reshape(Nx + 1, Nx + 1)) / scaling).cpu()

# section MSE
loss_fn = nn.MSELoss()
error_f_nn = loss_fn(true_f_2d, f_out)
error_u_nn = loss_fn(true_u_2d, u_out)
print("error_f_nn", error_f_nn.item())
print("error_u_nn", error_u_nn.item())

# Plot in the exact style: 2 rows, 3 columns
fig, axs = plt.subplots(2, 3, figsize=(10, 5))

# -- Row 0: f
im0 = axs[0, 0].imshow(true_f_2d, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 0].set_title("True f")
fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

axs[0, 1].set_title("Pred f (X)")
im1 = axs[0, 1].imshow(f_out, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

im2 = axs[0, 2].imshow(true_f_2d - f_out, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 2].set_title("error_f_nn")
fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

# -- Row 1: u
im3 = axs[1, 0].imshow(true_u_2d, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 0].set_title("True u")
fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

axs[1, 1].set_title("Pred u (Y)")
im4 = axs[1, 1].imshow(u_out, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

im5 = axs[1, 2].imshow(true_u_2d - u_out, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 2].set_title("error_u_nn")
fig.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"./results/{model_name}.png")
plt.show()
