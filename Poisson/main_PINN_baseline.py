import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# 1. Device selection (similar to get_device in your snippet)
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
# 2. Simple MLP (similar to QRES) for u and f
#    We add skip connections or residuals in a mild way to mimic QRES,
#    but you can tailor it further if QRES is more specialized.
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


# Create two models: model_u and model_f
model_u = QRES(input_dim=2, output_dim=1, hidden_dims=(36, 36, 36)).to(device)
model_f = QRES(input_dim=2, output_dim=1, hidden_dims=(36, 36, 36)).to(device)

# ------------------------------------------------------------------------------
# 3. Load data and create DataLoader(s)
#    - We replicate your domain points xx (the PDE "grid")
#    - We replicate your data for the boundary/solution constraints
# ------------------------------------------------------------------------------
scaling = 100.0
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
Nx, Ny = 512, 512

# PDE domain points (xx)
xx_1d = np.linspace(xmin, xmax, Nx + 1).astype(np.float32)
xx_2d = np.array(np.meshgrid(xx_1d, xx_1d)).T.reshape(-1, 2)  # shape: ( (Nx+1)*(Ny+1), 2 )
xx_torch = torch.from_numpy(xx_2d).to(device)  # PDE points

# Load data (f,u) but in your snippet you only use u from data:
# data file shape is presumably ( (Nx+1)^2, 2 ) => [f, u].
loaded_data = np.load('poisson_data_f_u.npy').astype(np.float32)
loaded_data = scaling * loaded_data  # scale
# According to your snippet, you do data = data[:, 1:], i.e. only the 'u' column
# so data_y corresponds to "u" values
# We'll keep them separate here: data_u is the second column of loaded_data
# data_u = loaded_data[:, 1:2]  # shape: ( (Nx+1)^2, 1 )
true_f = loaded_data[:, 0:1]  # shape: ( (Nx+1)^2, 1 )
true_u = loaded_data[:, 1:2]  # shape: ( (Nx+1)^2, 1 )
true_f_2d = torch.from_numpy(true_f.reshape(Nx + 1, Nx + 1))  # shape (513, 513)
true_u_2d = torch.from_numpy(true_u.reshape(Nx + 1, Nx + 1))  # shape (513, 513)



# Convert to torch
data_x_torch = xx_torch  # same mesh points for input
data_u_torch = torch.from_numpy(true_u).to(device)

# Build PyTorch DataLoaders. We use large batch_size=50,000 (like your code).
# PDE "dataset" is just the entire domain points (xx_torch)
# Data "dataset" has the same x but target is data_u_torch
pde_dataset = torch.utils.data.TensorDataset(xx_torch)
data_dataset = torch.utils.data.TensorDataset(xx_torch, data_u_torch)

pde_loader = torch.utils.data.DataLoader(pde_dataset, batch_size=50000, shuffle=True, drop_last=False)
data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=50000, shuffle=True, drop_last=False)


# ------------------------------------------------------------------------------
# 4. PDE residual function: laplacian(u) + f
#    We manually compute second derivatives wrt x,y using autograd.
# ------------------------------------------------------------------------------
def laplacian(model_u, x):
    """
    Compute laplacian of model_u wrt x.
    x shape: (batch_size, 2)
    """
    # Enable gradient for x if not already
    x = x.detach()
    x.requires_grad_(True)

    # Forward pass
    u = model_u(x)  # shape: (batch_size, 1)

    # First derivatives
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]  # shape: (batch_size, 2), i.e. du/dx, du/dy

    # Second derivatives (sum of diagonal second partials)
    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        # derivative wrt x[:, i]
        d2u_dxi2 = torch.autograd.grad(
            outputs=grad_u[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(grad_u[:, i]),
            create_graph=True
        )[0][:, i].unsqueeze(-1)
        lap = lap + d2u_dxi2

    return lap


def pde_residual(x, model_u, model_f):
    """
    PDE residual = laplacian(u) + f(x).
    We want laplacian(u) + f = 0 => residual = 0
    """
    lap_u = laplacian(model_u, x)
    f_x = model_f(x)
    return lap_u + f_x


# ------------------------------------------------------------------------------
# 5. Training Loop
#    Combine PDE residual loss and data mismatch loss (u_pred vs. u_data).
#    Weighted by 10000.0 for the data condition, as in your snippet.
# ------------------------------------------------------------------------------
optimizer = optim.Adam(list(model_u.parameters()) + list(model_f.parameters()), lr=1e-3)

num_steps = 1000
pde_weight = 1.0  # PDE part has implicit weight 1
data_weight = 10000.0  # data mismatch part
model_name = "PINN_baseline"
t0 = time.time()
step_count = 0
# We will cycle through PDE and data loaders in parallel. If one finishes first,
# we just stop that epoch (like standard "zip" approach).
for epoch in range(num_steps):
    # Create iterators
    pde_iter = iter(pde_loader)
    data_iter = iter(data_loader)

    while True:
        try:
            pde_batch = next(pde_iter)[0]  # (batch_size, 2)
            data_batch_x, data_batch_y = next(data_iter)  # (batch_size, 2), (batch_size, 1)
        except StopIteration:
            break

        # PDE loss
        residual = pde_residual(pde_batch, model_u, model_f)  # (batch_size, 1)
        pde_loss = torch.mean(residual ** 2)

        # Data loss
        u_pred = model_u(data_batch_x)
        data_loss = torch.mean((u_pred - data_batch_y) ** 2)

        # Combined loss
        loss = pde_weight * pde_loss + data_weight * data_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_count += 1

    # Print something every 50 epochs or so
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{num_steps} | PDE loss={pde_loss.item():.4e} | Data loss={data_loss.item():.4e}")

t1 = time.time()
print(f"Training finished after {num_steps} epochs, in {t1 - t0:.2f} seconds.")

# ------------------------------------------------------------------------------
# 6. Evaluate and plot predicted f and u
# ------------------------------------------------------------------------------
xx_eval = xx_torch.detach()  # shape: ( (Nx+1)^2, 2 )
with torch.no_grad():
    f_out = model_f(xx_eval).cpu().reshape(Nx + 1, Nx + 1) / scaling
    u_out = model_u(xx_eval).cpu().reshape(Nx + 1, Nx + 1) / scaling


# section MSE
true_f_2d = true_f_2d/scaling
true_u_2d = true_u_2d/scaling

loss = nn.MSELoss()
error_f_nn = loss(true_f_2d, f_out)
error_u_nn = loss(true_u_2d, u_out)
print("error_f_nn", error_f_nn.item())
print("error_u_nn", error_u_nn.item())


# Plot
fig, axs = plt.subplots(2, 3, figsize=(10, 5))

im0 = axs[0, 0].imshow(true_f_2d, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 0].set_title("True f")
fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

axs[0, 1].set_title("Pred f (X)")
im1 = axs[0, 1].imshow(f_out, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

#  error_f_nn
im2 = axs[0, 2].imshow(true_f_2d - f_out, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 2].set_title("error_f_nn")
fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)


im3 = axs[1, 0].imshow(true_u_2d, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 0].set_title("True u")
fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

axs[1, 1].set_title("Pred u (Y)")
im4 = axs[1, 1].imshow(u_out, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

#  error_u_nn
im5 = axs[1, 2].imshow(true_u_2d - u_out, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 2].set_title("error_u_nn")
fig.colorbar(im5, ax=axs[1,2], fraction=0.046, pad=0.04)


plt.tight_layout()
plt.savefig(f"./results/{model_name}.png")
plt.show()