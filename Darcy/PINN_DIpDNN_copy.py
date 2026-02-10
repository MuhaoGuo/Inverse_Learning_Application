import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# 1. Device selection
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
# 2. Simple MLP (QRES) for Darcy: model_u(x)->u, model_f(x)->a (the coefficient)
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


# # model_u: X -> u
# model_u = QRES(input_dim=2, output_dim=1, hidden_dims=(36, 36, 36)).to(device)
# # model_f: X -> a  (the coefficient in Darcy)
# model_f = QRES(input_dim=2, output_dim=1, hidden_dims=(36, 36, 36)).to(device)
# They approximate the coefficient f(x) and the solution u(x).
model_f = QRES(input_dim=2, output_dim=1, hidden_dims=(30, 40, 40, 20, 20)).to(device)
model_u = QRES(input_dim=2, output_dim=1, hidden_dims=(50, 50, 20)).to(device)


# ------------------------------------------------------------------------------
# 3. Load data (Darcy): 'darcy_data_f_u.npy', shape: ((Nx+1)^2, 2) => [a, u]
# ------------------------------------------------------------------------------
scaling = 100.0
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
Nx, Ny = 512, 512

# PDE domain points (xx)
xx_1d = np.linspace(xmin, xmax, Nx + 1).astype(np.float32)
xx_2d = np.array(np.meshgrid(xx_1d, xx_1d)).T.reshape(-1, 2)  # shape: ((Nx+1)*(Ny+1), 2)
xx_torch = torch.from_numpy(xx_2d).to(device)  # PDE points

# Load file: 'darcy_data_f_u.npy'
#  => columns: [a, u]
loaded_data = np.load('darcy_data_f_u.npy').astype(np.float32)
loaded_data = scaling * loaded_data  # scale up

# True a, true u
true_f = loaded_data[:, 0:1]  # shape: ((Nx+1)^2, 1)
true_u = loaded_data[:, 1:2]  # shape: ((Nx+1)^2, 1)

# Data for training "u" on known solution
data_u = true_u

# Convert to torch
data_x_torch = xx_torch
data_u_torch = torch.from_numpy(data_u).to(device)

# Build DataLoaders
pde_dataset = torch.utils.data.TensorDataset(xx_torch)
data_dataset = torch.utils.data.TensorDataset(xx_torch, data_u_torch)

pde_loader = torch.utils.data.DataLoader(pde_dataset, batch_size=50000, shuffle=True, drop_last=False)
data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=50000, shuffle=True, drop_last=False)


# ------------------------------------------------------------------------------
# 4. Darcy PDE residual: div(f(x)*grad(u(x))) + 1 = 0
# ------------------------------------------------------------------------------
def divergence_of_f_grad_u(x, model_f, model_u):
    """
    Return div( f(x) * grad(u(x)) ).
    """
    x = x.detach()
    x.requires_grad_(True)

    f_val = model_f(x)  # (batch_size, 1)
    u_val = model_u(x)  # (batch_size, 1)

    grad_u = torch.autograd.grad(
        outputs=u_val,
        inputs=x,
        grad_outputs=torch.ones_like(u_val),
        create_graph=True
    )[0]  # shape: (batch_size, 2)

    # multiply: f(x)*grad_u => shape (batch_size, 2)
    f_grad_u = f_val * grad_u

    # divergence
    div_val = torch.zeros_like(u_val)
    for i in range(x.shape[1]):
        partial_i = torch.autograd.grad(
            outputs=f_grad_u[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(f_grad_u[:, i]),
            create_graph=True
        )[0][:, i].unsqueeze(-1)
        div_val = div_val + partial_i

    return div_val


def pde_residual(x, model_u, model_f):
    """
    Darcy PDE residual = div(f * grad(u)) + 1 => 0
    """
    div_term = divergence_of_f_grad_u(x, model_f, model_u)
    return div_term + 1.0


# ------------------------------------------------------------------------------
# 5. DipDNN definition (no change, just adapt for your new PDE)
# ------------------------------------------------------------------------------
class SmallNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

        self.register_buffer('mask_tril', torch.tril(torch.ones_like(self.fc1.weight)))
        self.register_buffer('mask_triu', torch.triu(torch.ones_like(self.fc2.weight)))

        self.fc1.weight = nn.Parameter(torch.mul(self.fc1.weight, self.mask_tril))
        self.fc2.weight = nn.Parameter(torch.mul(self.fc2.weight, self.mask_triu))

        # Force diagonal = 1.0
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

        # Register hooks
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
        # inverse of forward
        batch_size = y.shape[0]
        # 1) inverse leaky relu
        y1 = self.inverse_leaky_relu(y, self.negative_slope)
        # subtract bias
        y1 = y1 - self.fc2.bias
        # upper triangular solve
        y1_unsqueezed = y1.unsqueeze(2)
        fc2_weight_expanded = self.fc2.weight.unsqueeze(0).expand(batch_size, -1, -1)
        fc2_inv = torch.linalg.solve_triangular(fc2_weight_expanded, y1_unsqueezed, upper=True)
        x2 = fc2_inv.squeeze(2)

        # again inverse leaky relu
        x2 = self.inverse_leaky_relu(x2, self.negative_slope)
        x2 = x2 - self.fc1.bias
        fc1_weight_expanded = self.fc1.weight.unsqueeze(0).expand(batch_size, -1, -1)
        x2_unsqueezed = x2.unsqueeze(2)
        fc1_inv = torch.linalg.solve_triangular(fc1_weight_expanded, x2_unsqueezed, upper=False)
        x_final = fc1_inv.squeeze(2)
        return x_final

    def apply_weight_masks(self):
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)


class DipDNN_Block(nn.Module):
    def __init__(self, input_dim):
        super(DipDNN_Block, self).__init__()
        self.residual_function = SmallNetwork(input_dim)

    def forward(self, x):
        return self.residual_function(x)

    def inverse(self, y):
        return self.residual_function.inverse(y)

    def apply_weight_masks(self):
        self.residual_function.apply_weight_masks()


class DipDNN(nn.Module):
    def __init__(self, input_dim, num_blocks=2):
        super(DipDNN, self).__init__()
        self.blocks = nn.ModuleList([DipDNN_Block(input_dim) for _ in range(num_blocks)])
        self.input_dim = input_dim

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y):
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y

    def apply_weight_masks(self):
        for block in self.blocks:
            block.apply_weight_masks()


# Instantiate DipDNN with input_dim=513
model_d = DipDNN(input_dim=513, num_blocks=2).to(device)


# ------------------------------------------------------------------------------
# 6. Set up optimizer over all models
# ------------------------------------------------------------------------------
optimizer = optim.Adam(
    list(model_u.parameters()) + list(model_f.parameters()) + list(model_d.parameters()),
    lr=1e-3
)

pde_weight = 1.0    # PDE loss weight
data_weight = 10000.0  # data mismatch weight
dip_weight = 100.0     # weighting for DIP block
num_steps = 5000      # number of epochs
model_name = "PINN_DipDNN_Darcy_dw10000_hiddenmatch_5000epoch"


# ------------------------------------------------------------------------------
# 7. DIPDNN row-by-row MSE
# ------------------------------------------------------------------------------
def dip_loss_full_domain(model_d, f_true_2d, u_true_2d, device, do_inverse=True):
    """
    f_true_2d, u_true_2d: shape (513, 513).
    We'll do row-by-row to compute MSE( d(f_row), u_row ) and optionally
    MSE( d.inverse(u_row), f_row ). Return sum of forward & inverse MSE.
    """
    n_rows = f_true_2d.shape[0]  # = 513
    mse_fwd = 0.0
    mse_inv = 0.0

    for i in range(n_rows):
        row_f = f_true_2d[i, :].unsqueeze(0).to(device)  # (1, 513)
        row_u = u_true_2d[i, :].unsqueeze(0).to(device)  # (1, 513)
        # forward pass
        pred_u = model_d(row_f)
        mse_fwd += torch.mean((pred_u - row_u)**2)
        if do_inverse:
            # inverse pass
            pred_f = model_d.inverse(row_u)
            mse_inv += torch.mean((pred_f - row_f)**2)

    mse_fwd = mse_fwd / n_rows
    if do_inverse:
        mse_inv = mse_inv / n_rows
        return mse_fwd + mse_inv
    else:
        return mse_fwd


# ------------------------------------------------------------------------------
# 8. Training Loop
# ------------------------------------------------------------------------------
true_f_2d = torch.from_numpy(true_f.reshape(Nx + 1, Nx + 1))  # shape (513, 513)
true_u_2d = torch.from_numpy(true_u.reshape(Nx + 1, Nx + 1))  # shape (513, 513)

t0 = time.time()
step_count = 0
for epoch in range(num_steps):
    pde_iter = iter(pde_loader)
    data_iter = iter(data_loader)

    # We'll accumulate PDE & data losses over mini-batches
    running_loss = 0.0
    count_batches = 0

    while True:
        try:
            pde_batch = next(pde_iter)[0]            # (batch_size, 2)
            data_batch_x, data_batch_u = next(data_iter)
        except StopIteration:
            break

        # PDE loss
        residual = pde_residual(pde_batch, model_u, model_f)   # shape: (batch_size, 1)
        pde_loss = torch.mean(residual**2)

        # Data loss
        u_pred = model_u(data_batch_x)
        data_loss = torch.mean((u_pred - data_batch_u)**2)

        # Combine PDE + data
        mini_loss = pde_weight * pde_loss + data_weight * data_loss

        optimizer.zero_grad()
        mini_loss.backward()
        optimizer.step()

        running_loss += mini_loss.item()
        count_batches += 1
        step_count += 1

    # Now do DIP pass (full row-by-row). You can do it each epoch or every few epochs
    d_loss = dip_loss_full_domain(model_d, true_f_2d, true_u_2d, device, do_inverse=True)
    d_loss_total = dip_weight * d_loss

    optimizer.zero_grad()
    d_loss_total.backward()
    optimizer.step()

    if (epoch+1) % 1 == 0:
        avg_loss = running_loss / max(count_batches, 1)
        print(f"Epoch {epoch+1}/{num_steps}, PDE/Data avg. loss={avg_loss:.4e}, DIP loss={d_loss.item():.4e}")

t1 = time.time()
print(f"Training done after {num_steps} epochs in {t1 - t0:.2f} sec.")
print(f"Total PDE/Data steps: {step_count}")


# ------------------------------------------------------------------------------
# 9. Final Visualization (SAME OUTPUT STYLE)
#    We'll match your "Poisson code" style:
#    - Evaluate & plot predicted f, u
#    - Compare to true_f, true_u
#    - Also check DIP forward/inverse
# ------------------------------------------------------------------------------
# 9a) Evaluate model_f and model_u on domain
with torch.no_grad():
    f_out_model_f = model_f(xx_torch).reshape(Nx + 1, Ny + 1)
    u_out_model_u = model_u(xx_torch).reshape(Nx + 1, Ny + 1)

# 9b) DIP forward/inverse results
u_est_2d_real = torch.zeros_like(true_u_2d)  # DIP( f_true )
u_est_2d_fake = torch.zeros_like(true_u_2d)  # DIP( f_out_model_f )
f_est_2d_real = torch.zeros_like(true_f_2d)  # DIP.inverse( u_true )
f_est_2d_fake = torch.zeros_like(true_f_2d)  # DIP.inverse( u_out_model_u )

with torch.no_grad():
    for i in range(Nx + 1):
        # DIP( real f ) => u_est
        row_f_real = true_f_2d[i, :].unsqueeze(0).to(device)  # (1, 513)
        row_u_est = model_d(row_f_real)
        u_est_2d_real[i, :] = row_u_est.squeeze(0).cpu()

        # DIP( fake f ) => u_est
        row_f_fake = f_out_model_f[i, :].unsqueeze(0).to(device)
        row_u_est_fake = model_d(row_f_fake)
        u_est_2d_fake[i, :] = row_u_est_fake.squeeze(0).cpu()

        # DIP.inverse( real u ) => f_est
        row_u_real = true_u_2d[i, :].unsqueeze(0).to(device)
        row_f_est = model_d.inverse(row_u_real)
        f_est_2d_real[i, :] = row_f_est.squeeze(0).cpu()

        # DIP.inverse( fake u ) => f_est
        row_u_fake = u_out_model_u[i, :].unsqueeze(0).to(device)
        row_f_est_fake = model_d.inverse(row_u_fake)
        f_est_2d_fake[i, :] = row_f_est_fake.squeeze(0).cpu()


true_f_2d = true_f_2d / scaling
true_u_2d = true_u_2d / scaling
f_out_model_f = f_out_model_f.cpu() /scaling
u_out_model_u = u_out_model_u.cpu() /scaling
u_est_2d_fake = u_est_2d_fake / scaling
f_est_2d_fake = f_est_2d_fake / scaling
u_est_2d_real = u_est_2d_real / scaling
f_est_2d_real = f_est_2d_real / scaling
# ------------------------------------------------------------------------------
# 9c) MSE errors
# ------------------------------------------------------------------------------
loss_fn = nn.MSELoss()

error_f_nn  = loss_fn(true_f_2d, f_out_model_f)
error_u_nn  = loss_fn(true_u_2d, u_out_model_u)
error_f_real = loss_fn(true_f_2d, f_est_2d_real)
error_u_real = loss_fn(true_u_2d, u_est_2d_real)
error_f_fake = loss_fn(true_f_2d, f_est_2d_fake)
error_u_fake = loss_fn(true_u_2d, u_est_2d_fake)

print("error_f_nn",  error_f_nn.item())
print("error_u_nn",  error_u_nn.item())
print("error_f_real", error_f_real.item())
print("error_u_real", error_u_real.item())
print("error_f_fake", error_f_fake.item())
print("error_u_fake", error_u_fake.item())


# ------------------------------------------------------------------------------
# 9d) Plot in your requested style
#    We'll do a 4x4 subplot (similar to your final big figure),
#    but the first 2x2 is "f", next 2x2 is "u",
#    and the third/fourth rows show error maps.
# ------------------------------------------------------------------------------
fig, axs = plt.subplots(4, 4, figsize=(14, 10))

# The original "true_f" after dividing by scaling
im0 = axs[0, 0].imshow(true_f_2d, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 0].set_title("True f")
fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

im1 = axs[0, 1].imshow(f_out_model_f, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 1].set_title("f_out from model_f")
fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

im2 = axs[0, 2].imshow(f_est_2d_real, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 2].set_title("DIP f (inv from u_true)")
fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

im3 = axs[0, 3].imshow(f_est_2d_fake, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 3].set_title("DIP f (inv from u_est)")
fig.colorbar(im3, ax=axs[0, 3], fraction=0.046, pad=0.04)

# Row 1: "u"
im4 = axs[1, 0].imshow(true_u_2d, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 0].set_title("True u")
fig.colorbar(im4, ax=axs[1, 0], fraction=0.046, pad=0.04)

im5 = axs[1, 1].imshow(u_out_model_u, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 1].set_title("u_out from model_u")
fig.colorbar(im5, ax=axs[1, 1], fraction=0.046, pad=0.04)

im6 = axs[1, 2].imshow(u_est_2d_real, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 2].set_title("DIP u (f_true -> u)")
fig.colorbar(im6, ax=axs[1, 2], fraction=0.046, pad=0.04)

im7 = axs[1, 3].imshow(u_est_2d_fake, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 3].set_title("DIP u (f_est -> u)")
fig.colorbar(im7, ax=axs[1, 3], fraction=0.046, pad=0.04)

# Row 2: error for "f"
im8 = axs[2, 1].imshow(true_f_2d - f_out_model_f, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[2, 1].set_title("error_f_nn")
fig.colorbar(im8, ax=axs[2, 1], fraction=0.046, pad=0.04)

im9 = axs[2, 2].imshow(true_f_2d - f_est_2d_real, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[2, 2].set_title("error_f_real")
fig.colorbar(im9, ax=axs[2, 2], fraction=0.046, pad=0.04)

im10 = axs[2, 3].imshow(true_f_2d - f_est_2d_fake, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[2, 3].set_title("error_f_fake")
fig.colorbar(im10, ax=axs[2, 3], fraction=0.046, pad=0.04)

# Row 3: error for "u"
im11 = axs[3, 1].imshow(true_u_2d - u_out_model_u, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[3, 1].set_title("error_u_nn")
fig.colorbar(im11, ax=axs[3, 1], fraction=0.046, pad=0.04)

im12 = axs[3, 2].imshow(true_u_2d - u_est_2d_real, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[3, 2].set_title("error_u_real")
fig.colorbar(im12, ax=axs[3, 2], fraction=0.046, pad=0.04)

im13 = axs[3, 3].imshow(true_u_2d - u_est_2d_fake, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[3, 3].set_title("error_u_fake")
fig.colorbar(im13, ax=axs[3, 3], fraction=0.046, pad=0.04)

axs[2,0].set_visible(False)  # just an empty cell to keep layout
axs[3,0].set_visible(False)

plt.tight_layout()
plt.savefig(f"./results/{model_name}.png")
plt.show()
