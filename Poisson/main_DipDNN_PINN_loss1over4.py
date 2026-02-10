import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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
# 2. Simple MLP (QRES) for u and f
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


# model_u: X -> u
model_u = QRES(input_dim=2, output_dim=1, hidden_dims=(36, 36, 36)).to(device)
# model_f: X -> f
model_f = QRES(input_dim=2, output_dim=1, hidden_dims=(36, 36, 36)).to(device)

# ------------------------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------------------------
scaling = 100.0
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
Nx, Ny = 512, 512

# PDE domain points (xx)
xx_1d = np.linspace(xmin, xmax, Nx + 1).astype(np.float32)
xx_2d = np.array(np.meshgrid(xx_1d, xx_1d)).T.reshape(-1, 2)  # shape: ( (Nx+1)*(Ny+1), 2 )
xx_torch = torch.from_numpy(xx_2d).to(device)  # PDE points

# Load file: shape is ((Nx+1)^2, 2) => columns: [f, u]
loaded_data = np.load('poisson_data_f_u.npy').astype(np.float32)
loaded_data = scaling * loaded_data  # scale up

# True f, true u
true_f = loaded_data[:, 0:1]  # shape: ( (Nx+1)^2, 1 )
true_u = loaded_data[:, 1:2]  # shape: ( (Nx+1)^2, 1 )

# For your snippet, "data_u" is just the second column (true u)
data_u = true_u  # shape: ((Nx+1)^2, 1)

# Convert to torch
data_x_torch = xx_torch  # same input points
data_u_torch = torch.from_numpy(data_u).to(device)


pde_dataset = torch.utils.data.TensorDataset(xx_torch)
data_dataset = torch.utils.data.TensorDataset(xx_torch, data_u_torch)
# print(xx_torch.shape)
# print(data_u_torch.shape)
# exit()

pde_loader = torch.utils.data.DataLoader(pde_dataset, batch_size=50000, shuffle=True, drop_last=False)
data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=50000, shuffle=True, drop_last=False)


# ------------------------------------------------------------------------------
# 4. PDE residual
# ------------------------------------------------------------------------------
def laplacian(model_u, x):
    x = x.detach()
    x.requires_grad_(True)
    u = model_u(x)  # shape: (batch_size, 1)

    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]  # shape: (batch_size, 2)

    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        d2u_dxi2 = torch.autograd.grad(
            outputs=grad_u[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(grad_u[:, i]),
            create_graph=True
        )[0][:, i].unsqueeze(-1)
        lap = lap + d2u_dxi2
    return lap


def pde_residual(x, model_u, model_f):
    lap_u = laplacian(model_u, x)
    f_x = model_f(x)
    return lap_u + f_x  # want this = 0


# ------------------------------------------------------------------------------
# 5. DipDNN definition (from your snippet)
#    We add "model_d" with input_dim=513, out_dim=513, hidden_dim=513.
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

        # Register hooks to keep masks
        self.fc1.weight.register_hook(lambda grad: grad * self.mask_tril)
        self.fc2.weight.register_hook(lambda grad: grad * self.mask_triu)

        self.negative_slope = 0.5

    def inverse_leaky_relu(self, z, negative_slope):
        # inverse of LeakyReLU
        return torch.where(z >= 0, z, z / negative_slope)

    def forward(self, x):
        fc1_fwd = self.fc1(x)
        fc1_fwd = F.leaky_relu(fc1_fwd, negative_slope=self.negative_slope)
        fc2_fwd = self.fc2(fc1_fwd)
        fc2_fwd = F.leaky_relu(fc2_fwd, negative_slope=self.negative_slope)
        return fc2_fwd

    def inverse(self, y):
        # Invert forward pass
        batch_size = y.shape[0]

        # 1) inverse leaky relu
        y1 = self.inverse_leaky_relu(y, self.negative_slope)
        # subtract bias
        y1 = y1 - self.fc2.bias

        # triangular solve for fc2.weight
        y1_unsqueezed = y1.unsqueeze(2)  # shape (B, input_dim, 1)
        fc2_weight_expanded = self.fc2.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # upper triangular => solve
        fc2_inv = torch.linalg.solve_triangular(fc2_weight_expanded, y1_unsqueezed, upper=True)
        x2 = fc2_inv.squeeze(2)

        # 2) again inverse leaky relu
        x2 = self.inverse_leaky_relu(x2, self.negative_slope)
        x2 = x2 - self.fc1.bias

        fc1_weight_expanded = self.fc1.weight.unsqueeze(0).expand(batch_size, -1, -1)
        x2_unsqueezed = x2.unsqueeze(2)
        # lower triangular => solve
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
        # x shape: (batch_size, input_dim)
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
# 6. Set up optimizer
#    We optimize all three models at once: model_u, model_f, model_d
# ------------------------------------------------------------------------------
optimizer = optim.Adam(
    list(model_u.parameters()) + list(model_f.parameters()) + list(model_d.parameters()),
    lr=1e-3
)

pde_weight = 1.0  # PDE loss weight
data_weight = 10000.0  # data mismatch weight (like your snippet)
dip_weight = 100.0  # weighting for the DipDNN loss, tune as needed
num_steps = 1000 #2000  # 1000
model_name = "DipDNN_PINN_dip100_1over4"

# ------------------------------------------------------------------------------
# 7. DipDNN row-by-row MSE
#    We'll define a helper that computes MSE( model_d(f_true_row), u_true_row )
#    and optionally MSE( model_d.inverse(u_true_row), f_true_row ), if you want
#    to train the invertibility as well.
# ------------------------------------------------------------------------------
def dip_loss_full_domain(model_d, f_true_2d, u_true_2d, device, do_inverse=True):
    """
    f_true_2d, u_true_2d: shape (513, 513) each, on CPU or GPU.
    We'll compute row-by-row forward pass: MSE( d(f_row), u_row ).
    If do_inverse=True, also compute MSE( d.inverse(u_row), f_row ).
    Return the total MSE as a single scalar tensor.
    """

    n_rows = f_true_2d.shape[0]  # = 513
    mse_fwd = 0.0
    mse_inv = 0.0

    for i in range(n_rows):
        # forward direction
        row_f = f_true_2d[i, :].unsqueeze(0)  # shape (1, 513)
        row_u = u_true_2d[i, :].unsqueeze(0)  # shape (1, 513)

        # forward pass
        pred_u = model_d(row_f.to(device))  # shape (1, 513)
        mse_fwd += torch.mean((pred_u - row_u.to(device)) ** 2)

        if do_inverse:
            # inverse direction
            pred_f = model_d.inverse(row_u.to(device))  # shape (1, 513)
            mse_inv += torch.mean((pred_f - row_f.to(device)) ** 2)

    # average over rows
    mse_fwd = mse_fwd / n_rows
    if do_inverse:
        mse_inv = mse_inv / n_rows
        return mse_fwd + mse_inv
    else:
        return mse_fwd


# ------------------------------------------------------------------------------
# 8. Training Loop
#    We'll do PDE & data steps in mini-batches, and after finishing those
#    we do a DipDNN pass on the entire domain (row-by-row).
# ------------------------------------------------------------------------------
true_f_2d = torch.from_numpy(true_f.reshape(Nx + 1, Nx + 1))  # shape (513, 513)
true_u_2d = torch.from_numpy(true_u.reshape(Nx + 1, Nx + 1))  # shape (513, 513)

t0 = time.time()
step_count = 0
for epoch in range(num_steps):
    # PDE & data steps in mini-batches
    pde_iter = iter(pde_loader)
    data_iter = iter(data_loader)

    # We'll accumulate PDE loss and data loss over mini-batches, then do one step
    # (You could also do multiple steps, but let's illustrate a "single step at the end" approach.)
    running_loss = 0.0
    count_batches = 0

    if epoch % 4 == 0:
        while True:
            try:
                pde_batch = next(pde_iter)[0]  # shape (batch_size, 2)
                data_batch_x, data_batch_y = next(data_iter)  # (batch_size, 2), (batch_size, 1)
            except StopIteration:
                break

            # PDE loss
            residual = pde_residual(pde_batch, model_u, model_f)  # shape (batch_size, 1)
            pde_loss = torch.mean(residual ** 2)

            # Data loss
            u_pred = model_u(data_batch_x)
            data_loss = torch.mean((u_pred - data_batch_y) ** 2)

            # Combine PDE + data
            mini_loss = pde_weight * pde_loss + data_weight * data_loss

            # Accumulate
            running_loss += mini_loss.item()
            count_batches += 1

            # Backprop mini-batch
            optimizer.zero_grad()
            mini_loss.backward()
            optimizer.step()

            step_count += 1

    # Now do DipDNN step (full domain):
    # Because we want to pass entire (513, 513) f and u row-by-row
    # We'll do an additional forward/backward pass:
    d_loss = dip_loss_full_domain(model_d, true_f_2d, true_u_2d, device, do_inverse=True)
    # Weighted by dip_weight
    optimizer.zero_grad()
    d_loss_total = dip_weight * d_loss
    d_loss_total.backward()
    optimizer.step()

    # Print something every 50 epochs or so
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{num_steps}")
        print(f"  PDE/Data average loss (per mini-batch): {running_loss / max(count_batches, 1):.4e}")
        print(f"  DipDNN loss: {d_loss.item():.4e}")

t1 = time.time()
print(f"Training finished after {num_steps} epochs in {t1 - t0:.2f} seconds.")
print(f"Total steps of PDE/Data loops: {step_count}")


## save model: ---------------------------------------------------
torch.save({
    'model_u': model_u.state_dict(),
    'model_f': model_f.state_dict(),
    'model_d': model_d.state_dict()
}, f"{model_name}.pth")

checkpoint = torch.load(f"{model_name}.pth")
model_u.load_state_dict(checkpoint['model_u'])
model_f.load_state_dict(checkpoint['model_f'])
model_d.load_state_dict(checkpoint['model_d'])

# Set to evaluation mode
model_u.eval()
model_f.eval()
model_d.eval()


# ------------------------------------------------------------------------------
# 9. Final Visualization
#    We do NOT use model_u or model_f now. We only use model_d.
#    a) "Given true f, u_estimate = model_d(f)."
#    b) "Given true u, f_estimate = model_d.inverse(u)."
# ------------------------------------------------------------------------------
# Reshape your true f, true u -> (513, 513)
# We'll produce 2D arrays: u_est_2d, f_est_2d


with torch.no_grad():
    u_est_2d_fake = torch.zeros_like(true_u_2d)
    f_est_2d_fake = torch.zeros_like(true_f_2d)
    u_est_2d_real = torch.zeros_like(true_u_2d)
    f_est_2d_real = torch.zeros_like(true_f_2d)

    # f_out from model_f
    f_out_model_f = model_f(xx_torch).reshape(Nx + 1, Ny + 1)
    # u_out from model_u
    u_out_model_u = model_u(xx_torch).reshape(Nx + 1, Ny + 1)

    # model_d input real u and f
    for i in range(Nx + 1):
        # u real
        row_f = true_f_2d[i, :].unsqueeze(0).to(device)  # shape (1, 513)
        row_u_est = model_d(row_f)  # shape (1, 513)
        u_est_2d_real[i, :] = row_u_est.squeeze(0).cpu()
        # u real
        row_f = f_out_model_f[i, :].unsqueeze(0).to(device)  # shape (1, 513)
        row_u_est = model_d(row_f)  # shape (1, 513)
        u_est_2d_fake[i, :] = row_u_est.squeeze(0).cpu()

    for i in range(Nx + 1):
        # f real
        row_u = true_u_2d[i, :].unsqueeze(0).to(device)
        row_f_est = model_d.inverse(row_u)
        f_est_2d_real[i, :] = row_f_est.squeeze(0).cpu()
        # f fake
        row_u = u_out_model_u[i, :].unsqueeze(0).to(device)
        row_f_est = model_d.inverse(row_u)
        f_est_2d_fake[i, :] = row_f_est.squeeze(0).cpu()

# Scale back down if needed
f_true_plot = true_f_2d / scaling
u_true_plot = true_u_2d/ scaling
f_out_model_f = f_out_model_f.cpu() /scaling
u_out_model_u = u_out_model_u.cpu() /scaling

u_est_2d_fake = u_est_2d_fake / scaling
f_est_2d_fake = f_est_2d_fake / scaling
u_est_2d_real = u_est_2d_real / scaling
f_est_2d_real = f_est_2d_real / scaling

# section MSE:
loss = nn.MSELoss()
error_f_nn = loss(f_true_plot, f_out_model_f)
error_u_nn = loss(u_true_plot, u_out_model_u)
error_f_real = loss(f_true_plot, f_est_2d_real)
error_u_real = loss(u_true_plot, u_est_2d_real)
error_f_fake = loss(f_true_plot, f_est_2d_fake)
error_u_fake = loss(u_true_plot, u_est_2d_fake)
print("error_f_nn", error_f_nn.item())
print("error_u_nn", error_u_nn.item())
print("error_f_real", error_f_real.item())
print("error_u_real", error_u_real.item())
print("error_f_fake", error_f_fake.item())
print("error_u_fake", error_u_fake.item())


# 1.



# Plot result
fig, axs = plt.subplots(4, 4, figsize=(14, 10))

# True f
im0 = axs[0, 0].imshow(f_true_plot, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 0].set_title("True f")
fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

# f_out_model_f
im1 = axs[0, 1].imshow(f_out_model_f, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 1].set_title("f_out from model_f")
fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

# DIP estimate of f from u -- real
im2 = axs[0, 2].imshow(f_est_2d_real, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 2].set_title("DIP estimate of f (inverse from u)")
fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

# DIP estimate of f from u_est -- fake
im3 = axs[0, 3].imshow(f_est_2d_fake, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[0, 3].set_title("DIP estimate of f (inverse from u_est)")
fig.colorbar(im3, ax=axs[0, 3], fraction=0.046, pad=0.04)

# True u
im4 = axs[1, 0].imshow(u_true_plot, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 0].set_title("True u")
fig.colorbar(im4, ax=axs[1, 0], fraction=0.046, pad=0.04)

# u_out_model_u
im5 = axs[1, 1].imshow(u_out_model_u, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 1].set_title("u_out from model_u")
fig.colorbar(im5, ax=axs[1, 1], fraction=0.046, pad=0.04)

# DIP estimate of u from f real
im6 = axs[1, 2].imshow(u_est_2d_real, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 2].set_title("DIP estimate of u (forward from f)")
fig.colorbar(im6, ax=axs[1, 2], fraction=0.046, pad=0.04)

# DIP estimate of u from f_est fake
im7 = axs[1, 3].imshow(u_est_2d_fake, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[1, 3].set_title("DIP estimate of u (forward from f_est)")
fig.colorbar(im7, ax=axs[1, 3], fraction=0.046, pad=0.04)


#  error_f_nn
im8 = axs[2, 1].imshow(f_true_plot - f_out_model_f, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[2, 1].set_title("error_f_nn")
fig.colorbar(im8, ax=axs[2, 1], fraction=0.046, pad=0.04)

# error_f_real
im9 = axs[2, 2].imshow(f_true_plot-f_est_2d_real, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[2, 2].set_title("error_f_real")
fig.colorbar(im9, ax=axs[2, 2], fraction=0.046, pad=0.04)

# error_f_fake
im10 = axs[2, 3].imshow(f_true_plot-f_est_2d_fake , extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[2, 3].set_title("error_f_fake")
fig.colorbar(im10, ax=axs[2, 3], fraction=0.046, pad=0.04)


#  error_u_nn
im11 = axs[3, 1].imshow(u_true_plot-u_out_model_u, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[3, 1].set_title("error_u_nn")
fig.colorbar(im11, ax=axs[3, 1], fraction=0.046, pad=0.04)

# error_u_real
im12 = axs[3, 2].imshow(u_true_plot-u_est_2d_real, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[3, 2].set_title("error_u_real")
fig.colorbar(im12, ax=axs[3, 2], fraction=0.046, pad=0.04)

# error_u_fake
im13 = axs[3, 3].imshow(u_true_plot-u_est_2d_fake, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axs[3, 3].set_title("error_u_fake")
fig.colorbar(im13, ax=axs[3, 3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"./results/{model_name}.png")
plt.show()
