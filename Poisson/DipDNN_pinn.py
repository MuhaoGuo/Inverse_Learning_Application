import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
# ------------------------------------------------------------------------
# Small Invertible Network
# ------------------------------------------------------------------------
class SmallNetwork(nn.Module):
    """
    Invertible layer with triangular masked weights. Input dim = 513.
    """
    def __init__(self, input_dim=513):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

        # Triangular masks
        self.register_buffer('mask_tril', torch.tril(torch.ones_like(self.fc1.weight)))
        self.register_buffer('mask_triu', torch.triu(torch.ones_like(self.fc2.weight)))

        # Apply masks
        self.fc1.weight = nn.Parameter(self.fc1.weight * self.mask_tril)
        self.fc2.weight = nn.Parameter(self.fc2.weight * self.mask_triu)

        # Make diagonal of each triangular weight = 1 for invertibility
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

        # Register hooks
        self.fc1.weight.register_hook(lambda grad: grad * self.mask_tril)
        self.fc2.weight.register_hook(lambda grad: grad * self.mask_triu)

        self.negative_slope = 0.5

    def forward(self, x):
        """
        Forward pass
        x shape: (batch_size, 513)
        """
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        return x

    def inverse_leaky_relu(self, z, negative_slope):
        """Inverse of LeakyReLU."""
        return torch.where(z >= 0, z, z / negative_slope)

    def inverse(self, y):
        """
        Inverse pass using solve_triangular.
        y shape: (batch_size, 513)
        """
        batch_size = y.shape[0]
        # Step 1: inverse activation of fc2
        y1 = self.inverse_leaky_relu(y, self.negative_slope)
        y1 = y1 - self.fc2.bias
        y1_unsqueezed = y1.unsqueeze(2)
        fc2_weight_expanded = self.fc2.weight.unsqueeze(0).expand(batch_size, -1, -1)
        x2 = torch.linalg.solve_triangular(fc2_weight_expanded, y1_unsqueezed, upper=True)
        x2 = x2.squeeze(2)

        # Step 2: inverse activation of fc1
        x2 = self.inverse_leaky_relu(x2, self.negative_slope)
        x2 = x2 - self.fc1.bias
        x2_unsqueezed = x2.unsqueeze(2)
        fc1_weight_expanded = self.fc1.weight.unsqueeze(0).expand(batch_size, -1, -1)
        x1 = torch.linalg.solve_triangular(fc1_weight_expanded, x2_unsqueezed, upper=False)
        x1 = x1.squeeze(2)
        return x1

    def apply_weight_masks(self):
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)


# ------------------------------------------------------------------------
# DIPDNN (row-based)
# ------------------------------------------------------------------------
class DipDNN(nn.Module):
    """
    DIP network that can invert:
       forward: row of f -> row of u
       inverse: row of u -> row of f
    input_dim=513 (since each row is 513 in length).
    """
    def __init__(self, input_dim=513, num_blocks=2):
        super(DipDNN, self).__init__()
        self.blocks = nn.ModuleList([SmallNetwork(input_dim) for _ in range(num_blocks)])
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (batch_size, 513)
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y):
        # y shape: (batch_size, 513)
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y

    def apply_weight_masks(self):
        for block in self.blocks:
            block.apply_weight_masks()


# ------------------------------------------------------------------------
# 2D Laplacian via finite differences
# ------------------------------------------------------------------------
def compute_laplacian_2d(u_2d, Nx=512, Ny=512):
    """
    u_2d shape: (513, 513).
    domain is [0,1]x[0,1], dx=1/Nx=1/512, dy=1/512
    """
    lap = torch.zeros_like(u_2d)
    dx = 1.0 / Nx
    dy = 1.0 / Ny

    # central difference in x
    lap[1:-1, :] += (u_2d[2:, :] - 2*u_2d[1:-1, :] + u_2d[0:-2, :]) / (dx*dx)
    # central difference in y
    lap[:, 1:-1] += (u_2d[:, 2:] - 2*u_2d[:, 1:-1] + u_2d[:, 0:-2]) / (dy*dy)
    return lap

# ------------------------------------------------------------------------
# (NEW) boundary condition loss
# ------------------------------------------------------------------------
def compute_boundary_loss(u_2d, bc_value=0.0):
    """
    Dirichlet boundary condition u = bc_value on the edges.
    u_2d shape: (513, 513).

    top (i=0), bottom (i=-1), left (j=0), right (j=-1)
    """
    top    = u_2d[0, :]
    bottom = u_2d[-1, :]
    left   = u_2d[:, 0]
    right  = u_2d[:, -1]

    boundary_vals = torch.cat([top, bottom, left, right], dim=0)  # shape (4*513,) = 2052
    bc_target = torch.full_like(boundary_vals, bc_value)
    loss_bc = F.mse_loss(boundary_vals, bc_target)
    return loss_bc


# ------------------------------------------------------------------------
# Row-wise data loading
# ------------------------------------------------------------------------
def get_poisson_data_rowwise(device="cpu"):
    data_np = np.load('poisson_data_f_u.npy').astype(np.float32)  # shape (263169, 2)
    Nx, Ny = 512, 512

    f_2d = data_np[:, 0].reshape(Nx+1, Ny+1)  # shape (513, 513)
    u_2d = data_np[:, 1].reshape(Nx+1, Ny+1)  # shape (513, 513)

    f_2d_t = torch.tensor(f_2d, device=device)  # (513, 513)
    u_2d_t = torch.tensor(u_2d, device=device)  # (513, 513)

    dataset = TensorDataset(f_2d_t, u_2d_t)
    loader = DataLoader(dataset, batch_size=513, shuffle=False)

    return loader, f_2d_t, u_2d_t


# ------------------------------------------------------------------------
# Training loop with 4 PDE/data losses + boundary conditions
# ------------------------------------------------------------------------
def train_dipdnn_pde_bc(
    model, train_loader, optimizer, epochs,
    Nx=512, Ny=512, device=torch.device('cpu'),
    w_data_fwd=1.0,
    w_data_inv=1.0,
    w_pde_fwd=1.0,
    w_pde_inv=0.0,
    w_bc_fwd=0.0,
    w_bc_inv=0.0,
    bc_value=0.0
):
    """
    - w_data_fwd: weight for forward data loss => MSE(u_true, u_pred)
    - w_data_inv: weight for inverse data loss => MSE(f_true, f_pred)
    - w_pde_fwd:  weight for forward PDE => -lap(u_pred) ~ f_true
    - w_pde_inv:  weight for inverse PDE => -lap(f_pred) ~ u_true (if that makes sense)
    - w_bc_fwd:   weight for boundary loss on u_pred
    - w_bc_inv:   weight for boundary loss on f_pred (if you want a boundary condition on the inverse)
    - bc_value:   the boundary condition value (Dirichlet)

    We'll do a single batch of size=513 for the row-based approach.
    """
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for f_batch, u_batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                f_batch = f_batch.to(device)  # shape (513,513)
                u_batch = u_batch.to(device)  # shape (513,513)

                optimizer.zero_grad()

                # -------------------------------------------------------
                # Forward pass row-by-row => u_pred_2d
                # -------------------------------------------------------
                u_pred_rows = []
                for i in range(f_batch.shape[0]):  # 513
                    row_in = f_batch[i, :].unsqueeze(0)  # shape (1, 513)
                    row_out = model(row_in)              # shape (1, 513)
                    u_pred_rows.append(row_out)
                u_pred_2d = torch.cat(u_pred_rows, dim=0)  # (513, 513)

                # -------------------------------------------------------
                # Inverse pass row-by-row => f_pred_2d
                # -------------------------------------------------------
                f_pred_rows = []
                for i in range(u_batch.shape[0]):  # 513
                    row_in = u_batch[i, :].unsqueeze(0)  # shape (1, 513)
                    row_out = model.inverse(row_in)      # shape (1, 513)
                    f_pred_rows.append(row_out)
                f_pred_2d = torch.cat(f_pred_rows, dim=0)  # (513, 513)

                # -------------------------------------------------------
                # 1) Forward Data Loss
                # -------------------------------------------------------
                data_loss_fwd = torch.tensor(0.0, device=device)
                if w_data_fwd > 0.0 :
                    data_loss_fwd = F.mse_loss(u_pred_2d, u_batch)

                # -------------------------------------------------------
                # 2) Inverse Data Loss
                # -------------------------------------------------------
                data_loss_inv = torch.tensor(0.0, device=device)
                if w_data_inv > 0.0:
                    data_loss_inv = F.mse_loss(f_pred_2d, f_batch)

                # -------------------------------------------------------
                # 3) Forward PDE Loss: -Delta(u_pred) ~ f_true
                # -------------------------------------------------------
                pde_loss_fwd = torch.tensor(0.0, device=device)
                if w_pde_fwd > 0.0:
                    lap_u_pred = compute_laplacian_2d(u_pred_2d, Nx, Ny)
                    pde_res_fwd = -lap_u_pred - f_batch
                    pde_loss_fwd = torch.mean(pde_res_fwd**2)

                # -------------------------------------------------------
                # 4) Inverse PDE Loss: -Delta(f_pred) ~ u_true
                # -------------------------------------------------------
                pde_loss_inv = torch.tensor(0.0, device=device)
                if w_pde_inv > 0.0:
                    lap_f_pred = compute_laplacian_2d(f_pred_2d, Nx, Ny)
                    pde_res_inv = -lap_f_pred - u_batch
                    pde_loss_inv = torch.mean(pde_res_inv**2)

                # -------------------------------------------------------
                # (NEW) 5) Forward Boundary Loss: bc on u_pred_2d
                # -------------------------------------------------------
                bc_loss_fwd = torch.tensor(0.0, device=device)
                if w_bc_fwd > 0.0:
                    bc_loss_fwd = compute_boundary_loss(u_pred_2d, bc_value=bc_value)

                # -------------------------------------------------------
                # (NEW) 6) Inverse Boundary Loss: bc on f_pred_2d
                #  (if you want a boundary condition for f)
                # -------------------------------------------------------
                bc_loss_inv = torch.tensor(0.0, device=device)
                if w_bc_inv > 0.0:
                    bc_loss_inv = compute_boundary_loss(f_pred_2d, bc_value=bc_value)

                # -------------------------------------------------------
                # Combine total loss
                # -------------------------------------------------------
                loss = (w_data_fwd*data_loss_fwd
                        + w_data_inv*data_loss_inv
                        + w_pde_fwd*pde_loss_fwd
                        + w_pde_inv*pde_loss_inv
                        + w_bc_fwd*bc_loss_fwd
                        + w_bc_inv*bc_loss_inv)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # tepoch.set_postfix(loss=running_loss / (tepoch._index + 1))
                tepoch.set_postfix(loss=running_loss / (tepoch.n or 1))


        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.6f}")



# ------------------------------------------------------------------------
# Putting it all together
# ------------------------------------------------------------------------
if __name__ == "__main__":
    device = get_device()
    set_seed(42)
    print("Using device:", device)

    # 1) Load row-based data
    train_loader, f_2d_t, u_2d_t = get_poisson_data_rowwise(device)
    Nx, Ny = 512, 512

    # 2) Create DIPDNN with input_dim = 513
    model = DipDNN(input_dim=513, num_blocks=3).to(device)

    # 3) Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model_name = "DipDNN_b_111"
    data_name = "Poisson"

    # 4) Train with boundary condition
    epochs = 1000
    train_dipdnn_pde_bc(
        model, train_loader, optimizer, epochs,
        Nx=Nx, Ny=Ny, device=device,
        w_data_fwd=1.0,
        w_data_inv=1.0,
        w_pde_fwd=1.0,
        w_pde_inv=0.0,
        w_bc_fwd=0.0,   # big weight for boundary condition on u
        w_bc_inv=0.0,     # if you also want boundary on f, set > 0
        bc_value=0.0      # Dirichlet boundary is 0
    )

    # 5) Evaluate: forward & inverse row-by-row
    model.eval()
    with torch.no_grad():
        # build u_pred_2d
        u_pred_rows = []
        for i in range(f_2d_t.shape[0]):  # 513
            row_in = f_2d_t[i, :].unsqueeze(0)
            row_out = model(row_in)
            u_pred_rows.append(row_out)
        u_pred_2d = torch.cat(u_pred_rows, dim=0)  # (513, 513)

        # build f_pred_2d
        f_pred_rows = []
        for i in range(u_2d_t.shape[0]):
            row_in = u_2d_t[i, :].unsqueeze(0)
            row_out = model.inverse(row_in)
            f_pred_rows.append(row_out)
        f_pred_2d = torch.cat(f_pred_rows, dim=0)  # (513, 513)

    # Convert to numpy for plotting
    f_2d_np = f_2d_t.cpu().numpy()
    u_2d_np = u_2d_t.cpu().numpy()
    u_pred_2d_np = u_pred_2d.cpu().numpy()
    f_pred_2d_np = f_pred_2d.cpu().numpy()

    # 6) Visualization (8 panels)
    fig, axis = plt.subplots(1, 8, figsize=(20, 2))
    xmin, xmax, ymin, ymax = 0, 1, 0, 1

    # panel 0: True u
    im0 = axis[0].imshow(u_2d_np, extent=[xmin, xmax, ymin, ymax],
                         origin='lower', aspect='auto')
    axis[0].set_title("True u")

    # panel 1: Pred u
    im1 = axis[1].imshow(u_pred_2d_np, extent=[xmin, xmax, ymin, ymax],
                         origin='lower', aspect='auto')
    axis[1].set_title("Pred u")

    # panel 2: Fwd error
    im2 = axis[2].imshow(u_2d_np - u_pred_2d_np, extent=[xmin, xmax, ymin, ymax],
                         origin='lower', aspect='auto')
    axis[2].set_title("Fwd error")

    # panel 3: True f
    im3 = axis[3].imshow(f_2d_np, extent=[xmin, xmax, ymin, ymax],
                         origin='lower', aspect='auto')
    axis[3].set_title("True f")

    # panel 4: Fake inv f (f_pred from f->u->inverse?)
    # Actually the "fake" inverse from f->u->f would be the same as doing row_in = u_pred_2d.
    # But we'll just show the "real" inverse from y->x (u_true->f_pred).
    # We'll rename this "Inverse f" for clarity.
    im4 = axis[4].imshow(f_pred_2d_np, extent=[xmin, xmax, ymin, ymax],
                         origin='lower', aspect='auto')
    axis[4].set_title("Inverse f")

    # panel 5: Inverse error => f_true - f_pred
    im5 = axis[5].imshow(f_2d_np - f_pred_2d_np, extent=[xmin, xmax, ymin, ymax],
                         origin='lower', aspect='auto')
    axis[5].set_title("Inv error")

    # For completeness, we can do "fake inv" from the predicted u_pred_2d as well:
    # But let's skip or show side-by-side:

    # panel 6: "fake" inverse from u_pred_2d
    fake_inv_rows = []
    for i in range(u_pred_2d.shape[0]):
        row_in = u_pred_2d[i, :].unsqueeze(0)
        row_out = model.inverse(row_in)
        fake_inv_rows.append(row_out)
    fake_inv_2d = torch.cat(fake_inv_rows, dim=0)
    fake_inv_2d_np = fake_inv_2d.detach().cpu().numpy()

    im6 = axis[6].imshow(fake_inv_2d_np - f_2d_np, extent=[xmin, xmax, ymin, ymax],
                         origin='lower', aspect='auto')
    axis[6].set_title("Fake inv err")

    # panel 7: difference between fake_inv_2d and real f_2d
    # (already done above). We'll just rename or show " "

    im7 = axis[7].imshow(fake_inv_2d_np, extent=[xmin, xmax, ymin, ymax],
                         origin='lower', aspect='auto')
    axis[7].set_title("Fake inv f")

    plt.tight_layout()
    plt.savefig(f"./results/{data_name}_{model_name}.png", dpi=120)
    plt.show()

    # 7) MSE calculations
    fwd_error = mean_squared_error(u_2d_np.ravel(), u_pred_2d_np.ravel())
    inv_error = mean_squared_error(f_2d_np.ravel(), f_pred_2d_np.ravel())
    fake_inv_error = mean_squared_error(f_2d_np.ravel(), fake_inv_2d_np.ravel())

    print("Train Data Results:")
    print(f"Forward MSE (u_true vs u_pred): {fwd_error}")
    print(f"Inverse MSE (f_true vs f_pred): {inv_error}")
    print(f"Fake Inverse MSE (f_true vs model.inverse(u_pred)): {fake_inv_error}")
