import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

###############################################################################
# 1) Set device, define PDE domain points, and load data
###############################################################################
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Warning: NVIDIA GPU detected.')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Warning: Apple MPS detected.')
    else:
        device = torch.device('cpu')
        print('Warning: No NVIDIA GPU detected. Performance may be suboptimal.')
    print(f'Selected device: {device}')
    return device


device = get_device()

# Domain range and resolution
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
Nx, Ny = 512, 512
scaling = 100.0

# Create the full grid of PDE points in [0,1]x[0,1]
xx_lin = np.linspace(xmin, xmax, Nx + 1, dtype=np.float32)
yy_lin = np.linspace(ymin, ymax, Ny + 1, dtype=np.float32)
mesh = np.meshgrid(xx_lin, yy_lin)
xy = np.stack(mesh, axis=-1).reshape(-1, 2)
pde_points = torch.tensor(xy, device=device)

# Load reference data for (f, u) from file
raw_data = np.load("poisson_data_f_u.npy").astype(np.float32)
data_tensor = torch.tensor(raw_data, device=device)
data_tensor = scaling * data_tensor
data_fu = data_tensor[:, 1:]  # Adjust indexing based on your .npy structure

###############################################################################
# 2) Define simple PyTorch Datasets for PDE points & data points
###############################################################################
class PDEDataset(Dataset):
    def __init__(self, x_tensor):
        super().__init__()
        self.x = x_tensor

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


class DataDataset(Dataset):
    def __init__(self, x_tensor, data_tensor):
        super().__init__()
        self.x = x_tensor
        self.data = data_tensor

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.data[idx]


class CombinedDataset(Dataset):
    def __init__(self, pde_dataset, data_dataset):
        super().__init__()
        self.pde_dataset = pde_dataset
        self.data_dataset = data_dataset
        assert len(self.pde_dataset) == len(self.data_dataset), "Datasets must have the same length"

    def __len__(self):
        return len(self.pde_dataset)

    def __getitem__(self, idx):
        pde_batch = self.pde_dataset[idx]
        data_batch = self.data_dataset[idx]
        return pde_batch, data_batch


pde_dataset = PDEDataset(pde_points)
data_dataset = DataDataset(pde_points, data_fu)
combined_dataset = CombinedDataset(pde_dataset, data_dataset)

train_loader = DataLoader(combined_dataset, batch_size=50000, shuffle=False)

###############################################################################
# 3) Define MLPs for model_u and model_f
###############################################################################
def make_mlp(in_dim, out_dim, hidden=(36, 36, 36)):
    layers = []
    last_dim = in_dim
    for h in hidden:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.Tanh())
        last_dim = h
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class ModelU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = make_mlp(2, 1, hidden=(36, 36, 36))

    def forward(self, x):
        return self.net(x)


class ModelF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = make_mlp(2, 1, hidden=(36, 36, 36))

    def forward(self, x):
        return self.net(x)


model_u = ModelU().to(device)
model_f = ModelF().to(device)

###############################################################################
# 4) Define a LightningModule that combines PDE loss & data loss
###############################################################################
class PoissonLightningModule(pl.LightningModule):
    def __init__(self, model_u, model_f, weight_data=10000.0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model_u", "model_f"])
        self.model_u = model_u
        self.model_f = model_f
        self.weight_data = weight_data
        self.lr = lr

    def forward(self, x):
        u_pred = self.model_u(x)
        f_pred = self.model_f(x)
        return u_pred, f_pred

    def laplacian(self, u, x):
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        lap = torch.zeros_like(u)
        for i in range(x.shape[1]):
            grad2 = torch.autograd.grad(grad_u[:, i], x, grad_outputs=torch.ones_like(grad_u[:, i]), create_graph=True)[0][:, i].unsqueeze(-1)
            lap += grad2
        return lap

    def training_step(self, batch, batch_idx):
        pde_batch, data_batch = batch

        # PDE part
        x_pde = pde_batch.requires_grad_(True)
        u_pde, f_pde = self.forward(x_pde)
        lap_u = self.laplacian(u_pde, x_pde)
        pde_res = lap_u + f_pde
        pde_loss = torch.mean(pde_res ** 2)

        # Data part
        x_data, fu_data = data_batch
        u_ref = fu_data
        u_data_pred = self.model_u(x_data)
        data_loss = torch.mean((u_data_pred - u_ref) ** 2)

        loss = pde_loss + self.weight_data * data_loss
        self.log("pde_loss", pde_loss, prog_bar=True)
        self.log("data_loss", data_loss, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


###############################################################################
# 5) Trainer and fit
###############################################################################
pl.seed_everything(42)

lit_model = PoissonLightningModule(model_u, model_f, weight_data=10000.0, lr=1e-3).to(device)

trainer = pl.Trainer(
    accelerator="gpu" if device.type == "cuda" else "cuda",
    max_steps=1,
    logger=False,
    benchmark=True,
    enable_checkpointing=False
)

trainer.fit(lit_model, train_dataloaders=train_loader)

###############################################################################
# 6) Post-process and plot results
###############################################################################

model_u.eval()
model_f.eval()

with torch.no_grad():
    x_eval = pde_points
    u_pred, f_pred = lit_model.forward(x_eval)
    u_pred = (u_pred / scaling).cpu().numpy()
    f_pred = (f_pred / scaling).cpu().numpy()

u_out = u_pred.reshape(Nx + 1, Ny + 1)
f_out = f_pred.reshape(Nx + 1, Ny + 1)


fig, axis = plt.subplots(1, 2, figsize=(8, 3))
im0 = axis[0].imshow(f_out, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axis[0].set_title("Pred f (X)")

im1 = axis[1].imshow(u_out, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect='auto')
axis[1].set_title("Pred u (Y)")

plt.tight_layout()
plt.savefig("./results/pinn_baseline2")
plt.show()
