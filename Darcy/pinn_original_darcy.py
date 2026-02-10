import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torchphysics as tp
import scipy.signal
import h5py
import torch
import numpy as np
import os
import pytorch_lightning as pl
# from utils import get_device
import time
import matplotlib.pyplot as plt

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

''' 
X = tp.spaces.R2('x') # input is 2D
U = tp.spaces.R1('u') # output is 1D

square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])

model = tp.models.FCN(input_space=X, output_space=U, hidden=(50, 50, 50, 20))

# load external data

scaling = 0.01
steps = 8
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T[::steps, ::steps]
a = scaling * np.load('darcy_data_f_u.npy').astype(np.float32)[:, :1]


a = a.reshape(513, 513)
a = a[::steps, ::steps]
kernel = [[0, 0, 0], [1, 0, -1], [0, 0, 0]]
a_x = scipy.signal.convolve2d(a, kernel, mode="same", boundary='symm') * 513/steps
kernel = [[0, 1, 0], [0, 0, 0], [0, -1, 0]]
a_y = scipy.signal.convolve2d(a, kernel, mode="same", boundary='symm') * 513/steps

a = torch.tensor(a, dtype=torch.float32)
a_x, a_y = torch.tensor(a_x, dtype=torch.float32), torch.tensor(a_y, dtype=torch.float32)
weights = torch.ones_like(a)
weights[a_x != 0] = steps/513
weights[a_y != 0] = steps/513
a = a.reshape(-1, 1)
a_grad = torch.column_stack((a_x.reshape(-1, 1), a_y.reshape(-1, 1)))
weights = weights.reshape(-1, 1)

xx = xx.reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx), X)
pde_sampler = tp.samplers.DataSampler(xx)


def pde_residual(u, x):
    u_grad = tp.utils.grad(u, x)
    conv = torch.sum(u_grad * a_grad, dim=1).reshape(-1, 1)
    return (conv + a * tp.utils.laplacian(u, x, grad=u_grad) + 1.0) * weights


pde_cond = tp.conditions.PINNCondition(module=model, sampler=pde_sampler,
                                       residual_fn=pde_residual)


# boundary condition:
def bound_residual(u):
    return u

bound_sampler = tp.samplers.GridSampler(square.boundary, n_points=int(2052)).make_static()
bound_cond = tp.conditions.PINNCondition(module=model, sampler=bound_sampler,
                                         residual_fn=bound_residual, weight=100)


# here we start with Adam:
optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)

solver = tp.solver.Solver(train_conditions=[pde_cond, bound_cond], optimizer_setting=optim)

device = get_device()
a = a.to(device)
a_grad = a_grad.to(device)
weights = weights.to(device)
trainer = pl.Trainer(
                    # gpus=1,  # or None if CPU is used
                    accelerator="mps" if device.type == "mps" else None,
                     max_steps=1,
                     logger=False,
                     benchmark=True,
                     # checkpoint_callback=False
)

trainer.fit(solver)


xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx), X)
true_data = torch.tensor(np.load('darcy_data_f_u.npy').astype(np.float32))
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title(r"True u")
plt.imshow(true_data[:, 1].reshape(Nx+1, Nx+1), extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()


# first move the GPU:
xx = xx.to(device)
model.to(device)
start = time.time()
#u_out = u_constrain(model(xx).as_tensor, xx.as_tensor)
u_out = model(xx)
print('testing time:', time.time() - start)
u_out = (u_out.as_tensor * scaling).detach().cpu().numpy().reshape(1, Nx+1, Nx+1)
fig = plt.figure() #figsize=(10+4, 5)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title(r"PINN u")
plt.imshow(u_out[0], extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()
np.save("new_u_qres", u_out)


fig = plt.figure() #figsize=(10+4, 5)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title(r"error u")
diff = np.abs(u_out[0]-true_data[:, 1].numpy().reshape(Nx+1, Nx+1))
l2_rel = np.sqrt(np.sum(diff**2))/np.sqrt(sum(true_data[:, 1]**2))
plt.imshow(diff,
           extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()
print(l2_rel)
plt.show()
'''

##############################################################################
##############################################################################
# section inverse  ###########################################################
##############################################################################
##############################################################################
device = get_device()
print(device)

X = tp.spaces.R2('x')
U = tp.spaces.R1('u')
A = tp.spaces.R1('a')

square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])

model_u = tp.models.FCN(input_space=X, output_space=U, hidden=(50, 50, 20))
# model_u = tp.models.QRES(input_space=X, output_space=U, hidden=(36, 36, 15))
model_a = tp.models.FCN(input_space=X, output_space=A, hidden=(30, 40, 40, 20, 20))
# model_a = tp.models.QRES(input_space=X, output_space=A, hidden=(25, 35, 25, 15, 15))
parallel_model = tp.models.Parallel(model_u, model_a)
pytorch_total_params = sum(p.numel() for p in model_u.parameters() if p.requires_grad)
print(pytorch_total_params)


scaling = 100
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx), X)
data = scaling * torch.tensor(np.load('darcy_data_f_u.npy').astype(np.float32))
data = tp.spaces.Points(data[:, 1:], U)

def pde_residual(a, u, x):
    return tp.utils.div(a * tp.utils.grad(u, x), x) + 1.0

pde_sampler = tp.samplers.DataSampler(xx).make_static()

pde_cond = tp.conditions.PINNCondition(module=parallel_model, sampler=pde_sampler,
                                       residual_fn=pde_residual)

data_loader = tp.utils.PointsDataLoader((xx, data), batch_size=50000)
data_condition = tp.conditions.DataCondition(module=model_u,
                                             dataloader=data_loader,
                                             norm=2,
                                             use_full_dataset=True,
                                             weight=100)

optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)

solver = tp.solver.Solver([pde_cond, data_condition])
trainer = pl.Trainer(
                    # gpus=1,
                    accelerator="cuda" if device.type == "cuda" else None,
                    max_steps=5000,
                     logger=False,
                     benchmark=True,
                     # checkpoint_callback=False
)

trainer.fit(solver)


# image for the learned f:
from matplotlib import cm
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx), X)
f_out = model_a(xx).as_tensor * scaling
f_out = f_out.detach().reshape(1, Nx+1, Nx+1)
f_out = torch.clamp(f_out, 2.8)


# fig = plt.figure() #figsize=(10+4, 5)
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
# axs[0].set_xlabel('x')#, fontsize=16, labelpad=15)
# axs[0].set_ylabel('y')#, fontsize=16, labelpad=15)
axs[0].set_title(r"PINN f")
img = axs[0].imshow(f_out[0], extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
# plt.colorbar()
fig.colorbar(img, ax=axs[0], fraction=0.046, pad=0.04)

# np.save("pinn_a", f_out)


# true f:
# plt.figure()
true_data = torch.tensor(np.load('darcy_data_f_u.npy').astype(np.float32))
# axs[1].set_xlabel('x')#, fontsize=16, labelpad=15)
# axs[1].set_ylabel('y')#, fontsize=16, labelpad=15)
axs[1].set_title(r"True f")
img = axs[1].imshow(true_data[:, 0].reshape(Nx+1, Nx+1), extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
# plt.colorbar()
fig.colorbar(img, ax=axs[1], fraction=0.046, pad=0.04)



# error f:
# plt.figure()
# plt.xlabel('x')#, fontsize=16, labelpad=15)
# plt.ylabel('y')#, fontsize=16, labelpad=15)
axs[2].set_title(r"abs. error f")
diff = np.abs(f_out[0].numpy()-true_data[:, 0].numpy().reshape(Nx+1, Nx+1))
l2_rel = np.sqrt(np.sum(diff**2))/np.sqrt(sum(true_data[:, 0]**2))
img = axs[2].imshow(diff, extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
# plt.colorbar()
fig.colorbar(img, ax=axs[2], fraction=0.046, pad=0.04)
# print(l2_rel)
plt.savefig("./results/pinn_ori_darcy_fc.png")
plt.show()
