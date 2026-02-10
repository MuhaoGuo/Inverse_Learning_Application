from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import torchphysics as tp
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
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

'''
X = tp.spaces.R2('x') # input is 2D
U = tp.spaces.R1('u') # output is 1D
square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])


model = tp.models.FCN(input_space=X, output_space=U, hidden=(50, 50, 50))

scaling = 10
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx), X)
# f = scaling * torch.tensor(np.load('poisson_data_f_u.npy').astype(np.float32))

data = np.load('poisson_data_f_u.npy').astype(np.float32)
f_data = data[:, 0:1]  # just the first column for f
u_data = data[:, 1:2]  # just the second column for u (not used for forward)
f = scaling * torch.tensor(f_data)

pde_sampler = tp.samplers.DataSampler(xx)


def pde_residual(u, x):
    return tp.utils.laplacian(u, x) + f
pde_cond = tp.conditions.PINNCondition(module=model, sampler=pde_sampler,
                                       residual_fn=pde_residual)

# boundary condition:
def bound_residual(u):
    return u
bound_sampler = tp.samplers.GridSampler(square.boundary, n_points=2052).make_static()
bound_cond = tp.conditions.PINNCondition(module=model, sampler=bound_sampler,
                                         residual_fn=bound_residual, weight=10000)




# here we start with Adam:
optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)
solver = tp.solver.Solver(train_conditions=[pde_cond, bound_cond], optimizer_setting=optim)



device = get_device()
f = f.to(device)
trainer = pl.Trainer(
                     accelerator="cuda", # if device.type == "mps" else None,
                     max_steps=1,# 5000,
                     logger=False,
                     benchmark=True,
                     # checkpoint_callback=False
)
trainer.fit(solver)


# first move the GPU:
xx = xx.to(device)
model.to(device)
start = time.time()
#u_out = u_constrain(model(xx).as_tensor, xx.as_tensor)
u_out = model(xx)
print('testing time:', time.time() - start)
u_out = (u_out.as_tensor/ scaling).detach().cpu().numpy().reshape(1, Nx+1, Nx+1)
fig = plt.figure() #figsize=(10+4, 5)
plt.xlabel('x')#, fontsize=16, labelpad=15)
plt.ylabel('y')#, fontsize=16, labelpad=15)
plt.title(r"PINN u")
plt.imshow(u_out[0], extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
plt.colorbar()
plt.savefig("./results/pinn_u.png")
'''

##################################################################################################################
##################################################################################################################
# inverse
##################################################################################################################
##################################################################################################################
import torch
import numpy as np

X = tp.spaces.R2('x')
U = tp.spaces.R1('u')
F = tp.spaces.R1('f')
square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])

model_u = tp.models.QRES(input_space=X, output_space=U, hidden=(36, 36, 36))
model_f = tp.models.QRES(input_space=X, output_space=F, hidden=(36, 36, 36))
parallel_model = tp.models.Parallel(model_u, model_f)

scaling = 100
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx), X)
data = scaling * torch.tensor(np.load('poisson_data_f_u.npy').astype(np.float32))
data = tp.spaces.Points(data[:, 1:], U)

def pde_residual(x, f, u):
    return tp.utils.laplacian(u, x) + f
pde_sampler = tp.samplers.DataSampler(xx).make_static()
pde_cond = tp.conditions.PINNCondition(module=parallel_model, sampler=pde_sampler,
                                       residual_fn=pde_residual)


data_loader = tp.utils.PointsDataLoader((xx, data), batch_size=50000)
data_condition = tp.conditions.DataCondition(module=model_u,
                                             dataloader=data_loader,
                                             norm=2,
                                             use_full_dataset=True,
                                             weight=10000)


solver = tp.solver.Solver([pde_cond, data_condition])
optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)  # LBFGS
trainer = pl.Trainer(
                    # gpus=1,
                    # accelerator="mps" if device.type == "mps" else None,
                    accelerator= "cuda",
                     max_steps=5000, #5000,
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
f_out = model_f(xx).as_tensor / scaling
f_out = f_out.detach().reshape(1, Nx+1, Nx+1)

u_out = model_u(xx).as_tensor / scaling
u_out = u_out.detach().reshape(1, Nx+1, Nx+1)

# fig = plt.figure() #figsize=(10+4, 5)
fig, axis = plt.subplots(1, 2, figsize =(8,3)) #figsize=(10+4, 5)
axis[0].set_title("Pred f (X)")
axis[0].imshow(f_out[0], extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )

axis[1].set_title("Pred u (y)")
axis[1].imshow(u_out[0], extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
# plt.colorbar()
plt.savefig("./results/pinn_f.png")
plt.show()


#
# fig = plt.figure() #figsize=(10+4, 5)
# plt.xlabel('x')#, fontsize=16, labelpad=15)
# plt.ylabel('y')#, fontsize=16, labelpad=15)
# plt.title(r"PINN f")
# plt.imshow(f_out[0], extent=[xmin, xmax, ymin, ymax], origin='lower', aspect = 'auto')#, vmin=0, vmax=1, )
# plt.colorbar()
# plt.savefig("./results/pinn_f.png")
# plt.show()
#
