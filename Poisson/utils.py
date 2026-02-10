import os
# import torchphysics as tp
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
from sklearn.metrics import mean_squared_error
from torch.nn.modules import Module
from torch.nn import functional as F
from typing import Optional
from torch import Tensor
from torch.nn.utils import parametrize



def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility in convolutional layers
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility


def piecewise():
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate ground truth signal
    def generate_signal(N=1000):
        x = np.linspace(0, 1, N)
        signal = np.piecewise(x, [x < 0.4, (x >= 0.4) & (x < 0.7), x >= 0.7],
                              [lambda x: 2 * np.sin(50 * np.pi * x) + 25 * x,
                               lambda x: 2 * np.sin(50 * np.pi * x) + 25 * x + 50,
                               lambda x: 2 * np.sin(50 * np.pi * x) + 25 * x + 120])
        return signal

    # Add noise to the signal
    def add_noise(signal, noise_std=10):
        noise = np.random.normal(0, noise_std, signal.shape)
        return signal + noise

    # Generate and plot data
    N = 1000
    signal = generate_signal(N)
    noisy_signal = add_noise(signal)

    plt.figure(figsize=(10, 5))
    plt.plot(signal, label="Ground Truth")
    plt.plot(noisy_signal, label="Noisy Observation", alpha=0.6)
    plt.legend()
    plt.title("Signal Denoising")
    plt.show()

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


def get_poisson_data_sampling(device="cpu"):
    true_data = torch.tensor(np.load('poisson_data_f_u.npy').astype(np.float32))
    print(true_data.shape) #(513 * 513, 2)

    Nx, Ny = 512, 512

    f = true_data[:, [0]].reshape(-1, Nx+1)  # 把一行看成一个 sample
    u = true_data[:, [1]].reshape(-1, Nx+1)  # 把一行看成一个 sample

    X_train_tensor = torch.tensor(f, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(u, dtype=torch.float32).to(device)
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    print(X_train_tensor.shape, y_train_tensor.shape)

    batch_size = 20
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    test_loader, X_test_tensor, y_test_tensor = None, None, None

    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    Nx, Ny = 512, 512

    # true u:
    fig, axis = plt.subplots(1, 2, figsize=(8, 3))
    axis[0].set_xlabel('x')  # , fontsize=16, labelpad=15)
    axis[0].set_ylabel('y')  # , fontsize=16, labelpad=15)
    axis[0].set_title(r"True u")
    im1 = axis[0].imshow(true_data[:, 1].reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',aspect='auto')  # , vmin=0, vmax=1, )
    plt.colorbar(im1, ax=axis[0])

    # true f:
    axis[1].set_xlabel('x')  # , fontsize=16, labelpad=15)
    axis[1].set_ylabel('y')  # , fontsize=16, labelpad=15)
    axis[1].set_title(r"True f")
    im2 = axis[1].imshow(true_data[:, 0].reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',aspect='auto')  # , vmin=0, vmax=1, )
    plt.colorbar(im2, ax=axis[1])
    # plt.show()


    return train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def poisson_data():
    true_data = torch.tensor(np.load('poisson_data_f_u.npy').astype(np.float32))
    print(true_data.shape) #(513 * 513, 2)

    # X = tp.spaces.R2('x')  # input is 2D
    # U = tp.spaces.R1('u')  # output is 1D
    # xx = np.linspace(xmin, xmax, Nx + 1)
    # xx = np.array(np.meshgrid(xx, xx)).T.reshape(-1, 2).astype(np.float32)
    # xx = tp.spaces.Points(torch.tensor(xx), X)

    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    Nx, Ny = 512, 512

    # true u:
    fig, axis = plt.subplots(1,2, figsize=(8,3))
    axis[0].set_xlabel('x')  # , fontsize=16, labelpad=15)
    axis[0].set_ylabel('y')  # , fontsize=16, labelpad=15)
    axis[0].set_title(r"True u")
    im1 = axis[0].imshow(true_data[:, 1].reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',aspect='auto')  # , vmin=0, vmax=1, )
    plt.colorbar(im1, ax=axis[0])

    # true f:
    axis[1].set_xlabel('x')  # , fontsize=16, labelpad=15)
    axis[1].set_ylabel('y')  # , fontsize=16, labelpad=15)
    axis[1].set_title(r"True f")
    im2 = axis[1].imshow(true_data[:, 0].reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',aspect='auto')  # , vmin=0, vmax=1, )
    plt.colorbar(im2, ax=axis[1])
    # plt.show()

if __name__ == "__main__":
    # piecewise()
    # poisson_data()
    # get_poisson_data_sampling()
    pass