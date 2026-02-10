
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto
import torch
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.modules import Module
from torch.nn import functional as F
from typing import Optional
import argparse
import os


def visualize_results(model, test_loader, device, num_samples=3, save_path="./res.png"):
    model.eval()
    samples_visualized = 0
    plt.figure(figsize=(12, num_samples * 3))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass: input -> inverted
            outputs = model(inputs)

            # Inverse pass: inverted -> input
            reconstructed_inputs = model.inverse(outputs)

            # Inverse pass: input -> some output
            inverse_of_y = model.inverse(targets)

            for i in range(inputs.size(0)):
                if samples_visualized >= num_samples:
                    break

                # Original Input x
                original = inputs[i].cpu().numpy().reshape(28, 28)

                # target y 
                target = targets[i].cpu().numpy().reshape(28, 28)

                # Forward Output  f(x)
                forward_output = outputs[i].cpu().numpy().reshape(28, 28)

                # Inverse of f(x)
                inverse_forward = reconstructed_inputs[i].cpu().numpy().reshape(28, 28)

                # Inverse of y
                inverse_input = inverse_of_y[i].cpu().numpy().reshape(28, 28)

                # Plotting
                titles = ['Input x', "Target y", 'model f(x)', 'Inverse of f(x)', 'Inverse of y']
                images = [original, target, forward_output, inverse_forward, inverse_input]

                for j in range(5):
                    plt.subplot(num_samples, 5, samples_visualized * 5 + j + 1)
                    plt.imshow(images[j], cmap='gray')
                    plt.title(titles[j])
                    plt.axis('off')

                samples_visualized += 1

            if samples_visualized >= num_samples:
                break

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()