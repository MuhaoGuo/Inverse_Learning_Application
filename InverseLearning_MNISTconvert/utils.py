# from MNIST_negative_data import *
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
# from utils import *
from enum import Enum, auto
import torch
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.modules import Module
from torch.nn import functional as F
from typing import Optional
import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def invert_colors(image):
    """Invert black/white (color inversion)"""
    return transforms.functional.invert(image)


def generate_dataset(transform_type='invert_colors', data=None, direction='horizontal', noise_level=20):
    """Generate datasets with chosen transformation applied"""
    transform_functions = {
        'invert_colors': invert_colors,
    }
    transform_fn = transform_functions.get(transform_type)

    if transform_fn is None:
        raise ValueError(f"Invalid transform type: {transform_type}")

    # Apply transformation to the entire dataset
    transformed_dataset = [(image, transform_fn(image)) for image, label in data]

    return transformed_dataset


# Example to use dataset as NN input-output pairs
def prepare_for_nn(dataset):
    """Prepare the dataset in the format required for NN (inputs and outputs as tensors)"""
    inputs = []
    outputs = []
    for original_image, transformed_image in dataset:
        input_tensor = transforms.ToTensor()(original_image)  # Convert to tensor
        output_tensor = transforms.ToTensor()(transformed_image)
        inputs.append(input_tensor)
        outputs.append(output_tensor)

    inputs = torch.stack(inputs)  # Stack into batch
    outputs = torch.stack(outputs)
    return inputs, outputs


# Visualize some examples from the generated dataset
def visualize_transformed_examples(dataset, num_examples=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_examples):
        original_image, transformed_image = dataset[i]
        plt.subplot(2, num_examples, i+1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, num_examples, i+1+num_examples)
        plt.imshow(transformed_image, cmap='gray')
        plt.title('Transformed')
        plt.axis('off')
        plt.savefig("./results/raw.png")
    # plt.show()



def load_MNIST_convert_data(batch_size = 128, samples= 30):

    # Load MNIST dataset
    data_path = '../data/'
    mnist_data_train = torchvision.datasets.MNIST(root=data_path, train=True, download=False, )
    mnist_data_test = torchvision.datasets.MNIST(root=data_path, train=False, download=False, )

    ##################################################
    # section: Create a subset of the dataset
    from torch.utils.data import Subset
    subset_indices = list(range(samples))
    mnist_data_train = Subset(mnist_data_train, subset_indices)
    mnist_data_test = Subset(mnist_data_test, subset_indices)
    ##################################################

    # Example: Generate dataset with color inversion
    transformed_dataset_train = generate_dataset(transform_type='invert_colors', data=mnist_data_train)
    transformed_dataset_test = generate_dataset(transform_type='invert_colors', data=mnist_data_test)

    # Visualize the transformed examples (color inversion in this case)
    visualize_transformed_examples(transformed_dataset_train)
    visualize_transformed_examples(transformed_dataset_test)

    # Prepare the dataset for training the neural network
    inputs_train, outputs_train = prepare_for_nn(transformed_dataset_train)
    inputs_test, outputs_test = prepare_for_nn(transformed_dataset_test)

    # inputs and outputs are now ready to be fed into a neural network
    print(f"inputs_train: {inputs_train.shape}")
    print(f"outputs_train: {outputs_train.shape}")

    print(f"inputs_test: {inputs_test.shape}")
    print(f"outputs_test: {outputs_test.shape}")

    train_tensor = TensorDataset(inputs_train, outputs_train)
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False)

    test_tensor = TensorDataset(inputs_test, outputs_test)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



def load_FashionMNIST_convert_data(batch_size = 128, samples= 30):

    # Load MNIST dataset
    data_path = '../data/'
    mnist_data_train = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, )
    mnist_data_test = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, )

    ##################################################
    # section: Create a subset of the dataset
    from torch.utils.data import Subset
    subset_indices = list(range(samples))
    mnist_data_train = Subset(mnist_data_train, subset_indices)
    mnist_data_test = Subset(mnist_data_test, subset_indices)
    ##################################################

    # Example: Generate dataset with color inversion
    transformed_dataset_train = generate_dataset(transform_type='invert_colors', data=mnist_data_train)
    transformed_dataset_test = generate_dataset(transform_type='invert_colors', data=mnist_data_test)

    # Visualize the transformed examples (color inversion in this case)
    visualize_transformed_examples(transformed_dataset_train)
    visualize_transformed_examples(transformed_dataset_test)

    # Prepare the dataset for training the neural network
    inputs_train, outputs_train = prepare_for_nn(transformed_dataset_train)
    inputs_test, outputs_test = prepare_for_nn(transformed_dataset_test)

    # inputs and outputs are now ready to be fed into a neural network
    print(f"inputs_train: {inputs_train.shape}")
    print(f"outputs_train: {outputs_train.shape}")

    print(f"inputs_test: {inputs_test.shape}")
    print(f"outputs_test: {outputs_test.shape}")

    # select_l = 28
    # inputs_train = inputs_train[:, :, 0:select_l, 0:select_l] #* 255
    # outputs_train = outputs_train[:, :, 0:select_l, 0:select_l] #* 255
    # inputs_test = inputs_test[:, :, 0:select_l, 0:select_l] #* 255
    # outputs_test = outputs_test[:, :, 0:select_l, 0:select_l] #* 255


    # inputs_train = 0.2 + inputs_train * (0.8 - 0.2)
    # outputs_train = 0.2 + outputs_train * (0.8 - 0.2)
    # inputs_test = 0.2 + inputs_test * (0.8 - 0.2)
    # outputs_test = 0.2 + outputs_test * (0.8 - 0.2)

    # print( inputs_train[2, 0, 0:select_l, 0:select_l])
    # print( outputs_train[2, 0, 0:select_l, 0:select_l])
    # exit()

    train_tensor = TensorDataset(inputs_train, outputs_train)
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False)

    test_tensor = TensorDataset(inputs_test, outputs_test)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def visualize_results(model, test_loader, device, num_samples=3, save_path="./res.png"):
    model.eval()
    image_l = 28
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
                original = inputs[i].cpu().numpy().reshape(image_l, image_l)

                # target y 
                target = targets[i].cpu().numpy().reshape(image_l, image_l)

                # Forward Output  f(x)
                forward_output = outputs[i].cpu().numpy().reshape(image_l, image_l)

                # Inverse of f(x)
                inverse_forward = reconstructed_inputs[i].cpu().numpy().reshape(image_l, image_l)

                # Inverse of y
                inverse_input = inverse_of_y[i].cpu().numpy().reshape(image_l, image_l)

                # Plotting
                titles = ['Input x', "Target y", 'model f(x)', 'Inverse of f(x)', 'Inverse of y']
                images = [original, target, forward_output, inverse_forward, inverse_input]

                # print(original)
                # print(target)

                for j in range(5):
                    plt.subplot(num_samples, 5, samples_visualized * 5 + j + 1)
                    plt.imshow(images[j], cmap='gray', vmin=0, vmax=1)
                    plt.title(titles[j])
                    plt.axis('off')

                samples_visualized += 1

            if samples_visualized >= num_samples:
                break

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


    