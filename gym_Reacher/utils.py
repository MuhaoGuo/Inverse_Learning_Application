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
from PIL import Image
import os

def get_data(device):
    data = pd.read_csv("results/Truth_reacher_behavior.csv")

    action = data["action"]
    action = action.apply(lambda x: eval(x))
    action = np.array(list(action.values))

    Torque_1 = action[:, [0]]
    Torque_2 = action[:, [1]]

    observation = data["observation"]
    observation = observation.apply(lambda x: eval(x))
    observation = np.array(list(observation.values))

    cos1 = observation[:, [0]]
    cos2 = observation[:, [1]]
    sin1 = observation[:, [2]]
    sin2 = observation[:, [3]]
    angular_v1 = observation[:, [6]]
    angular_v2 = observation[:, [7]]
    dx = observation[:, [8]]
    dy = observation[:, [9]]

    # Torque = np.concatenate((Torque_1, Torque_2), axis=1)
    # Angular_v = np.concatenate((angular_v1, angular_v2), axis=1)

    Torque = Torque_1
    Angular_v = angular_v1

    X_train_tensor = torch.tensor(Torque, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(Angular_v, dtype=torch.float32).to(device)
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    batch_size = 100
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    test_loader, X_test_tensor, y_test_tensor = None, None, None
    return train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


    # X_train, X_test, y_train, y_test = train_test_split(Torque, Angular_v, test_size=0.8, random_state=42, shuffle=False)
    # X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    # y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    # X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # batch_size = 30

    # dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    # train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    # dataset_test = TensorDataset(X_test_tensor, y_test_tensor)
    # test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # return train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility in convolutional layers
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

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

def raw_data_vis():
    data = pd.read_csv("results/Truth_reacher_behavior.csv")

    observation = data["observation"]
    observation = observation.apply(lambda x: eval(x))
    observation = np.array(list(observation.values))
    print(observation.shape)

    action = data["action"]
    action = action.apply(lambda x: eval(x))
    action = np.array(list(action.values))
    print(action.shape)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(data["step"], action[:, 0])
    axes[1].plot(data["step"], action[:, 1])

    fig, axes = plt.subplots(10, 1, figsize=(10, 8))
    axes[0].plot(data["step"], observation[:, 0])
    axes[1].plot(data["step"], observation[:, 1])
    axes[2].plot(data["step"], observation[:, 2])
    axes[3].plot(data["step"], observation[:, 3])
    axes[4].plot(data["step"], observation[:, 4])
    axes[5].plot(data["step"], observation[:, 5])
    axes[6].plot(data["step"], observation[:, 6])
    axes[7].plot(data["step"], observation[:, 7])
    axes[8].plot(data["step"], observation[:, 8])
    axes[9].plot(data["step"], observation[:, 9])
    plt.show()

def generate_data_vis():
    data = pd.read_csv("results/Truth_reacher_behavior.csv")
    # data2 = pd.read_csv("Reacher_NICE_real_inv.csv")
    # data2 = pd.read_csv("Reacher_DipDNN_real_inv.csv")
    data2 = pd.read_csv("results/Reacher_iResNet_real_inv.csv")

    action = data["action"]
    action = action.apply(lambda x: eval(x))
    action = np.array(list(action.values))

    action2 = data2["action"]
    action2 = action2.apply(lambda x: eval(x))
    action2 = np.array(list(action2.values))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(data["step"], action[:, 0])
    axes[1].plot(data["step"], action[:, 1])
    plt.title("Raw action")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(data["step"], action2[:, 0])
    axes[1].plot(data["step"], action2[:, 1])
    plt.title("DipDNN inv action")

    plt.show()


def gif_to_row_of_images(gif_path, output_dir, combined_output="combined.png", num_frames=5):
    """
    1) Extracts `num_frames` frames at equally spaced intervals from a GIF and saves them as individual PNG files.
    2) Creates one combined image with these frames laid out in a single row.
    """
    # 1. Open the GIF and get the total number of frames
    gif = Image.open(gif_path)
    total_frames = gif.n_frames  # Total number of frames in the GIF

    # If the GIF has fewer frames than required, reduce num_frames accordingly
    num_frames = min(num_frames, total_frames)

    # Calculate the interval between frames
    interval = total_frames // num_frames

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    frame_indices = [i * interval for i in range(num_frames)]  # Indices of frames to extract

    for frame_index in frame_indices:
        try:
            gif.seek(frame_index)
            # Copy the current frame
            frame = gif.copy()
            # Convert to 'RGB' if it has an alpha channel or palette
            frame = frame.convert('RGB')

            frames.append(frame)
        except EOFError:
            break  # No more frames in GIF

    if not frames:
        print("No frames extracted, check the GIF file.")
        return

    # 2. Combine frames into a single horizontal image
    # Calculate total width and max height for the combined image
    total_width = sum(frame.width for frame in frames)
    max_height = max(frame.height for frame in frames)

    # Create a new blank image with enough width to hold all frames horizontally
    combined_image = Image.new("RGB", (total_width, max_height))

    # Paste frames side by side
    x_offset = 0
    for frame in frames:
        combined_image.paste(frame, (x_offset, 0))
        x_offset += frame.width

    # Save the combined row-of-frames image
    combined_path = os.path.join(output_dir, combined_output)
    combined_image.save(combined_path)
    print(f"Combined image saved to: {combined_path}")


if __name__ == "__main__":
    # raw_data_vis()
    # generate_data_vis()

    output_dir = "./results"       # Where to save extracted frames

    gif_path = "results/Reacher_DipDNN_real_inv_behavior.gif"  # Path to your GIF
    combined_png = "Reacher_DipDNN_real_inv_behavior.png"  # Name of the combined output

    gif_path = "results/Truth_reacher_behavior.gif"  # Path to your GIF
    combined_png = "Truth_reacher_behavior.png"  # Name of the combined output

    gif_path = "results/Reacher_NICE_real_inv_behavior.gif"  # Path to your GIF
    combined_png = "Reacher_NICE_real_inv_behavior.png"  # Name of the combined output

    gif_path = "results/Reacher_iResNet_real_inv_behavior.gif"  # Path to your GIF
    combined_png = "Reacher_iResNet_real_inv_behavior.png"  # Name of the combined output

    gif_path = "results/Reacher_ResNet_real_inv_behavior.gif"  # Path to your GIF
    combined_png = "Reacher_ResNet_real_inv_behavior.png"  # Name of the combined output

    gif_to_row_of_images(gif_path, output_dir, combined_png, num_frames=5)