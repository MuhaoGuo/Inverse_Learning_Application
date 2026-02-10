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
def get_InvertedPendulum_data(device="cpu"):
    data = pd.read_csv("results/InvertedPendulum_truth.csv")

    action = data["action"]
    action = action.apply(lambda x: eval(x))
    action = np.array(list(action.values))

    Force = action[:, [0]]
    zeros = np.zeros(Force.shape)

    observation = data["observation"]
    observation = observation.apply(lambda x: eval(x))
    observation = np.array(list(observation.values))

    pos = observation[:, [0]]
    angle = observation[:, [1]]
    velocity = observation[:, [2]]
    angle_velocity = observation[:, [3]]

    Force = np.concatenate((Force, zeros, zeros, zeros), axis=1)
    pos = np.concatenate((pos, zeros, zeros, zeros), axis=1)

    X_train_tensor = torch.tensor(Force, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(pos, dtype=torch.float32).to(device)
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    batch_size = 20
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    test_loader, X_test_tensor, y_test_tensor = None, None, None
    return train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


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
    data = pd.read_csv("results/InvertedPendulum_truth.csv")

    observation = data["observation"]
    observation = observation.apply(lambda x: eval(x))
    observation = np.array(list(observation.values))
    print(observation.shape)

    action = data["action"]
    action = action.apply(lambda x: eval(x))
    action = np.array(list(action.values))
    print(action.shape)

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    axes.plot(data["step"], action[:, 0])

    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    axes[0].plot(data["step"], observation[:, 0])
    axes[1].plot(data["step"], observation[:, 1])
    axes[2].plot(data["step"], observation[:, 2])
    axes[3].plot(data["step"], observation[:, 3])
    plt.show()

def generate_data_vis():
    data1 = pd.read_csv("results/InvertedPendulum_truth.csv")
    data2 = pd.read_csv("results/InvertedPendulum_NICE_real_inv.csv")
    data3 = pd.read_csv("results/InvertedPendulum_DipDNN_real_inv.csv")
    data4 = pd.read_csv("results/InvertedPendulum_iResNet_real_inv.csv")

    action1 = data1["action"]
    action1 = action1.apply(lambda x: eval(x))
    action1 = np.array(list(action1.values))

    action2 = data2["action"]
    action2 = action2.apply(lambda x: eval(x))
    action2 = np.array(list(action2.values))

    action3 = data3["action"]
    action3 = action3.apply(lambda x: eval(x))
    action3 = np.array(list(action3.values))

    action4 = data4["action"]
    action4 = action4.apply(lambda x: eval(x))
    action4 = np.array(list(action4.values))

    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    axes[0].plot(data1["step"], action1[:, 0], label= "Truth action")
    axes[1].plot(data2["step"], action2[:, 0], label= "NICE action")
    axes[2].plot(data3["step"], action3[:, 0], label="DipDNN action")
    axes[3].plot(data4["step"], action4[:, 0], label="iResNet action")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[3].legend()
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
    # get_InvertedPendulum_data()
    # raw_data_vis()
    # generate_data_vis()

    output_dir = "./results"       # Where to save extracted frames

    gif_path = "results/InvertedPendulum_DipDNN_real_inv_behavior.gif"  # Path to your GIF
    combined_png = "InvertedPendulum_DipDNN_real_inv_behavior.png"  # Name of the combined output

    gif_path = "results/InvertedPendulum_iResNet_real_inv_behavior.gif"  # Path to your GIF
    combined_png = "InvertedPendulum_iResNet_real_inv_behavior.png"  # Name of the combined output
    #
    gif_path = "results/InvertedPendulum_NICE_real_inv_behavior.gif"  # Path to your GIF
    combined_png = "InvertedPendulum_NICE_real_inv_behavior.png"  # Name of the combined output

    gif_path = "results/InvertedPendulum_ResNet_real_inv_behavior.gif"  # Path to your GIF
    combined_png = "InvertedPendulum_ResNet_real_inv_behavior.png"  # Name of the combined output
    #
    gif_path = "results/InvertedPendulum_truth.gif"  # Path to your GIF
    combined_png = "InvertedPendulum_truth.png"  # Name of the combined output

    gif_to_row_of_images(gif_path, output_dir, combined_png, num_frames=5)
    pass