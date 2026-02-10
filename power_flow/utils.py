
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json  




# Code cell
def preprocess_data(df):
    # Remove bus 1 data (load_P_bus_1, load_Q_bus_1, voltage_bus_1, theta_bus_1)
    df = df.drop(columns=[col for col in df.columns if col.endswith('bus_1')])

    # Input: load_P and load_Q from all buses except bus 1
    load_P_columns = [col for col in df.columns if 'load_P_bus' in col]
    load_Q_columns = [col for col in df.columns if 'load_Q_bus' in col]

    # Output: voltage and theta from all buses except bus 1
    voltage_columns = [col for col in df.columns if 'voltage_bus' in col]
    theta_columns = [col for col in df.columns if 'theta_bus' in col]

    # Stack inputs (load_P, load_Q) and outputs (voltage, theta)
    # X = np.hstack([df[load_P_columns].values, df[load_Q_columns].values])
    # y = np.hstack([df[voltage_columns].values, df[theta_columns].values])

    X = np.hstack([df[load_Q_columns].values])
    y = np.hstack([df[voltage_columns].values])


    X = X[::10, :] 
    y = y[::10, :]

    # X = X[:, [0,2]]
    # y = y[:, [0,2]]
    print(X.shape, y.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Subplot 1: Input data x
    axes[0].plot(X, label='X', alpha=0.6)
    axes[0].set_title('Input Data x')
    axes[0].legend()

    axes[1].plot(y, label='y', alpha=0.6)
    axes[1].set_title('Output Data y')
    axes[1].legend()
    
    plt.savefig("./results/powerflow_raw_data.png")

    return X, y




def visualize(json_file_path, model_name):

    with open(json_file_path, 'r') as json_file:
        results = json.load(json_file)

    # Extract data
    x_test_original = np.array(results["x_test_original"])
    y_test_original = np.array(results["y_test_original"])
    y_pred_original = np.array(results["y_pred"])
    x_inv_original = np.array(results["x_inv"])
    x_inv_real_original = np.array(results["x_inv_real"])
    mse = results["MSE"]
    
    # Visualization with 5 Subplots (Modified)
    # Select a subset of data for visualization (e.g., first 100 samples)
    num_samples = len(x_test_original)  #100
    indices = np.arange(num_samples)

    # Prepare data for plotting
    x_plot = x_test_original[:num_samples]
    y_plot = y_test_original[:num_samples]
    y_pred_plot = y_pred_original[:num_samples]
    x_inv_plot = x_inv_original[:num_samples]
    x_inv_real_plot = x_inv_real_original[:num_samples]


    # Create a figure with 5 subplots in one row
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    # Subplot 1: Input data x
    axes[0].plot(indices, x_plot[:, 0], label='Q1', alpha=0.6)
    axes[0].plot(indices, x_plot[:, -1], label='Q2', alpha=0.6)
    axes[0].set_title('Input Data x')
    axes[0].legend()

    # Subplot 2: Target data y
    axes[1].plot(indices, y_plot[:, 0], label='voltage1', alpha=0.6)
    axes[1].plot(indices, y_plot[:, -1], label='voltage2', alpha=0.6)
    axes[1].set_title('Target Data y')
    axes[1].legend()

    # Subplot 3: Forward predicted data y_pred
    axes[2].plot(indices, y_pred_plot[:, 0], label='voltage_pred1', alpha=0.6)
    axes[2].plot(indices, y_pred_plot[:, -1], label='voltage_pred2', alpha=0.6)
    axes[2].set_title('Predicted Data y_pred')
    axes[2].legend()

    # Subplot 4: Inverse data x_inv from y_pred
    axes[3].plot(indices, x_inv_plot[:, 0], label='Q1_inv', alpha=0.6)
    axes[3].plot(indices, x_inv_plot[:, -1], label='Q2_inv', alpha=0.6)
    axes[3].set_title('Inverse Data x_inv from y_pred')
    axes[3].legend()

    # Subplot 5: Real inverse data x_inv_real from y
    axes[4].plot(indices, x_inv_real_plot[:, 0], label='Q1_inv_real', alpha=0.6)
    axes[4].plot(indices, x_inv_real_plot[:, -1], label='Q2_inv_real', alpha=0.6)
    axes[4].set_title('Real Inverse Data x_inv_real from y')
    axes[4].legend()

    # axes[5].plot(indices, y_pred_original[:, 0] - y_plot[:, 0], label='forward error', alpha=0.6)
    # axes[5].plot(indices, y_pred_original[:, 1] - y_plot[:, -1], label='forward error', alpha=0.6)
    # axes[5].set_title('error of y and f(x)')
    # axes[5].legend()

    plt.tight_layout()
    plt.savefig(f"./results/{model_name}_powerflow.png")
    plt.show()



if __name__ == "__main__":
    pass