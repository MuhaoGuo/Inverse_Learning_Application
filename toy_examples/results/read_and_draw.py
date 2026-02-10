
import json
import numpy as np
import matplotlib.pyplot as plt


def visualize(data_name, model_name, json_file_path):
    print(json_file_path)

    with open(json_file_path, 'r') as json_file:
        results = json.load(json_file)

    t = np.array(results["t"])
    X = np.array(results["X"])
    y = np.array(results["y"])
    y_pred = np.array(results["y_pred"])
    x_inv = np.array(results["x_inv"])
    x_inv_real = np.array(results["x_inv_real"])


    input_dim = 2
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(input_dim):
        axs[0].scatter(t, y_pred[:, i], alpha=0.8, s=10, label='Predict y {}'.format(i))
        axs[0].plot(t, y[:, i], alpha=0.8, label='True y {}'.format(i))
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y')
    axs[0].set_title('DipDNN: Followed Pass')
    axs[0].legend()

    for i in range(input_dim):
        axs[1].scatter(t, X[:, i], alpha=1, label='True X {}'.format(i), c="black")
        axs[1].scatter(t, x_inv[:, i], alpha=0.8, s=10, label='Reconstruct X {}'.format(i))
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('X value')
    axs[1].set_title('DipDNN: Reconstructed X vs X')
    axs[1].legend()

    for i in range(input_dim):
        axs[2].scatter(t, X[:, i], alpha=1, label='True X {}'.format(i), c="black")
        axs[2].scatter(t, x_inv_real[:, i], alpha=0.8, s=10, label='Reconstruct X from y{}'.format(i))
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('X value')
    axs[2].set_title('DipDNN: Reconstructed X (from y) vs X')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"./results/{data_name}_{model_name}.png")
    plt.show()


data_name = "bad1"
# data_name = "bad2"

model_name = "DipDNN"
# model_name = "iResNet"
# model_name = "ResNet"
# model_name = "NICE"

json_file_path = f"./results/{data_name}_{model_name}_results.json"

visualize(data_name, model_name, json_file_path)

