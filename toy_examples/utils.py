import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.autograd import grad

def estimate_lipschitz(model, dataloader, num_samples=1000):
    max_grad_norm = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        if i >= num_samples:
            break

        inputs = inputs.requires_grad_(True)  # Enable gradient computation for inputs
        z, log_det_jacobian = model(inputs)

        # Compute the gradient of the output w.r.t. the input
        loss = z.norm()
        gradients = grad(outputs=loss, inputs=inputs, create_graph=True)[0]

        # Compute the norm of the gradients
        grad_norm = gradients.norm()

        # Update the maximum gradient norm
        if grad_norm.item() > max_grad_norm:
            max_grad_norm = grad_norm.item()

        # print(f"Sample {i + 1}/{num_samples}: Grad Norm = {grad_norm.item()}")

    return max_grad_norm



def calculate_mse_per_column(X, x_reconstruct):
    # Step 1: Compute the element-wise differences
    differences = X - x_reconstruct

    # Step 2: Square the differences
    squared_differences = np.square(differences)

    # Step 3: Compute the mean of the squared differences for each column
    mse_per_column = np.mean(squared_differences, axis=0)

    return mse_per_column



def generate_synthetic_data():
    # Generate time series data
    x_values = np.linspace(-10, 10, 1000).astype(np.float32)
    y_values = (x_values ** 2).astype(np.float32)
    data = np.column_stack((x_values, y_values))  # Shape (1000, 2)
    print(data.shape)  # Should be (1000, 2)
    return data


def generate_synthetic_data_gaussian(mean = 0, std_dev = 1):
    '''
    x follows a normal distribution, and y = f(x)
    '''
    x_values = np.random.normal(mean, std_dev, 1000).astype(np.float32)
    y_values = (x_values ** 2).astype(np.float32)
    data = np.column_stack((x_values, y_values))  # Shape (1000, 2)
    # print(data.shape)  # Should be (1000, 2)
    x_values = x_values.reshape(-1, 1)
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)

    return x_values, y_values



def normalize_data(X, y):
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True)

    X_normalized = (X - X_mean) / X_std
    y_normalized = (y - y_mean) / y_std

    return X_normalized, y_normalized


def maxmin_normalize_data(X, y):
    min_vals, _ = torch.min(X, dim=0, keepdim=True)
    max_vals, _ = torch.max(X, dim=0, keepdim=True)
    X_normalized = (X - min_vals) / (max_vals - min_vals)

    min_vals, _ = torch.min(y, dim=0, keepdim=True)
    max_vals, _ = torch.max(y, dim=0, keepdim=True)
    y_normalized = (y - min_vals) / (max_vals - min_vals)

    return X_normalized, y_normalized


def deterministic_data_case1():
    np.random.seed(7)
    x_values0 = np.linspace(-1, 1, 100).astype(np.float32)
    x_values1 = np.random.normal(0, 1, 100).astype(np.float32)
    x_values1.sort()
    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)

    y_values0 = x_values0 + x_values1
    y_values1 = x_values0 - x_values1

    X = torch.concat((x_values0, x_values1), dim=1)
    y = torch.concat((y_values0, y_values1), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values1, y_values0, label="X1 Y0")
    plt.scatter(x_values1, y_values1, label="X1 Y1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t
def deterministic_data_case2():
    np.random.seed(7)
    x_values0 = np.linspace(-1, 1, 100).astype(np.float32)
    x_values1 = np.random.normal(0, 1, 100).astype(np.float32)
    x_values1.sort()
    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)

    y_values0 = x_values0**2 + x_values1
    y_values1 = x_values0**2 - x_values1

    X = torch.concat((x_values0, x_values1), dim=1)
    y = torch.concat((y_values0, y_values1), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values1, y_values0, label="X1 Y0")
    plt.scatter(x_values1, y_values1, label="X1 Y1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t




def deterministic_data_case3():
    np.random.seed(7)
    x_values0 = np.linspace(-1, 1, 100).astype(np.float32)
    x_values1 = np.random.normal(0, 1, 100).astype(np.float32)
    x_values1.sort()
    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)

    y_values0 = x_values0**3 + x_values1 **2 + x_values0
    y_values1 = x_values0**3 - x_values1 **2 - x_values0

    X = torch.concat((x_values0, x_values1), dim=1)
    y = torch.concat((y_values0, y_values1), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values1, y_values0, label="X1 Y0")
    plt.scatter(x_values1, y_values1, label="X1 Y1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()
    return X, y, t


def deterministic_data_high_dim_case3(dims = 2):
    np.random.seed(7)

    x_values = []
    for i in range(dims):
        # x_valuesi = np.linspace(-1, 1, 100).astype(np.float32)
        x_valuesi = np.random.normal(0, 1, 100).astype(np.float32)
        x_valuesi.sort()
        x_valuesi = torch.tensor(x_valuesi).view(-1, 1)
        x_values.append(x_valuesi)

    t = np.array(list(range(0, len(x_valuesi), 1)))

    X = torch.concat((x_values), dim=1)

    y_values = []
    for i in range(dims):
        y_valuesi = X[:, [i]] ** 3 - X[:, [i]] ** 2 + X[:, [i]]
        y_values.append(y_valuesi)

    y = torch.concat((y_values), dim=1)
    # print(X.shape, y.shape)
    # print(y)
    X, y = maxmin_normalize_data(X, y)
    # print(y)

    plt.figure()
    for i in range(dims):
        plt.scatter(t, y[:, i], label="t y{}".format(i), s = 10)
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case4_1():
    np.random.seed(7)

    # index = np.linspace(-1, 1, 100).astype(np.float32)
    # x_values0 = index
    # x_values1 = index**2
    # x_values2 = np.sin(index)
    # x_values3 = np.cos(index)

    n_samples = 100
    # # Generate i.i.d. samples from a normal distribution
    # x_values0 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    # x_values1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    # x_values2 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    # x_values3 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    #
    # Generate i.i.d. samples from a normal distribution and take absolute value to ensure positivity
    # x_values0 = np.abs(np.random.normal(loc=0.0, scale=1.0, size=n_samples)).astype(np.float32)
    # x_values1 = np.abs(np.random.normal(loc=0.0, scale=1.0, size=n_samples)).astype(np.float32)
    # x_values2 = np.abs(np.random.normal(loc=0.0, scale=1.0, size=n_samples)).astype(np.float32)
    # x_values3 = np.abs(np.random.normal(loc=0.0, scale=1.0, size=n_samples)).astype(np.float32)

    # # Generate i.i.d. samples from a normal distribution and take absolute value to ensure positivity
    # x_values0 = np.abs(np.random.normal(loc=0.0, scale=1.0, size=n_samples)).astype(np.float32)
    # x_values1 = -np.abs(np.random.normal(loc=0.0, scale=1.0, size=n_samples)).astype(np.float32)
    # x_values2 = np.abs(np.random.normal(loc=0.0, scale=1.0, size=n_samples)).astype(np.float32)
    # x_values3 = -np.abs(np.random.normal(loc=0.0, scale=1.0, size=n_samples)).astype(np.float32)

    # n_samples = 100
    # Generate i.i.d. samples from a uniform distribution
    x_values0 = np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))
    x_values1 = -np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))
    x_values2 = np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))
    x_values3 = -np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))

    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values0 + x_values0**3
    y_values1 = x_values1 + x_values1**3
    y_values2 = x_values2 + x_values2**3
    y_values3 = x_values3 + x_values3**3

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("x vs y", fontsize=16)
    # Plot x vs y for each combination
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.scatter(X[:, i].numpy(), y[:, j].numpy(), s=10, alpha=0.6, label=f"x{i} vs y{j}")
            ax.set_xlabel(f"x{i}")
            ax.set_ylabel(f"y{j}")
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])


    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t

def deterministic_data_case4_2():
    np.random.seed(7)

    # index = np.linspace(-1, 1, 100).astype(np.float32)
    # x_values0 = index
    # x_values1 = index**2
    # x_values2 = np.sin(index)
    # x_values3 = np.cos(index)

    n_samples = 100
    # Generate i.i.d. samples from a normal distribution
    # x_values0 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    # x_values1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    # x_values2 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    # x_values3 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    #
    x_values0 = np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))
    x_values1 = -np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))
    x_values2 = np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))
    x_values3 = -np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))

    #

    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = 0.5 * (x_values0 + x_values1) + x_values0 * x_values1 #y0  = 0.5(x0 + x1) + x0x1,
    y_values1 = 0.5 * (x_values0 - x_values1) + x_values0 * x_values1  # y1  = 0.5(x0 - x1) + x0x1,
    y_values2 = 0.5 * (x_values2 + x_values3) + x_values2 * x_values3  # y2  = 0.5(x2 + x3) + x2x3,
    y_values3 = 0.5 * (x_values2 - x_values3) + x_values2 * x_values3 # y3  = 0.5(x2 – x3) + x2x3,

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case4_3():
    np.random.seed(7)

    # index = np.linspace(-1, 1, 100).astype(np.float32)
    # x_values0 = index
    # x_values1 = index**2
    # x_values2 = np.sin(index)
    # x_values3 = np.cos(index)

    n_samples = 100
    # Generate i.i.d. samples from a normal distribution
    # x_values0 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    # x_values1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    # x_values2 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    # x_values3 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    #
    x_values0 = np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))
    x_values1 = -np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))
    x_values2 = np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))
    x_values3 = -np.abs(np.random.uniform(low=-1.0, high=1.0, size=n_samples).astype(np.float32))


    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values3 + x_values0 * x_values3  + x_values1 * x_values2 # y0  = x3 + x0x3 + x1x2,
    y_values1 = x_values2 + x_values1 * x_values3  + x_values0 * x_values2 # y1  = x2 + x1x3 + x0x2;
    y_values2 = x_values0 + x_values1 * x_values2**2  + x_values0 * x_values3 ** 2 # y2  = x0 + x1x2^2 + x0x3^2,
    y_values3 = x_values1 + x_values1 * x_values3**2  + x_values0 * x_values2 ** 2  # y3  = x1 + x1x3^2 + x0x2^2

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("x vs y", fontsize=16)
    # Plot x vs y for each combination
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.scatter(X[:, i].numpy(), y[:, j].numpy(), s=10, alpha=0.6, label=f"x{i} vs y{j}")
            ax.set_xlabel(f"x{i}")
            ax.set_ylabel(f"y{j}")
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t

def deterministic_data_case4():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    x_values2 = np.sin(index)
    x_values3 = np.cos(index)


    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values0 * x_values3 + x_values0**2  #y0 = x0x3 + x0^2
    y_values1 = x_values0 * x_values1 + x_values1**2  #y1 = x0x1 + x1^2
    y_values2 = x_values2 * x_values3 + x_values2**2  #y2 = x2x3 + x2^2
    y_values3 = x_values0 * x_values3 + x_values3**2  #y3 = x0x3 + x3^2

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("x vs y", fontsize=16)
    # Plot x vs y for each combination
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.scatter(X[:, i].numpy(), y[:, j].numpy(), s=10, alpha=0.6, label=f"x{i} vs y{j}")
            ax.set_xlabel(f"x{i}")
            ax.set_ylabel(f"y{j}")
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])


    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case4_iid():
    np.random.seed(7)

    # Number of samples
    n_samples = 1000
    # Generate i.i.d. samples from a normal distribution
    x_values0 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    x_values1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    x_values2 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    x_values3 = np.random.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)

    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values0 * x_values3 + x_values0**2  #y0 = x0x3 + x0^2
    y_values1 = x_values0 * x_values1 + x_values1**2  #y1 = x0x1 + x1^2
    y_values2 = x_values2 * x_values3 + x_values2**2  #y2 = x2x3 + x2^2
    y_values3 = x_values0 * x_values3 + x_values3**2  #y3 = x0x3 + x3^2

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x0 vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    # Create a 4x4 plot
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("x vs y", fontsize=16)
    # Plot x vs y for each combination
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.scatter(X[:, i].numpy(), y[:, j].numpy(), s=10, alpha=0.6, label=f"x{i} vs y{j}")
            ax.set_xlabel(f"x{i}")
            ax.set_ylabel(f"y{j}")
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()

    return X, y, t


def deterministic_data_case5():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    x_values2 = np.sin(index)
    x_values3 = np.cos(index)

    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values0 * x_values2 + x_values0**2  # x0x2 + x0^2
    y_values1 = x_values0 * x_values3 + x_values1**2  # x1x3 + x1^2
    y_values2 = x_values2 * x_values3 + x_values2**2  # x2x3 + x2^2
    y_values3 = x_values2 * x_values3 + x_values3**2  # x2x3 + x3^2

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t



def deterministic_data_case6():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    x_values2 = np.sin(index)
    x_values3 = np.cos(index)


    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values0 * x_values2 + x_values0**2  # 𝑥_0 𝑥_2+𝑥_0^2
    y_values1 = x_values1 * x_values3 + x_values1**2  # 𝑥_1 𝑥_3+𝑥_1^2
    y_values2 = x_values0 * x_values2 + x_values2**2  # 𝑥_0 𝑥_2+𝑥_2^2
    y_values3 = x_values1 * x_values3 + x_values3**2  # 𝑥_1 𝑥_3+𝑥_3^2

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case7():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    x_values2 = np.sin(index)
    x_values3 = np.cos(index)

    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values1 + x_values1**2  # y0 = 𝑥_1 + 𝑥_1^2
    y_values1 = x_values0 + x_values0**2  # y1 = 𝑥_0 + 𝑥_0^2
    y_values2 = x_values3 + x_values3**2  # y2 = 𝑥_3 + 𝑥_3^2
    y_values3 = x_values2 + x_values2**2  # y3 = 𝑥_2 + 𝑥_2^2

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case8():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    x_values2 = np.sin(index)
    x_values3 = np.cos(index)


    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values0 * x_values3 + x_values1 * x_values2 + x_values0**2  # 𝑥_0 𝑥_3+ 𝑥_1 𝑥_2 + x_0^2
    y_values1 = x_values1**2  # 𝑥_1^2
    y_values2 = x_values0 * x_values3 + x_values1 * x_values2 + x_values2**2  # 𝑥_0 𝑥_3+ 𝑥_1 𝑥_2 + x_2^2
    y_values3 = x_values3**2  # 𝑥_3^2

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case9():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    x_values2 = np.sin(index)
    x_values3 = np.cos(index)


    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values0 * x_values3 + x_values1 * x_values2 + x_values0**2  # 𝑥_0 𝑥_3+ 𝑥_1 𝑥_2+ 𝑥_0^2
    y_values1 = x_values0 * x_values3 + x_values1 * x_values2 + x_values1**2  # 𝑥_0 𝑥_3+ 𝑥_1 𝑥_2+ 𝑥_1^2
    y_values2 = x_values2**2  # x_2^2
    y_values3 = x_values3**2  # 𝑥_3^2

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case10():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    x_values2 = np.sin(index)
    x_values3 = np.cos(index)

    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values0 * x_values3 + x_values1 * x_values2 + x_values0**2 # 𝑥_0 𝑥_3+ 𝑥_1 𝑥_2 + 𝑥_0^2
    y_values1 = x_values0 * x_values3 + x_values1 * x_values2 + x_values1**2 # 𝑥_0 𝑥_3+ 𝑥_1 𝑥_2 + 𝑥_1^2
    y_values2 = x_values0 * x_values1 + x_values2 * x_values3 + x_values2**2 # 𝑥_0 𝑥_1+ 𝑥_2 𝑥_3 + 𝑥_2^2
    y_values3 = x_values0 * x_values1 + x_values2 * x_values3 + x_values3**2 # 𝑥_0 𝑥_1+ 𝑥_2 𝑥_3 + 𝑥_3^2

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case11():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    x_values2 = np.sin(index)
    x_values3 = np.cos(index)


    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = x_values0 * x_values1 + x_values0**2 # 𝑥_0 𝑥_1 + 𝑥_0^2
    y_values1 = x_values0 * x_values1 + x_values1**2 # 𝑥_0 𝑥_1 + 𝑥_1^2
    y_values2 = x_values2 * x_values3 + x_values2**2 # 𝑥_2 𝑥_3 + 𝑥_2^2
    y_values3 = x_values2 * x_values3 + x_values3**2 # 𝑥_2 𝑥_3 + 𝑥_3^2

    X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values0, y_values1, label="X0 Y1")
    plt.scatter(x_values0, y_values2, label="X0 Y2")
    plt.scatter(x_values0, y_values3, label="X0 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case12():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    x_values2 = np.sin(index)
    x_values3 = np.cos(index)
    x_values4 = np.tan(index)
    x_values5 = 2 * index
    x_values6 = index**3
    x_values7 = 0.1 * np.e ** index

    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)
    x_values4 = torch.tensor(x_values4).view(-1,1)
    x_values5 = torch.tensor(x_values5).view(-1,1)
    x_values6 = torch.tensor(x_values6).view(-1,1)
    x_values7 = torch.tensor(x_values7).view(-1,1)


    y_values0 = x_values0 * x_values2 * x_values4 * x_values6 + x_values1 * x_values3 * x_values5 * x_values7 + x_values0 ** 2 # 𝑥_0 𝑥_2 𝑥_4 𝑥_6 + 𝑥_1 𝑥_3 𝑥_5 𝑥_7 + x_0^2
    y_values1 = x_values0 * x_values2 * x_values4 * x_values6 + x_values1 * x_values3 * x_values5 * x_values7 + x_values1 ** 2 # 𝑥_0 𝑥_2 𝑥_4 𝑥_6 + 𝑥_1 𝑥_3 𝑥_5 𝑥_7 + x_1^2
    y_values2 = x_values2  # 𝑥_2
    y_values3 = x_values3  # 𝑥_3
    y_values4 = x_values0 * x_values2 * x_values4 * x_values6 + x_values1 * x_values3 * x_values5 * x_values7 + x_values4 ** 2  # 𝑥_0 𝑥_2 𝑥_4 𝑥_6 + 𝑥_1 𝑥_3 𝑥_5 𝑥_7 + x_4^2
    y_values5 = x_values0 * x_values2 * x_values4 * x_values6 + x_values1 * x_values3 * x_values5 * x_values7 + x_values5 ** 2  # 𝑥_0 𝑥_2 𝑥_4 𝑥_6 + 𝑥_1 𝑥_3 𝑥_5 𝑥_7 + x_5^2
    y_values6 = x_values6  # 𝑥_6
    y_values7 = x_values7  # 𝑥_7


    X = torch.concat((x_values0, x_values1, x_values2, x_values3, x_values4, x_values5, x_values6, x_values7), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3, y_values4, y_values5, y_values6, y_values7), dim=1)

    # print(X.shape, y.shape)

    # plt.figure()
    # plt.scatter(x_values0, y_values0, label="X0 Y0")
    # plt.scatter(x_values0, y_values1, label="X0 Y1")
    # plt.scatter(x_values0, y_values2, label="X0 Y2")
    # plt.scatter(x_values0, y_values3, label="X0 Y3")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("x vs y")
    # plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")
    plt.scatter(t, x_values4, label="t x4")
    plt.scatter(t, x_values5, label="t x5")
    plt.scatter(t, x_values6, label="t x6")
    plt.scatter(t, x_values7, label="t x7")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.scatter(t, y_values4, label="t y4")
    plt.scatter(t, y_values5, label="t y5")
    plt.scatter(t, y_values6, label="t y6")
    plt.scatter(t, y_values7, label="t y7")

    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case13():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    x_values2 = np.sin(index)
    x_values3 = np.cos(index)
    x_values4 = np.tan(index)
    x_values5 = 2 * index
    x_values6 = index**3
    x_values7 = 0.1 * np.e ** index

    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    x_values2 = torch.tensor(x_values2).view(-1,1)
    x_values3 = torch.tensor(x_values3).view(-1,1)
    x_values4 = torch.tensor(x_values4).view(-1,1)
    x_values5 = torch.tensor(x_values5).view(-1,1)
    x_values6 = torch.tensor(x_values6).view(-1,1)
    x_values7 = torch.tensor(x_values7).view(-1,1)


    y_values0 = x_values0 * x_values2 * x_values4 * x_values6 + x_values1 * x_values3 * x_values5 * x_values7 + x_values0 ** 2 # 𝑥_0 𝑥_2 𝑥_4 𝑥_6 + 𝑥_1 𝑥_3 𝑥_5 𝑥_7 + x_0^2
    y_values1 = x_values0 * x_values2 * x_values4 * x_values6 + x_values1 * x_values3 * x_values5 * x_values7 + x_values1 ** 2 # 𝑥_0 𝑥_2 𝑥_4 𝑥_6 + 𝑥_1 𝑥_3 𝑥_5 𝑥_7 + x_1^2
    y_values2 = x_values0 * x_values2 * x_values4 * x_values6 + x_values1 * x_values3 * x_values5 * x_values7 + x_values2 ** 2 # 𝑥_0 𝑥_2 𝑥_4 𝑥_6 + 𝑥_1 𝑥_3 𝑥_5 𝑥_7 + x_2^2
    y_values3 = x_values0 * x_values2 * x_values4 * x_values6 + x_values1 * x_values3 * x_values5 * x_values7 + x_values3 ** 2 # 𝑥_0 𝑥_2 𝑥_4 𝑥_6 + 𝑥_1 𝑥_3 𝑥_5 𝑥_7 + x_3^2
    y_values4 = x_values4  # 𝑥_4
    y_values5 = x_values5  # 𝑥_5
    y_values6 = x_values6  # 𝑥_6
    y_values7 = x_values7  # 𝑥_7


    X = torch.concat((x_values0, x_values1, x_values2, x_values3, x_values4, x_values5, x_values6, x_values7), dim=1)
    y = torch.concat((y_values0, y_values1, y_values2, y_values3, y_values4, y_values5, y_values6, y_values7), dim=1)

    # print(X.shape, y.shape)

    # plt.figure()
    # plt.scatter(x_values0, y_values0, label="X0 Y0")
    # plt.scatter(x_values0, y_values1, label="X0 Y1")
    # plt.scatter(x_values0, y_values2, label="X0 Y2")
    # plt.scatter(x_values0, y_values3, label="X0 Y3")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("x vs y")
    # plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    plt.scatter(t, x_values2, label="t x2")
    plt.scatter(t, x_values3, label="t x3")
    plt.scatter(t, x_values4, label="t x4")
    plt.scatter(t, x_values5, label="t x5")
    plt.scatter(t, x_values6, label="t x6")
    plt.scatter(t, x_values7, label="t x7")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    plt.scatter(t, y_values2, label="t y2")
    plt.scatter(t, y_values3, label="t y3")
    plt.scatter(t, y_values4, label="t y4")
    plt.scatter(t, y_values5, label="t y5")
    plt.scatter(t, y_values6, label="t y6")
    plt.scatter(t, y_values7, label="t y7")

    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t




def deterministic_data_case_bad1():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2
    # x_values2 = np.sin(index)
    # x_values3 = np.cos(index)


    t = np.array(list(range(0, len(x_values0), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)
    # x_values2 = torch.tensor(x_values2).view(-1,1)
    # x_values3 = torch.tensor(x_values3).view(-1,1)


    y_values0 = 2 * x_values0
    y_values1 = 2 * x_values1
    # y_values2 = 2 * x_values2
    # y_values3 = 2 * x_values3

    # X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    # y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)
    #
    X = torch.concat((x_values0, x_values1), dim=1)
    y = torch.concat((y_values0, y_values1), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values1, y_values1, label="X1 Y1")
    # plt.scatter(x_values2, y_values2, label="X2 Y2")
    # plt.scatter(x_values3, y_values3, label="X3 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    # plt.scatter(t, x_values2, label="t x2")
    # plt.scatter(t, x_values3, label="t x3")

    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    # plt.scatter(t, y_values2, label="t y2")
    # plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t


def deterministic_data_case_bad2():
    np.random.seed(7)

    index = np.linspace(-1, 1, 100).astype(np.float32)
    x_values0 = index
    x_values1 = index**2

    # x_values0 = - index**2   # todo


    t = np.array(list(range(0, len(x_values1), 1)))

    x_values0 = torch.tensor(x_values0).view(-1,1)
    x_values1 = torch.tensor(x_values1).view(-1,1)


    y_values0 = - x_values0
    y_values1 = - x_values1


    # X = torch.concat((x_values0, x_values1, x_values2, x_values3), dim=1)
    # y = torch.concat((y_values0, y_values1, y_values2, y_values3), dim=1)
    X = torch.concat((x_values0, x_values1), dim=1)
    y = torch.concat((y_values0, y_values1), dim=1)

    # print(X.shape, y.shape)

    plt.figure()
    plt.scatter(x_values0, y_values0, label="X0 Y0")
    plt.scatter(x_values1, y_values1, label="X1 Y1")
    # plt.scatter(x_values2, y_values2, label="X2 Y2")
    # plt.scatter(x_values3, y_values3, label="X3 Y3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y")
    plt.legend()

    plt.figure()
    plt.scatter(t, x_values0, label="t x0")
    plt.scatter(t, x_values1, label="t x1")
    # plt.scatter(t, x_values2, label="t x2")
    # plt.scatter(t, x_values3, label="t x3")
    plt.scatter(t, y_values0, label="t y0")
    plt.scatter(t, y_values1, label="t y1")
    # plt.scatter(t, y_values2, label="t y2")
    # plt.scatter(t, y_values3, label="t y3")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title("index vs values")
    # plt.show()

    return X, y, t

if __name__ == "__main__":
    # deterministic_data_case1()
    # deterministic_data_case4()
    # deterministic_data_high_dim_case3()
    # deterministic_data_case13()
    # deterministic_data_case_bad1()
    # deterministic_data_case_bad2()
    # deterministic_data_case4_iid()
    deterministic_data_case4_1()
    # deterministic_data_case4_2()
    # deterministic_data_case4_3()
    plt.show()
    pass
