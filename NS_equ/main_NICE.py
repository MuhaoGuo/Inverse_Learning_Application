import pandas as pd
import numpy as np
import glob
import re
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility in convolutional layers
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility


# Set the seed before initializing the model or any random operations
set_seed(42)  # You can use any seed value of your choice

datapath = "/Users/muhaoguo/Documents/study/Paper_Projects/Datasets/Data_Jiaqi/data"

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


file_pattern = '{}/chemical_*.csv'.format(datapath)
files = glob.glob(file_pattern)
files = sorted(files)

data_all = pd.DataFrame()
pattern = r'{}/chemical_(\d+)\.csv'.format(datapath)

for file in files:
    match = re.search(pattern, file)
    if match:
        time_step = int(match.group(1))
    else:
        raise ValueError('Missing time step: ' + file)
    data = pd.read_csv(file)
    data.drop(columns=['z', 'wz'], inplace=True)
    data['t'] = time_step
    data_all = pd.concat([data_all, data], ignore_index=True)


inputs = data_all[['x', 'y', 't', 'wx', 'wy']].values
targets = data_all[['uA', 'uB', 'uC', 'uA', 'uB']].values


inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)
dataset = TensorDataset(inputs, targets)

batch_size = 24460
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, process_first_part=True):
        """
        Affine Coupling Layer as per NICE architecture.

        Args:
            input_dim (int): Dimension of the input.
            process_first_part (bool): Whether to process the first or second part of the input.
        """
        super(AffineCouplingLayer, self).__init__()
        self.process_first_part = process_first_part
        self.input_dim = input_dim
        self.split_dim1 = input_dim // 2
        self.split_dim2 = input_dim - self.split_dim1

        if self.process_first_part:
            network_input_dim = self.split_dim1
            network_output_dim = self.split_dim2
        else:
            network_input_dim = self.split_dim2
            network_output_dim = self.split_dim1

        self.network = nn.Sequential(
            nn.Linear(network_input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, network_output_dim)
        )

    def forward(self, x):
        """
        Forward pass of the Affine Coupling Layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Output tensor after coupling.
        """
        x1, x2 = x.split([self.split_dim1, self.split_dim2], dim=1)
        if self.process_first_part:
            shift = self.network(x1)
            y1 = x1
            y2 = x2 + shift
        else:
            shift = self.network(x2)
            y1 = x1 + shift
            y2 = x2
        y = torch.cat((y1, y2), dim=1)
        return y

    def inverse(self, y):
        """
        Inverse pass of the Affine Coupling Layer.

        Args:
            y (Tensor): Output tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Reconstructed input tensor.
        """
        y1, y2 = y.split([self.split_dim1, self.split_dim2], dim=1)
        if self.process_first_part:
            shift = self.network(y1)
            x1 = y1
            x2 = y2 - shift
        else:
            shift = self.network(y2)
            x1 = y1 - shift
            x2 = y2
        x = torch.cat((x1, x2), dim=1)
        return x


class NICEResNet(nn.Module):
    def __init__(self, input_dim, num_blocks):
        """
        Modified NICE model inspired by ResNet architecture.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Number of hidden units in the coupling layers.
            num_blocks (int): Number of Affine Coupling Layers (blocks) in the model.
        """
        super(NICEResNet, self).__init__()
        self.input_dim = input_dim
        self.num_blocks = num_blocks

        # Initialize Affine Coupling Layers
        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            process_first_part = (i % 2 == 0)  # Alternate processing parts for each block
            self.layers.append(AffineCouplingLayer(input_dim, process_first_part))

        # Learnable scaling factor
        self.scaling_factor = nn.Parameter(torch.ones(1, input_dim))

    def forward(self, x):
        """
        Forward pass through the NICEResNet model.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Transformed output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        x = x * self.scaling_factor
        return x

    def inverse(self, y):
        """
        Inverse pass through the NICEResNet model.

        Args:
            y (Tensor): Output tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Reconstructed input tensor.
        """
        y = y / self.scaling_factor
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

    def count_parameters(self):
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_model(model, train_loader, optimizer, criterion, epochs, device=torch.device('cpu')):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # model.apply_weight_masks()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))
torch.manual_seed(1234)


input_dim = 5
num_blocks = 3
learning_rate = 1e-3
# num_epochs = 5000
epochs = 400
time = 60

device = get_device()
device = torch.device('cpu')
model_name = "NICE"

model = NICEResNet(input_dim=input_dim, num_blocks=num_blocks).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

### train model #########################
train_model(model, train_loader, optimizer, criterion, epochs, device)

model.eval()
with torch.no_grad():
    inverse_error = 0.0
    inverse_error_real = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass: input -> inverted
        outputs = model(inputs)

        # Inverse pass: f(x) -> input
        reconstructed_inputs = model.inverse(outputs)
        reconstructed_inputs_real = model.inverse(targets)
        # print("reconstructed_inputs", reconstructed_inputs)

        # Compute inverse error (MSE between original inputs and reconstructed inputs)
        loss = criterion(reconstructed_inputs, inputs)
        loss_real = criterion(reconstructed_inputs_real, inputs)

        inverse_error += loss.item() * inputs.size(0)
        inverse_error_real += loss_real.item() * inputs.size(0)

    avg_inverse_error = inverse_error / len(train_loader.dataset)
    avg_inverse_error_real = inverse_error_real / len(train_loader.dataset)
    print(f'Inverse Pass Error (MSE) after training: {avg_inverse_error}')
    print(f'Inverse Pass Error (MSE) after training real: {avg_inverse_error_real}')


### test model #########################

data_selected = data_all[data_all['t'] == time]
test_input = data_selected[['x', 'y', 't', 'wx', 'wy']].values
test_target = data_selected[['uA', 'uB', 'uC', 'uA', 'uB']].values

test_input = torch.tensor(test_input, dtype=torch.float32).to(device)
test_target = torch.tensor(test_target, dtype=torch.float32).to(device)

test_pred = model(test_input)
inverse_of_fx = model.inverse(test_pred)
inverse_of_y = model.inverse(test_target.to(device))

print(test_pred.shape)
print(inverse_of_fx.shape)
print(inverse_of_y.shape)

test_pred = test_pred.cpu().detach().numpy()
inverse_of_fx = inverse_of_fx.cpu().detach().numpy()
inverse_of_y = inverse_of_y.cpu().detach().numpy()


# Concatenate arrays along the columns (axis=1)
combined_data = np.concatenate((test_pred, inverse_of_fx, inverse_of_y), axis=1)
# print(combined_data.shape)
# Define column names for each array
columns = ['uA_pred', 'uB_pred', 'uC_pred', 'uA_notuse', 'uB_notuse',
           'x_inv', 'y_inv', 't_inv', 'wx_inv', 'wy_inv',
           'x_inv_real', 'y_inv_real', 't_inv_real', 'wx_inv_real', 'wy_inv_real']
data_pred = pd.DataFrame(combined_data, columns=columns)

x = data_selected['x'].values
y = data_selected['y'].values
wx = data_selected['wx'].values
wy = data_selected['wy'].values
t = data_selected['t'].values

uA_true = data_selected['uA'].values
uA_pred = data_pred['uA_pred'].values
uA_error = (uA_true - uA_pred) ** 2
uB_true = data_selected['uB'].values
uB_pred = data_pred['uB_pred'].values
uB_error = (uB_true - uB_pred) ** 2
uC_true = data_selected['uC'].values
uC_pred = data_pred['uC_pred'].values
uC_error = (uC_true - uC_pred) ** 2


x_inv = data_pred['x_inv'].values
y_inv = data_pred['y_inv'].values
t_inv = data_pred['t_inv'].values
wx_inv = data_pred['wx_inv'].values
wy_inv = data_pred['wy_inv'].values

x_inv_real = data_pred['x_inv_real'].values
y_inv_real = data_pred['y_inv_real'].values
t_inv_real = data_pred['t_inv_real'].values
wx_inv_real = data_pred['wx_inv_real'].values
wy_inv_real = data_pred['wy_inv_real'].values

x_error = (x - x_inv) ** 2
x_error_real = (x - x_inv_real) ** 2
y_error = (y - y_inv) ** 2
y_error_real = (y - y_inv_real) ** 2
t_error = (t - t_inv) ** 2
t_error_real = (t - t_inv_real) ** 2
wx_error = (wx - wx_inv) ** 2
wx_error_real = (wx - wx_inv_real) ** 2
wy_error = (wy - wy_inv) ** 2
wy_error_real = (wy - wy_inv_real) ** 2


# Add 'x' and 'y' as new columns to the existing DataFrame
data_pred['uA_true'] = uA_true
data_pred['uB_true'] = uB_true
data_pred['uC_true'] = uC_true
data_pred['x'] = x
data_pred['y'] = y
data_pred['wx'] = wx
data_pred['wy'] = wy
data_pred['t'] = t

data_pred['uA_error'] = uA_error
data_pred['uB_error'] = uB_error
data_pred['uC_error'] = uC_error

data_pred['x_error'] = x_error
data_pred['y_error'] = y_error
data_pred['t_error'] = t_error
data_pred['wx_error'] = wx_error
data_pred['wy_error'] = wy_error

data_pred['x_error_real'] = x_error_real
data_pred['y_error_real'] = y_error_real
data_pred['t_error_real'] = t_error_real
data_pred['wx_error_real'] = wx_error_real
data_pred['wy_error_real'] = wy_error_real


csv_file_path = f'./results/{model_name}_data_selected_{time}.csv'
data_pred.to_csv(csv_file_path, index=False)

print(data_pred.columns)
