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


class SmallNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

        self.register_buffer('mask_tril', torch.tril(torch.ones_like(self.fc1.weight)))
        self.register_buffer('mask_triu', torch.triu(torch.ones_like(self.fc2.weight)))
        self.fc1.weight = nn.Parameter(torch.mul(self.fc1.weight, self.mask_tril))
        self.fc2.weight = nn.Parameter(torch.mul(self.fc2.weight, self.mask_triu))

        # Apply masks and ensure diagonal elements are ones
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

        # Register hooks to enforce masking after each update
        self.fc1.weight.register_hook(lambda grad: grad * self.mask_tril)
        self.fc2.weight.register_hook(lambda grad: grad * self.mask_triu)

        self.negative_slope = 0.5


    def forward(self, x):
        fc1_fwd = self.fc1(x)
        fc1_fwd = F.leaky_relu(fc1_fwd, negative_slope=self.negative_slope)
        fc2_fwd = self.fc2(fc1_fwd)
        fc2_fwd = F.leaky_relu(fc2_fwd, negative_slope=self.negative_slope)
        return fc2_fwd

    def inverse(self, y):
        batch_size = y.shape[0]
        # Inverse of the last activation
        y1 = F.leaky_relu(y, negative_slope=1 / self.negative_slope)
        # Subtract bias
        y1 = y1 - self.fc2.bias
        # Solve fc2.weight * x = y1 using triangular solve
        y1_unsqueezed = y1.unsqueeze(2)  # shape (batch_size, input_dim, 1)
        fc2_weight_expanded = self.fc2.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # fc2_inv, _ = torch.triangular_solve(y1_unsqueezed, fc2_weight_expanded, upper=True)
        fc2_inv = torch.linalg.solve_triangular(fc2_weight_expanded, y1_unsqueezed, upper=True)

        fc2_inv = fc2_inv.squeeze(2)

        # Inverse of the first activation
        y2 = F.leaky_relu(fc2_inv, negative_slope=1 / self.negative_slope)
        # Subtract bias
        y2 = y2 - self.fc1.bias
        # Solve fc1.weight * x = y2 using triangular solve
        y2_unsqueezed = y2.unsqueeze(2)
        fc1_weight_expanded = self.fc1.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # fc1_inv, _ = torch.triangular_solve(y2_unsqueezed, fc1_weight_expanded, upper=False)
        fc1_inv = torch.linalg.solve_triangular(fc1_weight_expanded, y2_unsqueezed, upper=False)

        fc1_inv = fc1_inv.squeeze(2)
        return fc1_inv

    def apply_weight_masks(self):
        with torch.no_grad():
            self.fc1.weight.data.mul_(self.mask_tril)
            torch.diagonal(self.fc1.weight.data).fill_(1.0)
            self.fc2.weight.data.mul_(self.mask_triu)
            torch.diagonal(self.fc2.weight.data).fill_(1.0)

class DipDNN_Block(nn.Module):
    def __init__(self, input_dim, alpha=0.9):
        super(DipDNN_Block, self).__init__()
        # self.alpha = alpha  # Scaling coefficient
        self.residual_function = SmallNetwork(input_dim)

    def forward(self, x):
        return self.residual_function(x)

    def inverse(self, y):
        return self.residual_function.inverse(y)

    def apply_weight_masks(self):
        self.residual_function.apply_weight_masks()


class DipDNN(nn.Module):
    def __init__(self, input_dim, num_blocks):
        super(DipDNN, self).__init__()
        self.blocks = nn.ModuleList(
            [DipDNN_Block(input_dim) for _ in range(num_blocks)])
        self.input_dim = input_dim

    def forward(self, x):

        # x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for block in self.blocks:
            x = block(x)

        # x = x.reshape(x.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        return x

    def inverse(self, y):
        # y = y.reshape(y.shape[0], -1)  # (batchsize, -1)

        for block in reversed(self.blocks):
            y = block.inverse(y)

        # y = y.reshape(y.shape[0], 1, int(np.sqrt(self.input_dim)), int(np.sqrt(self.input_dim)))
        return y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_weight_masks(self):
        for block in self.blocks:
            block.apply_weight_masks()


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
                model.apply_weight_masks()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))
torch.manual_seed(1234)


input_dim = 5
num_blocks = 3
learning_rate = 1e-3
epochs = 400
time = 60

device = get_device()
# device = torch.device('cpu')
model_name = "dipdnn"
model = DipDNN(input_dim=input_dim, num_blocks=num_blocks).to(device)
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



#
# time = 60
#
# data_selected = data_all[data_all['t'] == time]
# test_input = data_selected[['x', 'y', 't', 'wx', 'wy']].values
# np.random.seed(0)
# noise_wx = np.random.normal(0, np.mean(test_input[:, 3]) * 0.5, test_input[:, 3].shape)
# noise_wy = np.random.normal(0, np.mean(test_input[:, 4]) * 0.5, test_input[:, 4].shape)
#
# test_input[:, 3] += noise_wx
# test_input[:, 4] += noise_wy
# data_selected_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)
#
# test_pred = model(data_selected_tensor)
#
# data_pred = test_pred.cpu().detach().numpy()
#
# data_pred = pd.DataFrame(data_pred, columns=['uA', 'uB', 'uC'])
# x = data_selected['x'].values
# y = data_selected['y'].values
# uA_true = data_selected['uA'].values
# uA_pred = data_pred['uA'].values
# uA_error = (uA_true - uA_pred) ** 2
# uB_true = data_selected['uB'].values
# uB_pred = data_pred['uB'].values
# uB_error = (uB_true - uB_pred) ** 2
# uC_true = data_selected['uC'].values
# uC_pred = data_pred['uC'].values
# uC_error = (uC_true - uC_pred) ** 2
#
# data = pd.DataFrame({
#     'x': x,
#     'y': y,
#     'uA_true': uA_true,
#     'uA_pred': uA_pred,
#     'uA_error': uA_error,
#     'uB_true': uB_true,
#     'uB_pred': uB_pred,
#     'uB_error': uB_error,
#     'uC_true': uC_true,
#     'uC_pred': uC_pred,
#     'uC_error': uC_error
# })
#
# csv_file_path = f'noise_data_selected_{time}.csv'
#
# data.to_csv(csv_file_path, index=False)
#
