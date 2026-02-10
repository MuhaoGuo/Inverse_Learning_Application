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


datapath = "/Users/muhaoguo/Documents/study/Paper_Projects/Datasets/Data_Jiaqi/data"

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
targets = data_all[['uA', 'uB', 'uC']].values



inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)
dataset = TensorDataset(inputs, targets)



batch_size = 24460
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(5, 64)  # （x, y, time）
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)  # （uA, uB, uC, wx, wy）
    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = self.fc4(y)
        return y

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
                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))
torch.manual_seed(1234)

epochs = 50

device = get_device()



model = NN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_model(model, train_loader, optimizer, criterion, epochs, device)
time = 60
data_selected = data_all[data_all['t'] == time]

test_input = data_selected[['x', 'y', 't', 'wx', 'wy']].values
data_selected_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)
test_pred = model(data_selected_tensor)
data_pred = test_pred.cpu().detach().numpy()
data_pred = pd.DataFrame(data_pred, columns=['uA', 'uB', 'uC'])
x = data_selected['x'].values
y = data_selected['y'].values
uA_true = data_selected['uA'].values
uA_pred = data_pred['uA'].values
uA_error = (uA_true - uA_pred) ** 2
uB_true = data_selected['uB'].values
uB_pred = data_pred['uB'].values
uB_error = (uB_true - uB_pred) ** 2
uC_true = data_selected['uC'].values
uC_pred = data_pred['uC'].values
uC_error = (uC_true - uC_pred) ** 2

data = pd.DataFrame({
    'x': x,
    'y': y,
    'uA_true': uA_true,
    'uA_pred': uA_pred,
    'uA_error': uA_error,
    'uB_true': uB_true,
    'uB_pred': uB_pred,
    'uB_error': uB_error,
    'uC_true': uC_true,
    'uC_pred': uC_pred,
    'uC_error': uC_error
})

csv_file_path = f'data_selected_{time}.csv'
data.to_csv(csv_file_path, index=False)

time = 60

data_selected = data_all[data_all['t'] == time]
test_input = data_selected[['x', 'y', 't', 'wx', 'wy']].values
np.random.seed(0)
noise_wx = np.random.normal(0, np.mean(test_input[:, 3]) * 0.5, test_input[:, 3].shape)
noise_wy = np.random.normal(0, np.mean(test_input[:, 4]) * 0.5, test_input[:, 4].shape)

test_input[:, 3] += noise_wx
test_input[:, 4] += noise_wy
data_selected_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)

test_pred = model(data_selected_tensor)

data_pred = test_pred.cpu().detach().numpy()

data_pred = pd.DataFrame(data_pred, columns=['uA', 'uB', 'uC'])
x = data_selected['x'].values
y = data_selected['y'].values
uA_true = data_selected['uA'].values
uA_pred = data_pred['uA'].values
uA_error = (uA_true - uA_pred) ** 2
uB_true = data_selected['uB'].values
uB_pred = data_pred['uB'].values
uB_error = (uB_true - uB_pred) ** 2
uC_true = data_selected['uC'].values
uC_pred = data_pred['uC'].values
uC_error = (uC_true - uC_pred) ** 2

data = pd.DataFrame({
    'x': x,
    'y': y,
    'uA_true': uA_true,
    'uA_pred': uA_pred,
    'uA_error': uA_error,
    'uB_true': uB_true,
    'uB_pred': uB_pred,
    'uB_error': uB_error,
    'uC_true': uC_true,
    'uC_pred': uC_pred,
    'uC_error': uC_error
})

csv_file_path = f'noise_data_selected_{time}.csv'

data.to_csv(csv_file_path, index=False)

