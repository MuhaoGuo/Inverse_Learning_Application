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

data = pd.read_csv("data_selected_60.csv")
print(data.columns)
print(data.shape)