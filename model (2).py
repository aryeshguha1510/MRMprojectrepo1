# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.352488Z","iopub.execute_input":"2024-01-28T14:20:32.353218Z","iopub.status.idle":"2024-01-28T14:20:32.364761Z","shell.execute_reply.started":"2024-01-28T14:20:32.353188Z","shell.execute_reply":"2024-01-28T14:20:32.363780Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.366183Z","iopub.execute_input":"2024-01-28T14:20:32.366801Z","iopub.status.idle":"2024-01-28T14:20:32.375317Z","shell.execute_reply.started":"2024-01-28T14:20:32.366776Z","shell.execute_reply":"2024-01-28T14:20:32.374483Z"}}
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

from torch.utils.data import SubsetRandomSampler

# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.376614Z","iopub.execute_input":"2024-01-28T14:20:32.376891Z","iopub.status.idle":"2024-01-28T14:20:32.387865Z","shell.execute_reply.started":"2024-01-28T14:20:32.376868Z","shell.execute_reply":"2024-01-28T14:20:32.386978Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.389257Z","iopub.execute_input":"2024-01-28T14:20:32.389535Z","iopub.status.idle":"2024-01-28T14:20:32.465813Z","shell.execute_reply.started":"2024-01-28T14:20:32.389512Z","shell.execute_reply":"2024-01-28T14:20:32.465015Z"}}
from torchvision import datasets
from torchvision.transforms import ToTensor
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.467765Z","iopub.execute_input":"2024-01-28T14:20:32.468240Z","iopub.status.idle":"2024-01-28T14:20:32.473585Z","shell.execute_reply.started":"2024-01-28T14:20:32.468206Z","shell.execute_reply":"2024-01-28T14:20:32.472743Z"}}
print(train_data)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.474660Z","iopub.execute_input":"2024-01-28T14:20:32.474949Z","iopub.status.idle":"2024-01-28T14:20:32.484326Z","shell.execute_reply.started":"2024-01-28T14:20:32.474925Z","shell.execute_reply":"2024-01-28T14:20:32.483418Z"}}
print(test_data)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.486484Z","iopub.execute_input":"2024-01-28T14:20:32.487114Z","iopub.status.idle":"2024-01-28T14:20:32.494987Z","shell.execute_reply.started":"2024-01-28T14:20:32.487081Z","shell.execute_reply":"2024-01-28T14:20:32.494015Z"}}
print(train_data.data.size())

  

# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.496145Z","iopub.execute_input":"2024-01-28T14:20:32.496475Z","iopub.status.idle":"2024-01-28T14:20:32.754801Z","shell.execute_reply.started":"2024-01-28T14:20:32.496449Z","shell.execute_reply":"2024-01-28T14:20:32.753858Z"}}
import matplotlib.pyplot as plt
plt.imshow(train_data.data[50], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()


train_size=int(0.8 * len(train_data))
val_size = len(train_data) - train_size

train_sampler = SubsetRandomSampler(range(train_size))
val_sampler = SubsetRandomSampler(range(train_size, len(train_data)))
# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.756160Z","iopub.execute_input":"2024-01-28T14:20:32.756606Z","iopub.status.idle":"2024-01-28T14:20:32.764642Z","shell.execute_reply.started":"2024-01-28T14:20:32.756572Z","shell.execute_reply":"2024-01-28T14:20:32.763785Z"}}
from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100,
                                          sampler=train_sampler,
                                          #shuffle=True, 
                                          num_workers=1),
    'val' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100,
                                          sampler=val_sampler,
                                          #shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}
loaders

# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.767003Z","iopub.execute_input":"2024-01-28T14:20:32.767362Z","iopub.status.idle":"2024-01-28T14:20:32.778140Z","shell.execute_reply.started":"2024-01-28T14:20:32.767326Z","shell.execute_reply":"2024-01-28T14:20:32.777298Z"}}
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.779184Z","iopub.execute_input":"2024-01-28T14:20:32.779497Z","iopub.status.idle":"2024-01-28T14:20:32.790462Z","shell.execute_reply.started":"2024-01-28T14:20:32.779473Z","shell.execute_reply":"2024-01-28T14:20:32.789631Z"}}
cnn = CNN()
print(cnn)

# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.791576Z","iopub.execute_input":"2024-01-28T14:20:32.791841Z","iopub.status.idle":"2024-01-28T14:20:32.801080Z","shell.execute_reply.started":"2024-01-28T14:20:32.791817Z","shell.execute_reply":"2024-01-28T14:20:32.800214Z"}}
loss = nn.CrossEntropyLoss()   
loss

# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T14:20:32.802175Z","iopub.execute_input":"2024-01-28T14:20:32.802450Z","iopub.status.idle":"2024-01-28T14:20:32.814968Z","shell.execute_reply.started":"2024-01-28T14:20:32.802427Z","shell.execute_reply":"2024-01-28T14:20:32.814081Z"}}
from torch import optim
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   
optimizer




#cnn = CNN().to(device)
# %% [code]
