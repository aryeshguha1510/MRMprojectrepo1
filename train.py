# %% [code] {"execution":{"iopub.status.busy":"2024-01-29T03:32:49.793935Z","iopub.execute_input":"2024-01-29T03:32:49.794714Z","iopub.status.idle":"2024-01-29T03:32:49.802333Z","shell.execute_reply.started":"2024-01-29T03:32:49.794676Z","shell.execute_reply":"2024-01-29T03:32:49.801313Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-01-29T03:32:49.805605Z","iopub.execute_input":"2024-01-29T03:32:49.805965Z","iopub.status.idle":"2024-01-29T03:34:54.204960Z","shell.execute_reply.started":"2024-01-29T03:32:49.805932Z","shell.execute_reply":"2024-01-29T03:34:54.203837Z"},"jupyter":{"outputs_hidden":false}}
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import matplotlib.pyplot as plt
from model2 import CNN 
torch.manual_seed(4)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lossfunc = nn.CrossEntropyLoss() 

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]) ,       
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]) 
)

train_size=int(0.8 * len(train_data))
val_size = len(train_data) - train_size


train_sampler = SubsetRandomSampler(range(train_size))
val_sampler = SubsetRandomSampler(range(train_size, len(train_data)))


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999))


num_epochs = 10

losslist = []
best_vacc=0.0

#def train(num_epochs, cnn, loaders):
model.train()
    
    # Train the model
total_step = len(loaders['train'])
    
    
    
for epoch in range(num_epochs):
    correct=0
    samples=0
    totalLoss=0
    for images,labels in loaders['train']:
            
            # gives batch data, normalize x when iterate train_loader
        images = images.to(device)  # Assuming device is defined as torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        labels = labels.to(device)
        output = model(images)
        loss = lossfunc(output, labels)
            
            # clear gradients for this training step
        optimizer.zero_grad()
            
            # backpropagation, compute gradients
        loss.backward()
            
            # apply gradients
        optimizer.step()
            
        totalLoss += loss.item()
            
        _, predicted = torch.max(output, 1)
        samples += labels.size(0)
        correct += (predicted == labels).sum().item()
            
        
    losslist.append(totalLoss/len(loaders['train']))
    tacc = correct/samples
    
   



    correct = 0
    samples = 0    
    model.eval()
    with torch.no_grad():
        for images, labels in loaders['val']:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            
            _, predicted = torch.max(output, 1)
            samples += labels.size(0)
            correct += (predicted == labels).sum().item()
    vacc = correct/samples
    print(f"Epoch {epoch+1}, Training Acc: {tacc:.3f}, Valset Acc: {vacc:.3f}")
    if vacc > best_vacc:
        best_vacc = vacc
        torch.save(model.state_dict(), 'weights.pth')
        print('Weights Saved')
   


         
plt.plot(range(1, num_epochs + 1), losslist, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# %% [code]
