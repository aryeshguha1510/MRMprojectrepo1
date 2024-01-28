# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-01-28T17:29:26.550527Z","iopub.execute_input":"2024-01-28T17:29:26.551394Z","iopub.status.idle":"2024-01-28T17:29:26.557817Z","shell.execute_reply.started":"2024-01-28T17:29:26.551361Z","shell.execute_reply":"2024-01-28T17:29:26.556741Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-01-28T17:29:26.559380Z","iopub.execute_input":"2024-01-28T17:29:26.559733Z","iopub.status.idle":"2024-01-28T17:29:26.572410Z","shell.execute_reply.started":"2024-01-28T17:29:26.559706Z","shell.execute_reply":"2024-01-28T17:29:26.571444Z"}}
import torch
import model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-01-28T17:29:26.574267Z","iopub.execute_input":"2024-01-28T17:29:26.574693Z","iopub.status.idle":"2024-01-28T17:29:26.584445Z","shell.execute_reply.started":"2024-01-28T17:29:26.574660Z","shell.execute_reply":"2024-01-28T17:29:26.583694Z"}}
from torch.autograd import Variable

num_epochs = 10

def train(num_epochs, cnn, loaders):
    model.cnn.train()
    
    # Train the model
    total_step = len(loaders['train'])
    
    #losslist=[]
    #epochlist=[]
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]
            loss = model.loss(output, b_y)
            
            # clear gradients for this training step
            model.optimizer.zero_grad()
            
            # backpropagation, compute gradients
            loss.backward()
            
            # apply gradients
            model.optimizer.step()
            
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            #epochlist.append(epoch + 1)
            #losslist.append(loss.item())   

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-01-28T17:29:26.585445Z","iopub.execute_input":"2024-01-28T17:29:26.585717Z","iopub.status.idle":"2024-01-28T17:29:40.867014Z","shell.execute_reply.started":"2024-01-28T17:29:26.585696Z","shell.execute_reply":"2024-01-28T17:29:40.865653Z"}}
 
    
train(num_epochs, model.cnn, model.loaders)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-01-28T17:29:40.868002Z","iopub.status.idle":"2024-01-28T17:29:40.868349Z","shell.execute_reply.started":"2024-01-28T17:29:40.868181Z","shell.execute_reply":"2024-01-28T17:29:40.868197Z"}}
#plt.plot(epochlist, losslist, label='Training Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Loss vs Epoch')
#plt.legend()
#plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
