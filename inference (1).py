# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-01-28T17:44:19.726281Z","iopub.execute_input":"2024-01-28T17:44:19.726919Z","iopub.status.idle":"2024-01-28T17:44:19.736443Z","shell.execute_reply.started":"2024-01-28T17:44:19.726861Z","shell.execute_reply":"2024-01-28T17:44:19.735082Z"}}
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

import model
import training

# %% [code] {"execution":{"iopub.status.busy":"2024-01-28T17:46:19.217731Z","iopub.execute_input":"2024-01-28T17:46:19.218218Z","iopub.status.idle":"2024-01-28T17:46:22.127734Z","shell.execute_reply.started":"2024-01-28T17:46:19.218173Z","shell.execute_reply":"2024-01-28T17:46:22.125781Z"}}
def val():
    # Test the model
    model.cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for training.images, training.labels in model.loaders['val']:
            test_output, last_layer = model.cnn(training.images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == training.labels).sum().item() / float(training.labels.size(0))
           # pass
    print('Test Accuracy of the model on val set : %.2f' % accuracy)
    
    #pass
val()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-01-28T17:44:44.785980Z","iopub.execute_input":"2024-01-28T17:44:44.786510Z","iopub.status.idle":"2024-01-28T17:44:47.140445Z","shell.execute_reply.started":"2024-01-28T17:44:44.786463Z","shell.execute_reply":"2024-01-28T17:44:47.138077Z"}}
import torch
def test():
    # Test the model
    model.cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for training.images, training.labels in model.loaders['test']:
            test_output, last_layer = model.cnn(training.images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == training.labels).sum().item() / float(training.labels.size(0))
           # pass
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    
    #pass
test()

# %% [code]
