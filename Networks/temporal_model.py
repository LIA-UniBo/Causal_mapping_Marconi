import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
from utility import *


# model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # fully connected layer
        self.fc1 = nn.Linear(n_inputs, 64)
        # fully connected layer
        self.fc2 = nn.Linear(64, 128)
        # fully connected layer
        self.fc3 = nn.Linear(128, 1)
 
    # forward propagate input
    def forward(self, X):
        
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        #X = self.fc2(X)
        
        return X