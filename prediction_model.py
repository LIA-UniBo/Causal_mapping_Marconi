import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
import pandas as pd
from utility import *
import networkx as nx
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class DatasetPredMarconi:
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx, :-1].values
        label = self.data.iloc[idx, -1]

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return {'data' : data, 'label' : label}

# model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # fully connected layer
        self.fc1 = nn.Linear(n_inputs, 64)
        # fully connected layer
        self.fc2 = nn.Linear(64, 1)
 
    # forward propagate input
    def forward(self, X):
        
        X = F.relu(self.fc1(X))
        #X = F.relu(self.fc2(X))
        X = self.fc2(X)
        
        return X


graph = "./results/merged_0.8.graphml"
G = nx.read_graphml(graph)
G = nx.DiGraph(G)
feature = 'gpu1_mem_temp'
causes, weights = find_causes_weights_noloop('./results/merged_0.8.graphml', feature)
total = []
for i in causes:
    cause, weight = find_causes_weights_noloop('./results/merged_0.8.graphml', i)
    total.extend(cause)
    for j in weight:
        weights[j] = weights[i] + weight[j]
total.extend(causes)

causes = total
causes.append(feature)
print(causes, weights)
df = pd.read_csv('./Data/Marconi_data/sliced_data_r205n12/train/slice_0.csv')
df = df[causes]

#for col in weights:
 #   df[col] = df[col].shift(weights[col])

df = df.fillna(0)

train_data = DataLoader(DatasetPredMarconi(df))
dataset_len = len(train_data)

n_inputs = len(causes) - 1
net = MLP(n_inputs)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

#train
value_loss = []
for epoch in range(30):
    running_loss = 0.0
    for i, value in enumerate(train_data):
        inputs = value['data']
        labels = value['label']
        predictions = net(inputs)
        loss = criterion(predictions, labels)
        running_loss += loss.item() * inputs.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    value_loss.append((running_loss/dataset_len))
    print('Epoch:', epoch, 'loss:', running_loss/dataset_len)

#plt.show(value_loss)

#test
df = pd.read_csv('./Data/Marconi_data/sliced_data_r205n12/test/slice_31.csv')
df = df[causes]
test_data = DatasetPredMarconi(df)
dataset_len = len(test_data)

running_loss = 0.0
accuracy = 0.0
for i, value in enumerate(test_data):
    inputs = value['data']
    labels = value['label']

    pred = net(inputs)
    loss = criterion(pred, labels)

    running_loss += loss.item() * inputs.size(0)

print('Test loss:', running_loss/dataset_len)



        