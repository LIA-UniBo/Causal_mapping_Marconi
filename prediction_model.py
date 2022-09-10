import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
import pandas as pd
from utility import *
import networkx as nx


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
        self.fc2 = nn.Linear(64, 128)
        # fully connected layer
        self.fc3 = nn.Linear(128, 1)
 
    # forward propagate input
    def forward(self, X):
        
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        
        return X


graph = "./results/merged_0.8.graphml"
G = nx.read_graphml(graph)
G = nx.DiGraph(G)
feature = 'fan0_1'
causes, weights = find_causes_weights('./results/merged_0.8.graphml', feature)
causes.append(feature)
print(weights)
df = pd.read_csv('./Data/Marconi_data/sliced_data_r205n12/train/slice_0.csv')
df = df[causes]



train_data = DatasetPredMarconi(df)
dataset_len = len(train_data)

n_inputs = len(causes) - 1
net = MLP(n_inputs)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

#train
for epoch in range(100):
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
    print('Epoch:', epoch, 'loss:', running_loss/dataset_len)

#test
test_data = DatasetPredMarconi('./Data/Marconi_data/sliced_data_r205n12/test/slice_31.csv')
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


batch_sizes = []
lr_l = []
dense_sizes = []
epochs = []

for b_s in batch_sizes:
    for d_s in dense_sizes:
        for lr in lr_l:
            for epoch in epochs:
                print("Grid search for :")
                print("     Epochs: "+str(epoch))
                print("         Dense size: "+str(d_s))
                print("             Lr: "+str(lr))
        