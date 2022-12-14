import torch.optim as optim
from torch.utils.data import DataLoader
from DatasetPredMarconi import DatasetPredMarconi
from Networks.temporal_model import *
import json


graph_dir = './results/'
graph = "merged_0.8.graphml"
shift = False
features = {}
with open('results/temporal_evaluation/feature_causes.json', 'r') as fp:
    features = json.load(fp)
loss_dict = {}

epochs = 30
batch_size = 8


for feature in  features:
    causes, weights = find_causes_weights_noloop(graph_dir + graph, feature)

    total = []
    #consider a distance of 2 if the causes are < 5
    #if features[feature] < 5:
     #   for i in causes:
      #      cause, weight = find_causes_weights_noloop('./results/merged_0.8.graphml', i)
       #     total.extend(cause)
        #    for j in weight:
         #       weights[j] = weight[j] + weights[i]
    total.extend(causes)
    causes = total

    causes.append(feature)
    df = pd.read_csv('./Data/Marconi_data/sliced_data_merged/train/slice_2.csv')
    df = df[causes]

    #shift the values considering the temporal distance
    if shift:
        for col in weights:
               df[col] = df[col].shift(weights[col])

    #substitute the Nan values with 0
    df = df.fillna(0)

    #prepare data and model
    train_data = DataLoader(DatasetPredMarconi(df))
    dataset_len = len(train_data)
    n_inputs = len(causes) - 1
    net = MLP(n_inputs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    #train
    value_loss = []
    for epoch in range(epochs):
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
    loss_dict[feature] = value_loss 
    

#save the loss
if shift:
    with open('results/temporal_evaluation/loss_shift_epoch30_nodistance.json', 'w') as fp:
        json.dump(loss_dict, fp)
else:
    with open('results/temporal_evaluation/loss_without_shift_epoch30_nodistance.json', 'w') as fp:
        json.dump(loss_dict, fp)






        