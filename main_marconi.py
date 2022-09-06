import os

import pandas as pd
import torch

from DatasetMarconi import DatasetMarconi
from utility import *
from TCDF_net import *
from torch import optim
from tqdm import trange
import torch.nn as nn
from TCDF_module import *
from torch.utils.data import DataLoader




gt_name = ""
depth = 2
device = torch.device("cpu" if torch.cuda.is_available() else "cpu") #for now using only cpu
split = True
loss = nn.MSELoss()
node = "r207n02" #merged for considering multiple nodes

#Hyper-parameters
levels = 2      #   >2 in order to add middle layers
kernel_size = 4 #   size of the window
dilation = 4    #   dilation of the convolution
lr = 0.1
epochs = 4
batch_size = 8
confidence = 0.9

dict_loss_anomalies = {}
map_dataframes_x_train = {}
map_dataframes_y_train = {}
map_module = {}
pr_cause_effect_map = {}

#find columns name
for file in os.listdir("Data/Marconi_data/sliced_data_{}/train".format(node)):
    df_train = pd.read_csv("Data/Marconi_data/sliced_data_{}/train/{}".format(node,file))
    columns = df_train.columns
    break

map_dataframes_train = {col: DatasetMarconi() for col in columns}
map_dataframes_val = {col: DatasetMarconi() for col in columns}



for file in os.listdir("Data/Marconi_data/sliced_data_{}/train".format(node)):
    df_train = pd.read_csv("Data/Marconi_data/sliced_data_{}/train/{}".format(node,file))
    for col in columns:
        x_train = preprocessing(df_train, col)[0]
        y_train = preprocessing(df_train, col)[1]
        map_dataframes_train[col].append(x_train,y_train)

for file in os.listdir("Data/Marconi_data/sliced_data_{}/validation".format(node)):
    df_val = pd.read_csv("Data/Marconi_data/sliced_data_{}/validation/{}".format(node,file))
    for col in columns:
        x_val = preprocessing(df_val, col)[0]
        y_val = preprocessing(df_val, col)[1]
        map_dataframes_val[col].append(x_val,y_val)

#conversion Dataframe into DataLoader
for col in columns:
     dataframe_train = map_dataframes_train[col]
     dataframe_val = map_dataframes_val[col]
     map_dataframes_train[col] = DataLoader(dataframe_train,batch_size=batch_size,collate_fn=collate_function)
     map_dataframes_val[col] = DataLoader(dataframe_val,batch_size=1,collate_fn=collate_function)

in_channels = len(columns)

for col in map_dataframes_train:
    data_train = map_dataframes_train[col]
    data_val = map_dataframes_val[col]
    print("-------Training network for: "+str(col))
    module = TCDFModule(in_channels=in_channels,levels=levels,kernel_size=kernel_size,dilation=dilation,device=device,lr=lr,
                        epochs=epochs,confidence_s=confidence)
    module.train(training=data_train,validation=data_val,split=split)
    map_module[col] = module
print("-------Training done")


print("-------Saving all the models")
for col in map_dataframes_train:
    torch.save(map_module[col].network.state_dict(),"./trained_model/model_{}.pth".format(col))
print("-------Done")


effect = 0
for col in map_dataframes_train:
    print("-------Interpreting attention for: "+str(col))
    causes = map_module[col].find_causes()
    print("Found causes: "+str(causes))
    delays = map_module[col].find_delay(causes)
    print("Founded dealys: "+str(delays))
    for i in range(len(causes)):
        if causes[i]==effect:
            pr_cause_effect_map[tuple([causes[i],effect])] = delays[i]+1
        else:
            pr_cause_effect_map[tuple([causes[i], effect])] = delays[i]
    effect+=1
print("-------Interpreting attention done")

print("-------Evaluation")
if gt_name != "":
    cause_effect_map, cause_effect_extended_map = get_ground_truth(gt_name, map_dataframes_train, depth)
    f1,f1_prime = compute_f1(pr_cause_effect_map,cause_effect_map,cause_effect_extended_map,depth)
    print("F1 score: "+str(f1))
    print("F1 prime score: " + str(f1_prime))
plot_graph(pr_cause_effect_map, list(map_dataframes_train.keys()),node,confidence)

