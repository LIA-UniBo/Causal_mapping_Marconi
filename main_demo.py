import pandas as pd
from utility import *
from TCDF_net import *
from torch import optim
from tqdm import trange
import torch.nn as nn
from TCDF_module import *


file_name = "Data/Demo_data/demo_dataset.csv"
gt_name = "./Data/Demo_data/demo_groundtruth.csv"
gt_name = ""
depth = 2
device = torch.device("cpu" if torch.cuda.is_available() else "cpu") #for now using only cpu
split = False
loss = nn.MSELoss()

#Hyper-parameters
levels = 2      #   >2 in order to add middle layers
kernel_size = 4 #   size of the window
dilation = 4    #   dilation of the convolution
lr = 0.01
epochs = 1000

df = pd.read_csv(file_name)
map_dataframes = {}
map_module = {}
pr_cause_effect_map = {}

for col in df.columns:
    map_dataframes[col] = preprocessing(df,col)

in_channels = len(df.columns)

for col in map_dataframes:
    x = map_dataframes[col][0]
    y = map_dataframes[col][1]
    if split:
        x_train,y_train,x_val,y_val,x_test,y_test = split_train_val_test(x,y)
    else:
        x_train = x
        y_train = y
    print("-------Training network for: "+str(col))
    module = TCDFModule(in_channels=in_channels,levels=levels,kernel_size=kernel_size,dilation=dilation,device=device,lr=lr,
                        epochs=epochs)
    module.train_old(x_train=x_train,y_train=y_train)
    map_module[col] = module
print("-------Training done")

effect = 0
for col in map_dataframes:
    print("-------Interpreting attention for: "+str(col))
    causes = map_module[col].find_causes()
    print("Found causes: "+str(causes))
    delays = map_module[col].find_delay(causes)
    print("Founded dealys: "+str(delays))
    for i in range(len(causes)):
        pr_cause_effect_map[tuple([causes[i],effect])] = delays[i]
    effect+=1
print("-------Interpreting attention done")

print("-------Evaluation")
if gt_name != "":
    cause_effect_map, cause_effect_extended_map = get_ground_truth(gt_name,map_dataframes,depth)
    f1,f1_prime = compute_f1(pr_cause_effect_map,cause_effect_map,cause_effect_extended_map,depth)
    print("F1 score: "+str(f1))
    print("F1 prime score: " + str(f1_prime))
plot_graph(pr_cause_effect_map,list(map_dataframes.keys()))

