import os

import pandas as pd

from DatasetMarconi import DatasetMarconi
from Modules.TCDF_modules_validation import TCDFModule_validation
from utility import *
from Modules.TCDF_module import *
from torch.utils.data import DataLoader

device = torch.device("cpu" if torch.cuda.is_available() else "cpu") #for now using only cpu


extreme = False
node = "merged" #merged for considering multiple nodes
confidence = 0.8
distance = 4

#Hyper-parameters
levels = 2      #   >2 in order to add middle layers
kernel_size = 4 #   size of the window
dilation = 4    #   dilation of the convolution
lr = 0.1
epochs = 400
batch_size = 8
consider = []


split = True
loss = nn.MSELoss()
features = []
target = []
results = {}


if not extreme:
    xls = "./results/validation/{}_{}_{}.xlsx".format(node,distance,confidence)
else:
    xls = "./results/validation/{}_{}_{}_extreme.xlsx".format(node,distance,confidence)

graphml = "./results/{}_{}.graphml".format(node,confidence)

for file in os.listdir("Data/Marconi_data/sliced_data_{}/train".format(node)):
    df_train = pd.read_csv("Data/Marconi_data/sliced_data_{}/train/{}".format(node,file))
    columns = df_train.columns
    break


for effect in columns:
    causes,_ = find_causes_weights(graphml,effect)

    if causes==[]:
        consider.append(effect)
    else:
        indipendent_features = find_indipendent_feature(graphml,effect,distance=distance)
        if not extreme:
            if len(causes) > len(indipendent_features) and not extreme:
                consider.append(effect)
            else:
                indipendent_features = random.sample(indipendent_features, len(causes))
        features = []
        row = [effect]

        features.append(causes)
        features.append(indipendent_features)

        for feature in features:
            time_series_train = DatasetMarconi()
            time_series_val = DatasetMarconi()
            for file in os.listdir("Data/Marconi_data/sliced_data_{}/train".format(node)):
                df_train = pd.read_csv("Data/Marconi_data/sliced_data_{}/train/{}".format(node,file))
                x_train,y_train = preprocessing_validation(df_train, feature, effect)
                time_series_train.append(x_train,y_train)

            for file in os.listdir("Data/Marconi_data/sliced_data_{}/validation".format(node)):
                df_val = pd.read_csv("Data/Marconi_data/sliced_data_{}/validation/{}".format(node,file))
                x_val, y_val = preprocessing_validation(df_val, feature, effect)
                time_series_val.append(x_val,y_val)

            data_loader_train= DataLoader(time_series_train,batch_size=batch_size,collate_fn=collate_function)
            data_loader_val = DataLoader(time_series_val,batch_size=1,collate_fn=collate_function)
            in_channels = len(feature)
            print("-------Training network for {}".format(effect))
            module = TCDFModule_validation(in_channels=in_channels,levels=levels,kernel_size=kernel_size,dilation=dilation,device=device,lr=lr,
                                           epochs=epochs,confidence_s=confidence)
            module.train(training=data_loader_train,validation=data_loader_val,split=split)
            print("-------Training done")
            row.append(module.losses_val[-1])
            row.append(len(feature))
        results[effect] = row

#df_results = pd.DataFrame.from_dict(results,orient="index")
#df_results.to_excel(xls,index=False)
print(consider)






