import os

from DatasetMarconi import DatasetMarconi
from Modules.TCDF_modules_validation import TCDFModule_validation
from utility import *
from Modules.TCDF_module import *
from torch.utils.data import DataLoader

device = torch.device("cpu" if torch.cuda.is_available() else "cpu") #for now using only cpu
split = True
loss = nn.MSELoss()
node = "merged" #merged for considering multiple nodes
effect = "gpu3_mem_temp"
confidence = 0.8

#Hyper-parameters
levels = 2      #   >2 in order to add middle layers
kernel_size = 4 #   size of the window
dilation = 4    #   dilation of the convolution
lr = 0.1
epochs = 400
batch_size = 8



features = []
target = []

time_series_train = DatasetMarconi()
time_series_val = DatasetMarconi()

xls = "./results/validation/{}_{}.xlsx".format(node,confidence)
graphml = "./results/{}_{}.graphml".format(node,confidence)
df = pd.read_excel(xls)



causes = find_causes(graphml,effect)
indipendent_features = find_indipendent_feature(graphml,effect)

if len(causes) > len(indipendent_features):
    raise Exception("Causes has higher dimensionality")

indipendent_features = random.sample(indipendent_features, len(causes))

features = []
row = [effect]

features.append(causes)
features.append(indipendent_features)

for feature in features:
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

    in_channels = len(features)


    print("-------Training network")
    module = TCDFModule_validation(in_channels=in_channels,levels=levels,kernel_size=kernel_size,dilation=dilation,device=device,lr=lr,
                                   epochs=epochs,confidence_s=confidence)
    module.train(training=data_loader_train,validation=data_loader_val,split=split)
    print("-------Training done")
    row.append(module.losses_val[-1])

df.loc[len(df.index)] = row






