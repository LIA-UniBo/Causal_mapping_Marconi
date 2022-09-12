import os
from DatasetMarconi import DatasetMarconi
from utility import *
from Modules.TCDF_module import *
from torch.utils.data import DataLoader

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
node = "r215n10"
levels_list = [2]
kernel_size_list = [4]
lr_list = [1e-1]
epochs = 1000
batch_size = 8
split = True



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

for levels in levels_list:
    for kernel_size in kernel_size_list:
        dilation = kernel_size
        for lr in lr_list:
            print("Grid search for :")
            print("     Levels: "+str(levels))
            print("     Kernel: "+str(kernel_size))
            print("     Lr: "+str(lr))
            history_loss_train = {}
            history_loss_val = {}
            for col in map_dataframes_train:
                data_train = map_dataframes_train[col]
                data_val = map_dataframes_val[col]
                print("-------Training network for: " + str(col))
                module = TCDFModule(in_channels=in_channels, levels=levels, kernel_size=kernel_size,
                                    dilation=dilation, device=device, lr=lr,
                                    epochs=epochs)
                module.train(training=data_train, validation=data_val, split=split)
                history_loss_train[col] = module.losses_train
                history_loss_val[col] = module.losses_val
            df_loss_train = pd.DataFrame.from_dict(history_loss_train)
            df_loss_val = pd.DataFrame.from_dict(history_loss_val)
            df_loss_train['average'] = df_loss_train.mean(numeric_only=True, axis=1)
            df_loss_val['average'] = df_loss_val.mean(numeric_only=True, axis=1)
            df_loss_train.to_excel("./Grid_search/levels_{}_kernel_{}_lr{}/final_results_train_{}.xlsx"
                                 .format(levels,kernel_size,lr, node))
            df_loss_val.to_excel("./Grid_search/levels_{}_kernel_{}_lr{}/final_results_val_{}.xlsx"
                                 .format(levels, kernel_size, lr, node))


