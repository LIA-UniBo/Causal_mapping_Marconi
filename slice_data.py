import os
from utility import *
import json
import random
import math

data_dir = "Data/Marconi_data/new_data"
selected_features_file = "Data/Marconi_data/selected_features.json"
train = 'Data/Marconi_data/sliced_data_merged/train/'
validation = 'Data/Marconi_data/sliced_data_merged/validation/'
test = 'Data/Marconi_data/sliced_data_merged/test/'
selected_node_list = ["Data/Marconi_data/new_data/r215n10.gzip","Data/Marconi_data/new_data/r207n02.gzip"]
total_size = 0

#Hyperparameter
delete = 8
threshold = 100

map_data = {}
df_list = []
train_list = []
val_list = []
test_list = []


for file_name in os.listdir(data_dir):
    file_name = data_dir+'/'+file_name
    df = read_df(file_name)
    map_data[file_name]=df

for selected_node in selected_node_list:
    df = map_data[selected_node]
    print("Larghezza dataset: " + str(df.shape[0]))
    df = df.set_index('timestamp')
    y_new = df["New_label"]

    #keep only the feature from the file selected_features.json
    file = open(selected_features_file)
    cols = json.load(file)
    df = df[cols]

    #normalization
    df_normalized = (df - df.mean())/df.std()


    #add target on new dataset
    df_normalized['new_label'] = y_new

    window_dict = find_window_dict(df_normalized,delete=delete,threshold=threshold)


    for key in window_dict:
        mask = (df_normalized.index <= pd.Timestamp(key[1])) & (df_normalized.index >=  pd.Timestamp(key[0]))
        df_to_save = df_normalized.loc[mask]
        df_to_save = df_to_save.reset_index(drop=True)
        df_to_save = df_to_save.drop(['new_label'],axis=1)
        col_list = []
        for string in df_to_save.columns:
            col_list.append(string.split('avg:')[1])
        df_to_save.columns = col_list
        df_list.append(df_to_save)
        total_size += df_to_save.shape[0]

random.shuffle(df_list)
test_size = math.ceil(total_size*10/100)
train_size = math.ceil(total_size*90/100)
val_size = math.ceil(train_size*20/100)
train_size = math.ceil(train_size*80/100)

print("")
print("Train: "+str(train_size))
print("Validation: "+str(val_size))
print("Test: "+str(test_size))

counter=0
current_size = 0
for el in df_list:
    if current_size < train_size:
        el.to_csv(train+'slice_{}.csv'.format(counter),index=False)
    elif (train_size <= current_size <= train_size+val_size):
        el.to_csv(validation+'slice_{}.csv'.format(counter),index=False)
    elif train_size+val_size < current_size < train_size+val_size+test_size:
        el.to_csv(test+'slice_{}.csv'.format(counter), index=False)
    counter +=1
    current_size += el.shape[0]






