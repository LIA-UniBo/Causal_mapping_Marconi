import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx
import datetime as dt

def collate_function(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


def read_df(file_name):
    '''
    Read the dataframe, given a file_name
    '''
    df = pd.read_parquet(file_name)
    return df


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

def find_window_dict(df,delete,threshold):
    window = 0
    windows_dict = {}
    to_delete = dt.timedelta(minutes=15*delete)
    after_anomaly = False
    for i in range(len(df.index)-1):
        row = df.index[i]
        next_row = df.index[i+1]
        if window == 0:
            starting_date_current_window = row
        if df['new_label'][row] == 2:
            window = window - delete
            if window > threshold:
                if after_anomaly:
                    windows_dict[tuple([starting_date_current_window + to_delete,row - to_delete])] = window - 2*delete
                else:
                    windows_dict[tuple([starting_date_current_window, row - to_delete])] = window - delete
            window = 0
            after_anomaly = True
        elif ((next_row - row).total_seconds()/60) != 15.0:
            if window > threshold:
                if after_anomaly:
                    windows_dict[tuple([starting_date_current_window + to_delete ,row])] = window - delete
                else:
                    windows_dict[tuple([starting_date_current_window, row])] = window
            window = 0
            after_anomaly = False
        else:
            window +=1
    return windows_dict

def preprocessing(df,target):
    copy_df = df.copy()
    y = copy_df[[target]]
    y_shifted = y.shift(axis=0)
    copy_df[target] = y_shifted.fillna(0)
    x = torch.from_numpy(copy_df.values.astype('float32').transpose())
    y = torch.from_numpy(y.values.astype('float32').transpose())
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    return x,y

def split_train_val_test(x,y):
    size = x.shape[2]
    index_test = int(size*80/100)
    index_val = int(int(size*80/100)*80/100)
    x_tmp = x[:,:, :index_test]
    y_tmp = y[:, :, :index_test]
    x_test = x[:,:,index_test:]
    y_test = y[:,:,index_test:]
    x_train =  x_tmp[:, :, :index_val]
    y_train = y_tmp[:, :, :index_val]
    x_val = x_tmp[:, : ,index_val:]
    y_val = y_tmp[:, : ,index_val:]
    return x_train,y_train,x_val,y_val,x_test,y_test




def get_ground_truth(csv_location,df,depth):
    cause_effect_map = {}
    df = pd.read_csv(csv_location,header=None)
    for index, row in df.iterrows():
        cause = row[0]
        effect = row[1]
        delay = row[2]
        cause_effect_map[tuple([cause,effect])] =  delay
    #use the private function
    cause_effect_extended_map = get_extended_cause_effect_map(cause_effect_map,depth)
    return cause_effect_map,cause_effect_extended_map

def compute_delay(path,cause_effect_map):
    delay = 0
    for i in range(len(path)-1):
        start = path[i]
        end = path[i+1]
        delay += cause_effect_map[tuple([start,end])]
    return delay


def get_extended_cause_effect_map(cause_effect_map,depth):
    all_path = []
    graph = nx.DiGraph()
    cause_effect_extended_map = []
    nodes = list(set(el for t in cause_effect_map for el in t))
    for el in nodes:
        graph.add_node(el)
    for el in cause_effect_map:
        graph.add_edge(el[0],el[1])
    for start in nodes:
        for end in nodes:
            paths = list(nx.all_simple_paths(graph, start, end, cutoff=depth))
            if paths != []:
                all_path.append([start,end])

    for cause_effect_extended in all_path:
        if cause_effect_extended not in cause_effect_map:
            cause_effect_extended_map.append([cause_effect_extended[0],cause_effect_extended[1]])
    return cause_effect_extended_map


def compute_f1(pr_cause_effect_map, gt_cause_effect_map, depth):
    tp = 0
    tp_prime = 0
    fp = 0
    fp_prime = 0
    fn= 0
    fn_prime = 0
    for el in pr_cause_effect_map:
        if el in gt_cause_effect_map:
            tp +=1
        else:
            fp +=1
    for el in gt_cause_effect_map:
        if el not in pr_cause_effect_map:
            fn+=1

    pr_extended_cause_effect_map = get_extended_cause_effect_map(pr_cause_effect_map,depth)
    gt_extended_cause_effect_map = get_extended_cause_effect_map(gt_cause_effect_map,depth)
    pr_extended_cause_effect_map = pr_extended_cause_effect_map + pr_cause_effect_map
    gt_extended_cause_effect_map = gt_extended_cause_effect_map + gt_cause_effect_map
    for el in pr_extended_cause_effect_map:
        if el in gt_extended_cause_effect_map:
            tp_prime += 1
        else:
            fp_prime += 1
    for el in gt_extended_cause_effect_map:
        if el not in pr_extended_cause_effect_map:
            fn_prime += 1

    f1 = (2*tp)/(2*tp + 2*fn + fp)
    f1_prime = (2*tp_prime)/(2*tp_prime + 2*fn + fp_prime)

    return f1,f1_prime





def plot_graph(cause_effect_map,col_names,filename,confidence):
    graph = nx.DiGraph()
    cause_effect_named = {}
    for col in col_names:
        graph.add_node(col)
    for cause_effect in cause_effect_map:
        cause = cause_effect[0]
        effect = cause_effect[1]
        graph.add_edge(col_names[cause],col_names[effect],weight=cause_effect_map[cause_effect])

    for cause_effect in cause_effect_map:
        cause = cause_effect[0]
        effect = cause_effect[1]
        cause_effect_named[col_names[cause],col_names[effect]] = cause_effect_map[cause_effect]

    pos = nx.circular_layout(graph)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=cause_effect_named)
    nx.draw(graph, pos, with_labels=True)
    nx.draw_networkx(graph, pos=pos)
    plt.savefig("graph_one_dataset.png", format="PNG")
    plt.show()
    nx.write_graphml(graph, "./results/{}_{}.graphml".format(filename,confidence))
