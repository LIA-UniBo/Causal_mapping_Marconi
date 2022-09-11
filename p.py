from utility import *
import networkx as nx

graph = "./results/merged_0.8.graphml"
G = nx.read_graphml(graph)
G = nx.DiGraph(G)
feature = 'pkts_out'
cause_dict = {}
max = 0
deature_max = ''
for i in G.nodes:

    causes, weights = find_causes_weights_noloop('./results/merged_0.8.graphml', i)
    if len(causes) > max:
        max = len(causes)
        deature_max = i
    for j in causes:
        causes, weights = find_causes_weights('./results/merged_0.8.graphml', j)

print(deature_max)