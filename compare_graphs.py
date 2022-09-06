import networkx as nx
from utility import *

map_1 = "r207n02.graphml"
map_2 = "r207n02.graphml"
list_edges_1 = []
list_edges_2 = []
graph_1 = nx.read_graphml(map_1)
graph_2 = nx.read_graphml(map_2)
edges_1 = graph_1.edges(data=True)
edges_2 = graph_2.edges(data=True)

for edge in edges_1:
    list_edges_1.append([edge[0],edge[1]])
for edge in edges_2:
    list_edges_2.append([edge[0], edge[1]])


f1 = compute_f1(pr_cause_effect_map=list_edges_1, gt_cause_effect_map=list_edges_2,depth=3)
print(f1)