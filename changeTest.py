from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.util import try_gpu, Stochastic_run_graph
from util.model import StochasticSAGE, StochasticGATNet
import torch
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import util.util as util
import matplotlib.pyplot as plt

d_name = 'ogbn-products'
dataset = DglNodePropPredDataset(name = d_name)
evaluator = Evaluator(name = d_name)
split_idx = dataset.get_idx_split()
graph, labels = dataset[0]
# graph.add_edges(*graph.all_edges()[::-1])
graph = graph.remove_self_loop().add_self_loop()
node_features = graph.ndata['feat']

file = open('./savedata/products.adj', mode = 'w')

vec = []
edges = graph.edges()
num_nodes = node_features.shape[0]
num_edges = edges[0].shape[0]
num_features = node_features.shape[-1]
for _ in range(num_nodes):
    vec.append([])

for i in range(num_edges):
    u, v = edges[0][i].item(), edges[1][i].item()
    vec[u].append(v)

for i in range(num_nodes):
    file.write(str(i)+"\t")
    for j in range(num_features):
        file.write(str(node_features[i][j].item()) + " ")
    file.write("\t" + str(vec[i].__len__()) + " ")
    for j in range(vec[i].__len__()):
        file.write(str(vec[i][j]) + " ")
    file.write("\n")