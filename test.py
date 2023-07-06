from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.util import try_gpu, run_graph
from util.model import StochasticSAGE, StochasticGATNet
import torch
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import util.util as util
import matplotlib.pyplot as plt
from util.multiWorker import avgModel

d_name = 'ogbn-products'
dataset = DglNodePropPredDataset(name = d_name)
evaluator = Evaluator(name = d_name)
split_idx = dataset.get_idx_split()
graph, labels = dataset[0]
# graph.add_edges(*graph.all_edges()[::-1])
# graph = graph.remove_self_loop().add_self_loop()


num_epochs, num_hidden, num_layers, dropout, lr, batch_size = 500, 256, 3, 0.5, 0.005, 1024

node_features = graph.ndata['feat']

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
Model = StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout)

Model2 = StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout)
# Model = GATNet(num_input, num_hidden, num_output, num_layers, 4, dropout).to(util.try_gpu())

params = list(Model.parameters())
params2 = list(Model2.parameters())

params[-1].requires_grad = False
params[-1][:] = 114514

avgModel([Model, Model2])