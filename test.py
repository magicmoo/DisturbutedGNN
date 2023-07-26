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
from dgl.data import RedditDataset

d_name = 'ogbn-products'
dataset = DglNodePropPredDataset(name = d_name)
# dataset = RedditDataset()
# evaluator = Evaluator(name = d_name)
# split_idx = dataset.get_idx_split()
graph, labels = dataset[0]
# graph = dataset[0]
node_feature = graph.ndata['feat']
# labels = graph.ndata['label']
tmp = torch.arange(0, node_feature.shape[0])
print(labels)
# split_idx = {'train': graph.ndata['train_mask'], 'valid': graph.ndata['val_mask'], 'test': graph.ndata['test_mask']}

# print(labels.shape)
# # graph.add_edges(*graph.all_edges()[::-1])
# graph = graph.remove_self_loop().add_self_loop()