from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.model import StochasticSAGE, StochasticGATNet
from util.multiWorker import multi_Stochastic_run_graph, replaceModel, avgModel
from util.util import Stochastic_train, try_gpu, Stochastic_test
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


num_epochs, num_hidden, num_layers, dropout, lr = 30, 256, 2, 0.5, 0.001

node_features = graph.ndata['feat']

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
Loss = F.nll_loss

models, opts = [], []
num_workers = 4
for i in range(num_workers):
    models.append(StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout))
    opts.append(torch.optim.AdamW(models[i].parameters(), lr=lr))

batch_size = 1024
sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
dataloader = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    batch_size = (node_features[split_idx['train']].shape[0] + num_workers - 1) // num_workers,
    shuffle=True,
    drop_last=False,
    num_workers=0)

cnt = 0
it = iter(dataloader)
print(node_features[split_idx['train']].shape[0])
print((node_features[split_idx['train']].shape[0] + num_workers - 1) // num_workers)
for i in range(num_workers):
    it = next(it)
    input_nodes, output_nodes, blocks = it
    print(blocks)
