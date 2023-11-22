from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.model import StochasticSAGE, StochasticGATNet
from util.multiWorker import multi_Stochastic_run_graph, avgModel, replaceModel
from util.util import try_gpu, Stochastic_train, Stochastic_test, train
import time
import torch
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import util.util as util
import matplotlib.pyplot as plt
from math import exp
from dgl.data import RedditDataset
from time import sleep
import os

d_name = 'ogbn-products'
dataset = DglNodePropPredDataset(name = d_name)
evaluator = Evaluator(name = d_name)
split_idx = dataset.get_idx_split()
graph, labels = dataset[0]

graph.add_edges(*graph.all_edges()[::-1])
graph = graph.remove_self_loop().add_self_loop()
num_epochs, num_hidden, num_layers, dropout, lr = 500, 256, 3, 0.5, 0.005
batch_size = 2000
max_time = 60 * 1

node_features = graph.ndata['feat']

# r = int(node_features.shape[-1] * 0.25)
# node_features[:, r:] = 0

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
# Model = StochasticGATNet(num_input, num_hidden, num_output, num_layers, 4, dropout)
Loss = F.nll_loss

models, opts = [], []
num_workers = 4
for i in range(num_workers):
    models.append(StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout).to(try_gpu()))
    # models.append(StochasticGATNet(num_input, num_hidden, num_output, num_layers, 4, dropout))
    opts.append(torch.optim.AdamW(models[i].parameters(), lr=lr))

sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
dataloader = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    # batch_size = (node_features[split_idx['train']].shape[0] + num_workers - 1) // num_workers,
    batch_size = batch_size,
    shuffle=True,
    drop_last=False,
    device=try_gpu(),
    num_workers=0)

def cal_sample_time(dataloader):
    nums = 100
    cnt = 0
    t1 = time.time()
    while True:
        for input_nodes, output_nodes, blocks in dataloader:
            cnt += 1
            if cnt >= nums:
                break
        if cnt >= nums:
                break
    t2 = time.time()
    return (t2-t1)/nums

bandwidth = 10 * 1000 * 1000 * 1000 / 8
bandwidth2 = 25 * 1000 * 1000 * 1000 / 8
bandwidth3 = 40 * 1000 * 1000 * 1000 / 8
bandwidth4 = 100 * 1000 * 1000 * 1000 / 8

input_nodes, output_nodes, blocks = next(iter(dataloader))
print(blocks[0].device)
blocks = [b.to(try_gpu()) for b in blocks]
sample_time = cal_sample_time(dataloader)

print(f'sample_time: {sample_time}')