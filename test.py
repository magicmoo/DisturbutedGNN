from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.util import try_gpu, Stochastic_run_graph
from util.model import SAGE, GATNet
import torch
import dgl
import time
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


num_epochs, num_hidden, num_layers, dropout, lr = 30, 1024, 2, 0.5, 0.0005

node_features = graph.ndata['feat']

# r = int(node_features.shape[-1] * 0.25)
# node_features[:, r:] = 0

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
model = SAGE(num_input, num_hidden, num_output, num_layers, dropout)
# Model = GATNet(num_input, num_hidden, num_output, num_layers, 4, dropout)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
Loss = F.nll_loss


sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
dataloader = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0)

forward_times = []
backward_times = []

for epoch in range(num_epochs):
    model.train()

    # 前向传播时间
    start_time = time.perf_counter()
    logits = model(graph, node_features)
    print(logits.shape)
    loss = Loss(logits, labels)
    end_time = time.perf_counter()
    forward_time = end_time - start_time
    forward_times.append(forward_time)

    # 反向传播时间
    start_time = time.perf_counter()
    opt.zero_grad()
    loss.backward()
    opt.step()
    end_time = time.perf_counter()
    backward_time = end_time - start_time
    backward_times.append(backward_time)

average_forward_time = sum(forward_times) / num_epochs
average_backward_time = sum(backward_times) / num_epochs

print("Average forward propagation time: ", average_forward_time)
print("Average backward propagation time: ", average_backward_time)