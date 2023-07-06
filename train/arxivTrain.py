from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.util import SAGE, GATNet, try_gpu, run_graph
import torch
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import util.util as util
import matplotlib.pyplot as plt

d_name = 'ogbn-arxiv'
dataset = DglNodePropPredDataset(name = d_name)
evaluator = Evaluator(name = d_name)
split_idx = dataset.get_idx_split()
graph, labels = dataset[0]
graph.add_edges(*graph.all_edges()[::-1])
graph = graph.remove_self_loop().add_self_loop()

num_epochs, num_hidden, num_layers, dropout, lr = 500, 256, 3, 0.5, 0.005

node_features = graph.ndata['feat']

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
Model = SAGE(num_input, num_hidden, num_output, num_layers, dropout, try_gpu())
# Model = GATNet(num_input, num_hidden, num_output, num_layers, 4, dropout).to(util.try_gpu())
Opt = torch.optim.AdamW(Model.parameters(), lr=lr)
Loss = F.nll_loss

plt.xlabel('epoch')
plt.ylabel('train_acc')
pltx = [epoch+1 for epoch in range(num_epochs)]
loss_list, train_acc, valid_acc, test_acc = run_graph(graph, labels, split_idx, evaluator, num_epochs, Model, Loss, Opt, True)
plt.plot(pltx, loss_list)
plt.savefig('../image/train.jpg')