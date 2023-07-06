from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.util import try_gpu, run_graph
from util.model import SAGE, GATNet
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

num_epochs, num_hidden, num_layers, dropout, lr = 300, 128, 3, 0.5, 0.005

node_features = graph.ndata['feat']

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
Model = SAGE(num_input, num_hidden, num_output, num_layers, dropout, try_gpu())
# Model = GATNet(num_input, num_hidden, num_output, num_layers, 4, dropout).to(util.try_gpu())
Opt = torch.optim.AdamW(Model.parameters(), lr=lr)
Loss = F.nll_loss

pltx = [epoch+1 for epoch in range(num_epochs)]
plt.xlabel('epoch')
plt.ylabel('loss')
for i in [100, 50, 33, 25]:
    Model.reset_parameters()
    split_ratio = i / 100
    r = int(node_features.shape[-1]*split_ratio)
    node_features[:, r+1:] = 0

    loss_list, train_acc, valid_acc, test_acc = run_graph(graph, labels, split_idx, evaluator, num_epochs, Model, Loss, Opt, False) 
    plt.plot(pltx, loss_list)
    print(f'split_ratio: {split_ratio}, train_acc: {train_acc}, test_acc: {test_acc}')
plt.legend(['split_ratio = 1', 'split_ratio = 0.5', 'split_ratio = 0.33', 'split_ratio = 0.25'])
plt.savefig('./image/splitLossTest.jpg')