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


num_epochs, num_hidden, num_layers, dropout, lr = 50, 256, 2, 0.5, 0.001

node_features = graph.ndata['feat']

# r = int(node_features.shape[-1] * 0.25)
# node_features[:, r:] = 0

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
Model = StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout)
# Model = StochasticGATNet(num_input, num_hidden, num_output, num_layers, 4, dropout)
Opt = torch.optim.AdamW(Model.parameters(), lr=lr)
Loss = F.nll_loss


sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
dataloader = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0)

pltx, train_acc_list, test_acc_list = [], [], []
plt.xlabel('split_ratio')
plt.ylabel('test_acc')
for i in range(100, 0, -1):
    Model.reset_parameters()
    split_ratio = i / 100
    r = int(node_features.shape[-1]*split_ratio)
    node_features[:, r+1:] = 0
    pltx.append(split_ratio)
    loss_list, train_acc, valid_acc, test_acc = Stochastic_run_graph(graph, labels, dataloader, split_idx, evaluator, num_epochs, Model, Loss, Opt, False)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f'split_ratio: {split_ratio}, train_acc: {train_acc}, test_acc: {test_acc}')

plt.plot(pltx, train_acc_list)
plt.plot(pltx, test_acc_list)
plt.legend(['train_acc', 'test_acc'])
plt.savefig('./image/splitAccTest.jpg')