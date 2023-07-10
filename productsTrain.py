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


num_epochs, num_hidden, num_layers, dropout, lr = 20, 256, 2, 0.5, 0.001

node_features = graph.ndata['feat']

# r = int(node_features.shape[-1] * 0.25)
# node_features[:, r:] = 0

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
# Model = StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout)
Model = StochasticGATNet(num_input, num_hidden, num_output, num_layers, 4, dropout)
Opt = torch.optim.AdamW(Model.parameters(), lr=lr)
Loss = F.nll_loss


sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
dataloader = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0)

plt.xlabel('epoch')
plt.ylabel('loss')
pltx = [epoch+1 for epoch in range(num_epochs)]
loss_list, train_acc, valid_acc, test_acc = Stochastic_run_graph(graph, labels, dataloader, split_idx, evaluator, num_epochs, Model, Loss, Opt, True)
plt.plot(pltx, loss_list)
plt.savefig('./image/train.jpg')
print("----------------------------")
print(f'train_acc: {train_acc:.2}')
print(f'valid_acc: {valid_acc:.2}')
print(f'test_acc: {test_acc:.2}') 
