from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.model import StochasticSAGE, StochasticGATNet
from util.multiWorker import multi_Stochastic_run_graph
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
# Model = StochasticGATNet(num_input, num_hidden, num_output, num_layers, 4, dropout)
Loss = F.nll_loss

models, opts = [], []
num_workers = 4
for i in range(num_workers):
    models.append(StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout))
    opts.append(torch.optim.AdamW(models[-1].parameters(), lr=lr))

sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
dataloader = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0)

correct_list = [8, 16, 32, 64, 128]
plt.xlabel('epoch')
plt.ylabel('test_acc')

pltx = [epoch+1 for epoch in range(num_epochs)]
for correct_step in correct_list:
    for model in models:
        model.reset_parameters()

    loss_list, train_list, valid_list, test_list = multi_Stochastic_run_graph(graph, labels, dataloader, split_idx, evaluator, num_epochs, models, Loss, opts, correct_step, False)
    plt.plot(pltx, test_list)
    print("----------------------------")
    print(f'step: {correct_step}')
    print(f'train_acc: {train_list[-1]:.2}')
    print(f'valid_acc: {valid_list[-1]:.2}')
    print(f'test_acc: {test_list[-1]:.2}')

plt.legend([f'step = {step}' for step in correct_list])
plt.savefig('./image/correctionTest.jpg')
