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
import numpy as np


d_name = 'ogbn-products'
dataset = DglNodePropPredDataset(name = d_name)
evaluator = Evaluator(name = d_name)
split_idx = dataset.get_idx_split()
graph, labels = dataset[0]
# graph.add_edges(*graph.all_edges()[::-1])
graph = graph.remove_self_loop().add_self_loop()
num_epochs, num_hidden, num_layers, dropout, lr = 500000, 256, 2, 0.5, 0.001

node_features = graph.ndata['feat']

# r = int(node_features.shape[-1] * 0.25)
# node_features[:, r:] = 0
num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
# Model = StochasticGATNet(num_input, num_hidden, num_output, num_layers, 4, dropout)
Loss = F.nll_loss

model = StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])

def cal_compute_time(Model, Opt, blocks, labels, output_nodes, Loss):
    nums = 100
    Model = Model.to(try_gpu())
    labels = labels.to(try_gpu())
    t1 = time.time()
    for _ in range(nums):
        Model.train()
        Opt.zero_grad()

        node_features = blocks[0].srcdata['feat']
        train_labels = labels[output_nodes].squeeze(1)
        pred_labels = Model(blocks, node_features)
        # train_output = nn.functional.one_hot(labels[train_idx], num_classes=pred_train.shape[-1]).squeeze()
        loss = Loss(pred_labels, train_labels)
        loss.backward()
        Opt.step()
    t2 = time.time()

    Model.reset_parameters()
    Model = Model.to(torch.device('cpu'))
    labels = labels.to(torch.device('cpu'))
    return (t2-t1)/nums

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

def cal_parameter_size(model):
    sz = 0
    params = list(model.parameters())
    for param in params:
        sz += param.nelement() + param.element_size()
    return sz

bandwidth = 10 * 1000 * 1000 * 1000 / 8
batch_list = [1024, 2048, 4096]

computation_list, para_transmission_list, feature_list, neighbour_sampling_list = [], [], [], []

for batch_size in batch_list:
    dataloader = dgl.dataloading.DataLoader(
        graph, split_idx['train'], sampler,
        batch_size = batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0)
    input_nodes, output_nodes, blocks = next(iter(dataloader))
    blocks = [b.to(try_gpu()) for b in blocks]
    print(blocks)
    cal_compute_time(model, opt, blocks, labels, output_nodes, Loss)
    computation_list.append(cal_compute_time(model, opt, blocks, labels, output_nodes, Loss))
    para_transmission_list.append(2 * cal_parameter_size(model) / bandwidth)
    neighbour_sampling_list.append(cal_sample_time(dataloader))
    data = blocks[0].srcdata['feat']
    print(data.element_size())
    feature_list.append(data.element_size() * data.nelement() / bandwidth)

d = [0 for _ in range(batch_list.__len__())]
ind = np.arange(batch_list.__len__())
width = 0.35
plt.xticks(ind, (str(batch_size) for batch_size in batch_list))
p1 = plt.bar(ind, computation_list, width)#, yerr=menStd)

d = [d[i]+computation_list[i] for i in range(batch_list.__len__())]
p2 = plt.bar(ind, para_transmission_list, width, bottom=d)#, yerr=womenStd)

d = [d[i]+para_transmission_list[i] for i in range(batch_list.__len__())]
p3 = plt.bar(ind, feature_list, width, bottom=d)

d = [d[i]+feature_list[i] for i in range(batch_list.__len__())]
p4 = plt.bar(ind, neighbour_sampling_list, width, bottom=d)

plt.legend((p1[0], p2[0], p3[0], p4[0]), ('computation_time', 'para_transmission_time', 'feature_retrieval_time', 'neighbour_sampling_time'))
plt.ylabel('Wall clock time(s)')

plt.savefig('./image/timeBreakdownTest.jpg')

print(computation_list)
print(para_transmission_list)
print(feature_list)
print(neighbour_sampling_list)