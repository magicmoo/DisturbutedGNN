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

def cal_parameter_size(model):
    sz = 0
    params = list(model.parameters())
    for param in params:
        sz += param.nelement() + param.element_size()
    return sz

def _train(model, Loss, graph, labels, train_idx, opt):
    model.train()
    opt.zero_grad()

    node_features = graph.ndata['feat']
    pred_labels = model.cal(graph, node_features)
    pred_train = pred_labels[train_idx]
    # train_output = nn.functional.one_hot(labels[train_idx], num_classes=pred_train.shape[-1]).squeeze()
    train_output = labels[train_idx].squeeze(1)
    loss = Loss(pred_train, train_output)
    graph = graph.to(torch.device('cpu'))
    loss.backward()
    model = model.to(try_gpu())
    opt.step()
    return loss

def cal_loss(Model, Loss, blocks, output_nodes, labels):
    Model.train()

    Model = Model.to(try_gpu())
    node_features = blocks[0].srcdata['feat']
    train_labels = labels[output_nodes].squeeze(1).to(try_gpu())
    pred_labels = Model(blocks, node_features)
    # train_output = nn.functional.one_hot(labels[train_idx], num_classes=pred_train.shape[-1]).squeeze()
    loss = Loss(pred_labels, train_labels)
    
    Model = Model.to(torch.device('cpu'))
    return loss

def cal_loss2(labels, dataloader, Models, Loss, Opts):
    idx, num_workers, loss = 0, Models.__len__(), 0
    for input_nodes, output_nodes, blocks in dataloader:
        model, opt = Models[idx], Opts[idx]
        blocks = [b.to(try_gpu()) for b in blocks]
        loss += cal_loss(model, Loss, blocks, output_nodes, labels).item()/num_workers
        idx += 1
        if idx == num_workers:
            break
    return loss

def run2(graph, labels, dataloader, split_idx, evaluator, num_epochs, Models, Loss, Opts, is_output=False):
    num_workers = Models.__len__()
    model, opt = Models[0], Opts[0]
    time_now, pltx, pltx2 = 0, [], []
    loss_list, train_list, valid_list, test_list = [], [], [], []
    step = 100   # the step program output train's data
    idx, loss = 0, 0.0
    input_nodes, output_nodes, blocks = next(iter(dataloader))
    blocks = [b.to(try_gpu()) for b in blocks]
    compute_time = cal_compute_time(model, opt, blocks, labels, output_nodes, Loss)

    for epoch in range(num_epochs):
        _loss = _train(model, Loss, graph, labels, split_idx['train'], opt).item()
        time_now += 0.01

        loss_list.append(_loss)
        pltx.append(time_now/60)
        pltx2.append(time_now/60)
        # graph = graph.to(try_gpu())
        model = model.to(torch.device('cpu'))
        train_acc, valid_acc, test_acc = Stochastic_test(model, graph, labels, split_idx, evaluator)
        train_list.append(train_acc)
        valid_list.append(valid_acc)
        test_list.append(test_acc)

        # if time_now > max_time:
        #     break

        if is_output and (epoch+1)%(num_epochs//step) == 0:
            train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {_loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}')
            print(f'time_now: {time_now}')
    return loss_list, train_list, valid_list, test_list, pltx, pltx2

d_name = 'ogbn-products'
dataset = DglNodePropPredDataset(name = d_name)
evaluator = Evaluator(name = d_name)
split_idx = dataset.get_idx_split()
graph, labels = dataset[0]
graph.add_edges(*graph.all_edges()[::-1])
graph = graph.remove_self_loop().add_self_loop()
num_epochs, num_hidden, num_layers, dropout, lr = 5000, 256, 2, 0.5, 0.001

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
    opts.append(torch.optim.AdamW(models[i].parameters(), lr=lr))

sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
dataloader = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    # batch_size = (node_features[split_idx['train']].shape[0] + num_workers - 1) // num_workers,
    batch_size = 1024,
    shuffle=True,
    drop_last=False,
    num_workers=0)

dataloader2 = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    batch_size = (node_features[split_idx['train']].shape[0] + num_workers - 1) // num_workers,
    # batch_size = 1024,
    shuffle=True,
    drop_last=False,
    num_workers=0)

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

bandwidth = 10 * 1000 * 1000 * 1000 / 8

input_nodes, output_nodes, blocks = next(iter(dataloader))
blocks = [b.to(try_gpu()) for b in blocks]
cal_compute_time(models[0], opts[0], blocks, labels, output_nodes, Loss)
compute_time = cal_compute_time(models[0], opts[0], blocks, labels, output_nodes, Loss)
sample_time = cal_sample_time(dataloader)

print(f'sample_time: {sample_time}')
print(f'compute_time: {compute_time}')

max_time = 60 * 1

plt.subplot(1, 2, 1)
plt.xlabel('Wall-clock time(min)')
plt.ylabel('loss')
plt.subplot(1, 2, 2)
plt.xlabel('Wall-clock time(min)')
plt.ylabel('test_acc')

loss_list, train_list, valid_list, test_list, pltx, pltx2 = run2(graph, labels, dataloader2, split_idx, evaluator, num_epochs, models, Loss, opts, True)
plt.subplot(1, 2, 1)
plt.plot(pltx, loss_list)
plt.subplot(1, 2, 2)
plt.plot(pltx2, test_list)
for model in models:
    model.reset_parameters()
print(test_list[-1])

plt.legend(['baseline1', 'baseline2', 'baseline3(step=8)', 'baseline3(step=128)', 'baseline4', 'baseline5'])
plt.savefig('./image/test.jpg')