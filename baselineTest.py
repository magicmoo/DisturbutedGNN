from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.model import StochasticSAGE, StochasticGATNet
from util.multiWorker import multi_Stochastic_run_graph, avgModel, replaceModel
from util.util import try_gpu, Stochastic_train, Stochastic_test
import torch
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import util.util as util
import matplotlib.pyplot as plt

def run1(graph, labels, dataloader, split_idx, evaluator, num_epochs, Models, Loss, Opts, is_output=False):
    node_features = graph.ndata['feat']
    num_workers = Models.__len__()

    loss_list, train_list, valid_list, test_list = [], [], [], []
    step = 10   # the step program output train's data
    idx, loss = 0, 0.0
    for epoch in range(num_epochs):
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(try_gpu()) for b in blocks]
            loss += Stochastic_train(Models[idx], Loss, blocks, output_nodes, labels, Opts[idx])/num_workers
            idx = idx+1
            if idx == num_workers:
                loss_list.append(loss)
                _loss = loss
                print(_loss)
                loss, idx = 0.0, 0
                avgModel(Models)
        train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
        train_list.append(train_acc)
        valid_list.append(valid_acc)
        test_list.append(test_acc)

        if is_output and (epoch+1)%(num_epochs//step) == 0:
            train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {_loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}')
    return loss_list, train_list, valid_list, test_list

def run2(graph, labels, dataloader, split_idx, evaluator, num_epochs, Models, Loss, Opts, is_output=False):
    node_features = graph.ndata['feat']
    num_workers = Models.__len__()

    loss_list, train_list, valid_list, test_list = [], [], [], []
    step = 10   # the step program output train's data
    idx = 0
    for epoch in range(num_epochs):
        loss = 0
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(try_gpu()) for b in blocks]
            loss += Stochastic_train(Models[idx], Loss, blocks, output_nodes, labels, Opts[i])/num_workers
            idx = (idx+1)%num_workers
        loss_list.append(loss/num_workers)
        
        avgModel(Models)
        train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
        train_list.append(train_acc)
        valid_list.append(valid_acc)
        test_list.append(test_acc)

        if is_output and (epoch+1)%(num_epochs//step) == 0:
            train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}')
    return loss_list, train_list, valid_list, test_list

def run3(graph, labels, dataloader, split_idx, evaluator, num_epochs, Models, Loss, Opts, correct_step, is_output=False):
    node_features = graph.ndata['feat']
    num_workers, num_features = Models.__len__(), node_features.shape[-1]
    split_list = [0]
    for i in range(num_workers):
        split_list.append(split_list[-1] + (num_features//num_workers))
        if i < num_features%num_workers:
            split_list[-1] += 1

    loss_list, train_list, valid_list, test_list = [], [], [], []
    step = 10   # the step program output train's data
    idx, loss = 0, 0.0
    for epoch in range(num_epochs):
        if epoch%correct_step == 0:
            input_nodes, output_nodes, blocks = next(iter(dataloader))
            blocks = [b.to(try_gpu()) for b in blocks]
            loss = Stochastic_train(Models[0], Loss, blocks, output_nodes, labels, Opts[0])
            replaceModel(Models)
            continue

        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(torch.device('cpu')) for b in blocks]
            blocks[0].srcdata['feat'][:, :] = 0
            blocks[0].srcdata['feat'][:, split_list[i]:split_list[i+1]] = node_features[input_nodes, split_list[i]:split_list[i+1]]*num_workers
            blocks = [b.to(try_gpu()) for b in blocks]

            loss += Stochastic_train(Models[idx], Loss, blocks, output_nodes, labels, Opts[idx])/num_workers
            idx = idx+1
            if idx == num_workers:
                loss_list.append(loss)
                _loss = loss
                print(_loss)
                loss, idx = 0.0, 0
                avgModel(Models)
        train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
        train_list.append(train_acc)
        valid_list.append(valid_acc)
        test_list.append(test_acc)

        if is_output and (epoch+1)%(num_epochs//step) == 0:
            train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {_loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}')
    return loss_list, train_list, valid_list, test_list

def run4(graph, labels, dataloader, split_idx, evaluator, num_epochs, Models, Loss, Opts, is_output=False):
    node_features = graph.ndata['feat']
    num_workers, num_features = Models.__len__(), node_features.shape[-1]
    split_list = [0]
    for i in range(num_workers):
        split_list.append(split_list[-1] + (num_features//num_workers))
        if i < num_features%num_workers:
            split_list[-1] += 1

    loss_list, train_list, valid_list, test_list = [], [], [], []
    step = 10   # the step program output train's data
    idx, loss = 0, 0.0
    for epoch in range(num_epochs):
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(torch.device('cpu')) for b in blocks]
            blocks[0].srcdata['feat'][:, :] = 0
            blocks[0].srcdata['feat'][:, split_list[i]:split_list[i+1]] = node_features[input_nodes, split_list[i]:split_list[i+1]]*num_workers
            blocks = [b.to(try_gpu()) for b in blocks]

            loss += Stochastic_train(Models[idx], Loss, blocks, output_nodes, labels, Opts[idx])/num_workers
            idx = idx+1
            if idx == num_workers:
                loss_list.append(loss)
                _loss = loss
                print(_loss)
                loss, idx = 0.0, 0
                avgModel(Models)
        train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
        train_list.append(train_acc)
        valid_list.append(valid_acc)
        test_list.append(test_acc)

        if is_output and (epoch+1)%(num_epochs//step) == 0:
            train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {_loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}')
    return loss_list, train_list, valid_list, test_list


d_name = 'ogbn-products'
dataset = DglNodePropPredDataset(name = d_name)
evaluator = Evaluator(name = d_name)
split_idx = dataset.get_idx_split()
graph, labels = dataset[0]
# graph.add_edges(*graph.all_edges()[::-1])
graph = graph.remove_self_loop().add_self_loop()
num_epochs, num_hidden, num_layers, dropout, lr = 500, 256, 2, 0.5, 0.001

node_features = graph.ndata['feat']

# r = int(node_features.shape[-1] * 0.25)
# node_features[:, r:] = 0

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
# Model = StochasticGATNet(num_input, num_hidden, num_output, num_layers, 4, dropout)
Loss = F.nll_loss

models, opts = [], []
num_workers = 1
for i in range(num_workers):
    models.append(StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout))
    opts.append(torch.optim.AdamW(models[i].parameters(), lr=lr))

sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
dataloader = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    batch_size = (node_features[split_idx['train']].shape[0] + num_workers - 1) // num_workers,
    # batch_size = 1024,
    shuffle=True,
    drop_last=False,
    num_workers=0)

plt.xlabel('epoch')
plt.ylabel('loss')

# loss_list, train_list, valid_list, test_list = run1(graph, labels, dataloader, split_idx, evaluator, num_epochs, models, Loss, opts, True)
loss_list, train_list, valid_list, test_list = run4(graph, labels, dataloader, split_idx, evaluator, num_epochs, models, Loss, opts, True)
pltx = [iteration+1 for iteration in range(loss_list.__len__())]

plt.plot(pltx, loss_list)
plt.savefig('./image/train.jpg')