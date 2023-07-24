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

def cal_parameter_size(model):
    sz = 0
    params = list(model.parameters())
    for param in params:
        sz += param.nelement() + param.element_size()
    return sz

def run3(graph, labels, dataloader, split_idx, evaluator, num_epochs, Models, Loss, Opts, correct_step, is_output=False):
    node_features = graph.ndata['feat']
    num_workers, num_features = Models.__len__(), node_features.shape[-1]
    split_list = [0]
    time_now, pltx = 0, []
    for i in range(num_workers):
        split_list.append(split_list[-1] + (num_features//num_workers))
        if i < num_features%num_workers:
            split_list[-1] += 1

    loss_list, train_list, valid_list, test_list = [], [], [], []
    step = 10   # the step program output train's data
    idx, loss = 0, 0.0
    for epoch in range(num_epochs):
        for input_nodes, output_nodes, blocks in dataloader:
            data = blocks[0].srcdata['feat']
            feature_time = data.element_size() * data.nelement() / bandwidth

            if epoch%(correct_step+1) != 0: 
                feature_time /= num_workers
                blocks = [b.to(torch.device('cpu')) for b in blocks]
                blocks[0].srcdata['feat'][:, :] = 0
                blocks[0].srcdata['feat'][:, split_list[i]:split_list[i+1]] = node_features[input_nodes, split_list[i]:split_list[i+1]]*num_workers
            blocks = [b.to(try_gpu()) for b in blocks]

            loss += Stochastic_train(Models[idx], Loss, blocks, output_nodes, labels, Opts[idx])/num_workers
            idx = idx+1
            if idx == num_workers:
                loss_list.append(loss)
                _loss = loss
                loss, idx = 0.0, 0
                avgModel(Models)

        parameter_time = 2 * cal_parameter_size(Models[0]) / bandwidth
        time_now += compute_time + sample_time + feature_time + parameter_time
        pltx.append(time_now/60)
        train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
        train_list.append(train_acc)
        valid_list.append(valid_acc)
        test_list.append(test_acc)
        
        if time_now > max_time:
            break

        if is_output and (epoch+1)%(num_epochs//step) == 0:
            train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {_loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}')
    return loss_list, train_list, valid_list, test_list, pltx

d_name = 'ogbn-products'
dataset = DglNodePropPredDataset(name = d_name)
evaluator = Evaluator(name = d_name)
split_idx = dataset.get_idx_split()
graph, labels = dataset[0]
graph.add_edges(*graph.all_edges()[::-1])
graph = graph.remove_self_loop().add_self_loop()
num_epochs, num_hidden, num_layers, dropout, lr = 500000, 256, 2, 0.5, 0.005

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

bandwidth = 10 * 1024 * 1024 * 1024

input_nodes, output_nodes, blocks = next(iter(dataloader))
blocks = [b.to(try_gpu()) for b in blocks]
compute_time = cal_compute_time(models[0], opts[0], blocks, labels, output_nodes, Loss)
sample_time = cal_sample_time(dataloader)

print(f'sample_time: {sample_time}')
print(f'compute_time: {compute_time}')

max_time = 1*60
# mode = 'test_acc'
mode = 'loss'

correct_list = [1, 8, 16, 32, 64, 128]
plt.xlabel('epoch')
plt.ylabel('Wall-clock time(min)')

file = open('./savedata/correctStepTest.txt', mode = 'w')
pltx = [epoch+1 for epoch in range(num_epochs)]
for correct_step in correct_list:
    for model in models:
        model.reset_parameters()

    loss_list, train_list, valid_list, test_list, pltx = run3(graph, labels, dataloader, split_idx, evaluator, num_epochs, models, Loss, opts, correct_step, False)
    if mode == 'loss':
        plt.plot(pltx, loss_list)
    else:
        plt.plot(pltx, test_list)
    file.write(str(correct_step)+'\n')
    file.write(str(pltx)+'\n')
    file.write(str(loss_list)+'\n')
    file.write(str(test_list)+'\n')
    print("----------------------------")
    print(f'step: {correct_step}')
    print(f'train_acc: {train_list[-1]:.2}')
    print(f'valid_acc: {valid_list[-1]:.2}')
    print(f'test_acc: {test_list[-1]:.2}')

plt.legend([f'step = {step}' for step in correct_list])
plt.savefig('./image/correctStepTest.jpg')