import datetime
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.model import StochasticSAGE, StochasticGATNet
from util.multiWorker import multi_Stochastic_run_graph, replaceModel
from util.util import try_gpu, Stochastic_test, train
import time
import torch
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import util.util as util
import matplotlib.pyplot as plt
import os
from math import exp
from dgl.data import RedditDataset
import torch.distributed as dist

def Stochastic_train(Model, Loss, blocks, output_nodes, labels, Opt):

    Model.train()
    Opt.zero_grad()

    node_features = blocks[0].srcdata['feat']
    train_labels = labels[output_nodes].squeeze(1).to(device)
    pred_labels = Model(blocks, node_features)
    # train_output = nn.functional.one_hot(labels[train_idx], num_classes=pred_train.shape[-1]).squeeze()
    loss = Loss(pred_labels, train_labels)
    loss.backward()
    Opt.step()
    
    return loss.item()

# to be edit
@torch.no_grad()
def avgModel(model):
    model.eval()
    for p in model.parameters():
        dist.all_reduce(p, op=dist.ReduceOp.AVG)

# to be edit
def cal_loss2(labels, dataloader, Models, Loss, Opts):
    idx, num_workers, loss = 0, Models.__len__(), 0
    input_nodes, output_nodes, blocks = next(iter(dataloader))
    model, opt = Models[idx], Opts[idx]
    blocks = [b.to(device) for b in blocks]
    loss = cal_loss(model, Loss, blocks, output_nodes, labels)/num_workers
    print('debug1')
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    print('debug2')
    return loss.item()

def dist_run(graph, labels, dataloader, split_idx, evaluator, num_epochs, models, Loss, Opts, lr, is_output=False):
    node_features = graph.ndata['feat']
    num_workers, num_features = models.__len__(), node_features.shape[-1]
    overhead, iteration = 0, 0
    split_list = [0]
    for i in range(num_workers):
        split_list.append(split_list[-1] + (num_features//num_workers))
        if i < num_features%num_workers:
            split_list[-1] += 1

    loss_list, train_list, valid_list, test_list = [], [], [], []
    step = 10   # the step program output train's data
    idx, loss, cnt, correct_step = 0, 0.0, -0.5, 0
    pltx = []
    gradient_w1, loss_w1, loss_w2 = 0, 0, 0
    if rank == 0:
        m = cal_m(labels,dataloader, models, Loss, Opts, split_list)
    model, opt = models[0], Opts[0]

    t1 = time.perf_counter()
    for epoch in range(num_epochs):
        _loss = cal_loss2(labels, dataloader, models, Loss, Opts)
        # print(f'rank={rank}, _loss={_loss}')
        if rank == 0:
            loss_list.append(_loss)
            models[0] = models[0].to('cpu')
            train_acc, valid_acc, test_acc = Stochastic_test(models[0], graph, labels, split_idx, evaluator)
            models[0] = models[0].to(device)
            train_list.append(train_acc)
            valid_list.append(valid_acc)
            test_list.append(test_acc)
            pltx.append(time.perf_counter()-t1)
        for input_nodes, output_nodes, blocks in dataloader:
            iteration += 1
            print(f'rank={rank}, iteration={iteration}')
            
            data = blocks[0].srcdata['feat']
            if cnt < correct_step:
                blocks[0].srcdata['feat'][:, :] = 0
                blocks[0].srcdata['feat'][:, split_list[rank]:split_list[rank+1]] = node_features[input_nodes, split_list[idx]:split_list[idx+1]]
            else:
                overhead += data.element_size() * data.nelement() * (1-1/num_workers)

            blocks = [b.to(device) for b in blocks]
            # print(f'rank:{rank} {blocks[0].device}')
            # node_features = blocks[0].srcdata['feat']
            # print(f'rank:{rank} {node_features.device}')

            loss += Stochastic_train(model, Loss, blocks, output_nodes, labels, opt)/num_workers
            
            if rank == 0 and cnt >= correct_step:
                gradient_w1 += cal_gradient(model)/num_workers

            dist.barrier()
            avgModel(model)

            if rank == 0:
                _loss = loss
                # print(_loss)
                loss, idx = 0.0, 0
                if cnt >= correct_step:
                    cnt, loss_w1 = 0, _loss
                    # loss_w2 = cal_loss2(labels, dataloader, models, Loss, Opts)
                    # s1 = (data.shape[0] * data.shape[1] * data.element_size() / compute_time / bandwidth * (gradient_w1 + 2/lr*loss_w2))
                    # s2 = -(gradient_w1 + 2/lr*loss_w2 - 2/lr*loss_w1)
                    s1 = (data.shape[0] * data.shape[1] * data.element_size() / compute_time / bandwidth * (2/lr*loss_w1+(lr*lf-1)*gradient_w1*num_workers))
                    s2 = (1-lr*lf)*gradient_w1*num_workers
                    correct_step = min((s1/m) ** 0.5, s2/m)
                    correct_step = round(correct_step.item(), 0)
                    correct_step = torch.tensor(correct_step, device=device)
                    # correct_step = 0
                    # print(f'correct_step: {correct_step}')
                    # print(f's1: {s1}, s2: {s2}, m: {m}')
                    # print(f'gradient_w1: {gradient_w1}, loss_w1: {loss_w1}, loss_w2: {loss_w2}, time_now: {time_now}')
                    # print("---------------------")
                    gradient_w1, loss_w1, loss_w2 = 0, 0, 0
                else:
                    cnt += 1
            elif rank != 0 and cnt >= correct_step:
                cnt = 0
                correct_step = torch.tensor(0.0, device=device)
            else:
                cnt += 1
            if cnt == 0:
                dist.barrier()
                # print(f'rank={rank}, type={type(correct_step)}')
                dist.broadcast(correct_step, src=0)
                correct_step = correct_step.item()
                # print(f'rank={rank}, correct_step={correct_step}')

        # print(f'time_now: {time_now}')

        if is_output and (epoch+1)%(num_epochs//step) == 0 and rank == 0:
            train_acc, valid_acc, test_acc = Stochastic_test(model[0], graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {_loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}')
    # if rank == 0:
    print(f'finished! rank={rank}')
    _loss = cal_loss2(labels, dataloader, models, Loss, Opts)
    loss_list.append(_loss)
    models[0] = models[0].to('cpu')
    train_acc, valid_acc, test_acc = Stochastic_test(models[0], graph, labels, split_idx, evaluator)
    models[0] = models[0].to(device)
    train_list.append(train_acc)
    valid_list.append(valid_acc)
    test_list.append(test_acc)
    pltx.append(time.perf_counter()-t1)
    return loss_list, train_list, valid_list, test_list, pltx, overhead, iteration

def cal_parameter_size(model):
    sz = 0
    params = list(model.parameters())
    for param in params:
        sz += param.nelement() + param.element_size()
    return sz

def cal_loss(Model, Loss, blocks, output_nodes, labels):
    Model.train()

    node_features = blocks[0].srcdata['feat']
    train_labels = labels[output_nodes].squeeze(1).to(device)
    pred_labels = Model(blocks, node_features)
    # train_output = nn.functional.one_hot(labels[train_idx], num_classes=pred_train.shape[-1]).squeeze()
    loss = Loss(pred_labels, train_labels)
    
    return loss

def cal_gradient(model):
    params = list(model.parameters())
    ans = 0
    for param in params:
        ans += torch.norm(param.grad) ** 2
    return (ans).to(torch.device('cpu'))

def extract_gradient(model):
    gradients = []
    params = list(model.parameters())
    for param in params:
        gradients.append(param.grad.clone().to(device))
    return gradients

def extract_param(model):
    gradients = []
    params = list(model.parameters())
    for param in params:
        gradients.append(param.detach().clone())
    return gradients

def cal_norm(gradients):
    ans = 0
    for gradient in gradients:
        ans += torch.norm(gradient) ** 2
    return (ans ** 0.5).to(torch.device('cpu'))

def cal_lf(labels, dataloader, model, Loss, opt):
    nums = 100
    lf = 0
    input_nodes, output_nodes, blocks = next(iter(dataloader))
    blocks = [b.to(device) for b in blocks]
    for _ in range(nums):
        model.reset_parameters()
        params1 = extract_param(model)
        loss = cal_loss(model, Loss, blocks, output_nodes, labels)
        loss.backward()
        gradients1 = extract_gradient(model)
        
        opt.zero_grad()
        model.reset_parameters()
        params2 = extract_param(model)
        loss = cal_loss(model, Loss, blocks, output_nodes, labels)
        loss.backward()
        gradients2 = extract_gradient(model)
        
        for i in range(gradients1.__len__()):
            gradients1[i] -= gradients2[i]
        for i in range(params1.__len__()):
            params1[i] -= params2[i]
        lf += cal_norm(gradients1) / cal_norm(params1)
    return (lf/nums).item()

def cal_m(labels, dataloader, Models, Loss, Opts, split_list):
    idx = 0
    gradients_full, gradients_partition = [], []
    for input_nodes, output_nodes, blocks in dataloader:
        model, opt = Models[idx], Opts[idx]

        blocks = [b.to(device) for b in blocks]
        tmp_loss = cal_loss(model, Loss, blocks, output_nodes, labels)
        tmp_loss.backward()
        gradients = extract_gradient(model)
        if gradients_full.__len__()==0:
            gradients_full = gradients
        else:
            for i in range(gradients_full.__len__()):
                gradients_full[i] += gradients[i]

        blocks[0].srcdata['feat'][:, :] = 0
        blocks[0].srcdata['feat'][:, split_list[idx]:split_list[idx+1]] = node_features[input_nodes, split_list[idx]:split_list[idx+1]]
        blocks = [b.to(device) for b in blocks]
        opt.zero_grad()
        tmp_loss = cal_loss(model, Loss, blocks, output_nodes, labels)
        tmp_loss.backward()
        gradients = extract_gradient(model)
        if gradients_partition.__len__()==0:
            gradients_partition = gradients
        else:
            for i in range(gradients_partition.__len__()):
                gradients_partition[i] += gradients[i]
        idx += 1
        if idx == num_workers:
            idx = 0
            break
    for i in range(gradients_full.__len__()):
        gradients_full[i] -= gradients_partition[i]
        gradients_full[i] /= num_workers
    return cal_norm(gradients_full)**2

dist.init_process_group('nccl', init_method='tcp://www.gitd245.online:7002', timeout=datetime.timedelta(seconds=10), rank=0, world_size=2)
rank = dist.get_rank()
device = f'cuda:{rank}'
local_rank = os.environ['LOCAL_RANK']
master_addr = os.environ['MASTER_ADDR']
master_port = os.environ['MASTER_PORT']
print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
torch.cuda.set_device(rank)
d_name = 'ogbn-arxiv'
dataset = DglNodePropPredDataset(name = d_name)
evaluator = Evaluator(name = d_name)
split_idx = dataset.get_idx_split()
graph, labels = dataset[0]

# dataset = RedditDataset()
# evaluator = None
# graph = dataset[0]
# labels = graph.ndata['label'].reshape(-1, 1)
# node_feature = graph.ndata['feat']
# tmp = torch.arange(0, node_feature.shape[0])
# split_idx = {'train': tmp[graph.ndata['train_mask']], 'valid': tmp[graph.ndata['val_mask']], 'test': tmp[graph.ndata['test_mask']]}

graph.add_edges(*graph.all_edges()[::-1])
graph = graph.remove_self_loop().add_self_loop()
num_epochs, num_hidden, num_layers, dropout, lr = 50, 256, 2, 0.5, 0.005
batch_size = 1024
max_time = 60 * 1

node_features = graph.ndata['feat']

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
# Model = StochasticGATNet(num_input, num_hidden, num_output, num_layers, 4, dropout)
Loss = F.nll_loss

models, opts = [], []
num_workers = 2
for i in range(num_workers):
    models.append(StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout).to(device))
    # models.append(StochasticGATNet(num_input, num_hidden, num_output, num_layers, 4, dropout))
    opts.append(torch.optim.AdamW(models[i].parameters(), lr=lr))

sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
dataloader = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    # batch_size = (node_features[split_idx['train']].shape[0] + num_workers - 1) // num_workers,
    batch_size = batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0)

def cal_compute_time(Model, Opt, blocks, labels, output_nodes, Loss):
    nums = 100
    labels = labels.to(device)
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

if rank == 0:
    input_nodes, output_nodes, blocks = next(iter(dataloader))
    blocks = [b.to(device) for b in blocks]
    cal_compute_time(models[0], opts[0], blocks, labels, output_nodes, Loss)
    compute_time = cal_compute_time(models[0], opts[0], blocks, labels, output_nodes, Loss)
    sample_time = cal_sample_time(dataloader)

    print(f'sample_time: {sample_time}')
    print(f'compute_time: {compute_time}')

file = open('./savedata/baselineTest.txt', mode = 'w')

# lf = 0
if rank == 0:
    lf = cal_lf(labels, dataloader, models[0], Loss, opts[0])

loss_list, train_list, valid_list, test_list, pltx, overhead, iteration = dist_run(graph, labels, dataloader, split_idx, evaluator, num_epochs, models, Loss, opts, lr, False)
if rank == 0:
    plt.subplot(1, 2, 1)
    plt.plot(pltx, loss_list)
    plt.subplot(1, 2, 2)
    plt.plot(pltx, test_list)
    file.write(str(loss_list)+'\n')
    file.write(str(test_list)+'\n')
    file.write(str(pltx)+'\n')
    file.write(str(overhead) + ' ' + str(iteration))
    file.write('\n')
    plt.savefig("./image/distTest.png")

print(f'------------rank:{rank} {test_list[-1]}------------')
