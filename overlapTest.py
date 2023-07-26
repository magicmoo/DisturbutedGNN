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

def cal_gradient(model):
    params = list(model.parameters())
    ans = 0
    for param in params:
        ans += torch.norm(param.grad) ** 2
    return (ans).to(torch.device('cpu'))

def cal_gradient2(model):
    params = list(model.parameters())
    ans = 0
    for param in params:
        ans += torch.norm(param.grad)
    return (ans).to(torch.device('cpu'))

def extract_gradient(model):
    gradients = []
    params = list(model.parameters())
    for param in params:
        gradients.append(param.grad.clone().to(try_gpu()))
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
    blocks = [b.to(try_gpu()) for b in blocks]
    for _ in range(nums):
        model.reset_parameters()
        params1 = extract_param(model)
        loss = cal_loss(model, Loss, blocks, output_nodes, labels)
        model = model.to(try_gpu())
        loss.backward()
        model = model.to(torch.device('cpu'))
        gradients1 = extract_gradient(model)
        
        opt.zero_grad()
        model.reset_parameters()
        params2 = extract_param(model)
        loss = cal_loss(model, Loss, blocks, output_nodes, labels)
        model = model.to(try_gpu())
        loss.backward()
        model = model.to(torch.device('cpu'))
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

        blocks = [b.to(try_gpu()) for b in blocks]
        tmp_loss = cal_loss(model, Loss, blocks, output_nodes, labels)
        model = model.to(try_gpu())
        tmp_loss.backward()
        model = model.to(torch.device('cpu'))
        gradients = extract_gradient(model)
        if gradients_full.__len__()==0:
            gradients_full = gradients
        else:
            for i in range(gradients_full.__len__()):
                gradients_full[i] += gradients[i]

        blocks = [b.to(torch.device('cpu')) for b in blocks]
        blocks[0].srcdata['feat'][:, :] = 0
        blocks[0].srcdata['feat'][:, split_list[idx]:split_list[idx+1]] = node_features[input_nodes, split_list[idx]:split_list[idx+1]]*num_workers
        blocks = [b.to(try_gpu()) for b in blocks]
        opt.zero_grad()
        tmp_loss = cal_loss(model, Loss, blocks, output_nodes, labels)
        model = model.to(try_gpu())
        tmp_loss.backward()
        model = model.to(torch.device('cpu'))
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

def run5(graph, labels, dataloader, split_idx, evaluator, num_epochs, Models, Loss, Opts, lr, is_output=False):
    node_features = graph.ndata['feat']
    num_workers, num_features = Models.__len__(), node_features.shape[-1]
    split_list = [0]
    for i in range(num_workers):
        split_list.append(split_list[-1] + (num_features//num_workers))
        if i < num_features%num_workers:
            split_list[-1] += 1

    loss_list, train_list, valid_list, test_list = [], [], [], []
    step = 10   # the step program output train's data
    idx, loss, cnt, correct_step = 0, 0.0, -1, 0
    time_now, time_now2, pltx, pltx2 = 0, 0, [], []
    gradient_w1, loss_w1, loss_w2 = 0, 0, 0
    m = cal_m(labels,dataloader, Models, Loss, Opts, split_list)
    for epoch in range(num_epochs):
        _loss = cal_loss2(labels, dataloader, Models, Loss, Opts)
        loss_list.append(_loss)
        train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
        train_list.append(train_acc)
        valid_list.append(valid_acc)
        test_list.append(test_acc)
        pltx.append(time_now/60)
        pltx2.append(time_now2/60)
        for input_nodes, output_nodes, blocks in dataloader:
            model, opt = Models[idx], Opts[idx]
            data = blocks[0].srcdata['feat']
            feature_time = data.element_size() * data.nelement() * (1-1/num_workers) / bandwidth
            if cnt < correct_step:
                feature_time = 0
                blocks = [b.to(torch.device('cpu')) for b in blocks]
                blocks[0].srcdata['feat'][:, :] = 0
                blocks[0].srcdata['feat'][:, split_list[idx]:split_list[idx+1]] = node_features[input_nodes, split_list[idx]:split_list[idx+1]]*num_workers

            blocks = [b.to(try_gpu()) for b in blocks]

            loss += Stochastic_train(model, Loss, blocks, output_nodes, labels, opt)/num_workers
            
            if cnt >= correct_step:
                gradient_w1 += cal_gradient(model)/num_workers

            idx = idx+1
            if idx == num_workers:
                _loss = loss
                # print(_loss)
                loss, idx = 0.0, 0
                avgModel(Models)
                parameter_time = 2 * cal_parameter_size(Models[0]) / bandwidth
                time_now += compute_time + sample_time + feature_time + parameter_time
                if cnt >= correct_step:
                    time_now2 += max(feature_time, correct_step*(compute_time+parameter_time))
                    cnt, loss_w1 = 0, _loss
                    loss_w2 = cal_loss2(labels, dataloader, Models, Loss, Opts)
                    s1 = (data.shape[0] * data.shape[1] * data.element_size() / compute_time / bandwidth * (2/lr*loss_w1+(lr*lf-1)*gradient_w1*num_workers))
                    s2 = (1-lr*lf)*gradient_w1*num_workers
                    m = cal_m(labels,dataloader, Models, Loss, Opts, split_list)
                    correct_step = min((s1/m) ** 0.5, s2/m)
                    correct_step = round(correct_step.item(), 0)
                    # correct_step = 0
                    print(f'correct_step: {correct_step}')
                    print(f's1: {s1}, s2: {s2}, m: {m}')
                    print(f'gradient_w1: {gradient_w1}, loss_w1: {loss_w1}, loss_w2: {loss_w2}, time_now: {time_now}')
                    print(f'time1: {time_now}, time2: {time_now2}')
                    print("---------------------")
                    gradient_w1, loss_w1, loss_w2 = 0, 0, 0
                else:
                    cnt += 1

        # print(f'time_now: {time_now}')

        if time_now > max_time:
            break

        if is_output and (epoch+1)%(num_epochs//step) == 0:
            train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {_loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}')
    _loss = cal_loss2(labels, dataloader, Models, Loss, Opts)
    loss_list.append(_loss)
    train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
    train_list.append(train_acc)
    valid_list.append(valid_acc)
    test_list.append(test_acc)
    pltx.append(time_now/60)
    pltx2.append(time_now2/60)
    return loss_list, train_list, valid_list, test_list, pltx, pltx2

d_name = 'ogbn-papers100M'
dataset = DglNodePropPredDataset(name = d_name)
evaluator = Evaluator(name = d_name)
split_idx = dataset.get_idx_split()
graph, labels = dataset[0]
# graph.add_edges(*graph.all_edges()[::-1])
# graph = graph.remove_self_loop().add_self_loop()
num_epochs, num_hidden, num_layers, dropout, lr = 50000, 256, 2, 0.5, 0.005

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
bandwidth2 = 25 * 1000 * 1000 * 1000 / 8
bandwidth3 = 40 * 1000 * 1000 * 1000 / 8
bandwidth4 = 100 * 1000 * 1000 * 1000 / 8

input_nodes, output_nodes, blocks = next(iter(dataloader))
blocks = [b.to(try_gpu()) for b in blocks]
cal_compute_time(models[0], opts[0], blocks, labels, output_nodes, Loss)
compute_time = cal_compute_time(models[0], opts[0], blocks, labels, output_nodes, Loss)
sample_time = cal_sample_time(dataloader)

print(f'sample_time: {sample_time}')
print(f'compute_time: {compute_time}')

max_time = 60 * 0.5

plt.subplot(1, 2, 1)
plt.xlabel('Wall-clock time(min)')
plt.ylabel('loss')
plt.subplot(1, 2, 2)
plt.xlabel('Wall-clock time(min)')
plt.ylabel('test_acc')  

file = open('./savedata/overlapTest.txt', mode = 'w')

lf = cal_lf(labels, dataloader, models[0], Loss, opts[0])

loss_list, train_list, valid_list, test_list, pltx, pltx2 = run5(graph, labels, dataloader, split_idx, evaluator, num_epochs, models, Loss, opts, lr, False)
plt.subplot(1, 2, 1)
plt.plot(pltx, loss_list)
plt.plot(pltx2, loss_list)
plt.subplot(1, 2, 2)
plt.plot(pltx, test_list)
plt.plot(pltx2, test_list)
file.write(str(loss_list)+'\n')
file.write(str(test_list)+'\n')
file.write(str(pltx)+'\n')
file.write(str(pltx2)+'\n')
file.write('\n')
print(f'------------(6):{test_list[-1]}------------')

plt.legend(['no_overlap', 'overlap'])
plt.savefig('./image/overlapTest.jpg')