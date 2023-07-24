from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from util.model import StochasticSAGE, StochasticGATNet
from util.multiWorker import multi_Stochastic_run_graph, replaceModel, avgModel
from util.util import Stochastic_train, try_gpu, Stochastic_test
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


num_epochs, num_hidden, num_layers, dropout, lr = 30, 256, 2, 0.5, 0.001

node_features = graph.ndata['feat']

num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
# Model = StochasticGATNet(num_input, num_hidden, num_output, num_layers, 4, dropout)
Loss = F.nll_loss

models, opts = [], []
num_workers = 4
for i in range(num_workers):
    models.append(StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout))
    opts.append(torch.optim.AdamW(models[i].parameters(), lr=lr))

batch_size = 1024
sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
dataloader = dgl.dataloading.DataLoader(
    graph, split_idx['train'], sampler,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0)

split_list = [0]
num_features = node_features.shape[-1]
for i in range(num_workers):
    split_list.append(split_list[-1] + (num_features//num_workers))
    if i < num_features%num_workers:
        split_list[-1] += 1


def cal_gradient(Model, Loss, blocks, output_nodes, labels, Opt):
    Model.train()
    Opt.zero_grad()
    Model = Model.to(try_gpu())

    node_features = blocks[0].srcdata['feat']
    train_labels = labels[output_nodes].squeeze(1).to(try_gpu())
    pred_labels = Model(blocks, node_features)
    loss = Loss(pred_labels, train_labels)
    loss.backward()
    
    Model = Model.to(torch.device('cpu'))

def extract_gradient(model):
    gradients = []
    params = list(model.parameters())
    for param in params:
        gradients.append(param.grad.clone().to(try_gpu()))
    return gradients

def cal_norm(gradients):
    ans = 0
    for gradient in gradients:
        ans += torch.norm(gradient) ** 2
    return (ans ** 0.5).to(torch.device('cpu'))

def cal_diff_norm(models, opts, Loss, blocks, gradients2, input_nodes, output_nodes, type='average'):
    embedding = torch.zeros((output_nodes.shape[0], num_output)).to(try_gpu())
    for i, Model in enumerate(models):
        blocks = [b.to(torch.device('cpu')) for b in blocks]
        blocks[0].srcdata['feat'][:, :] = 0
        blocks[0].srcdata['feat'][:, split_list[i]:split_list[i+1]] = node_features[input_nodes, split_list[i]:split_list[i+1]]*num_workers
        blocks = [b.to(try_gpu()) for b in blocks]
        # loss += Stochastic_train(Model, Loss, blocks, output_nodes, labels, opts[i])/num_workers
        Model.train()
        Model = Model.to(try_gpu())
        embedding += Model.embedding(blocks, blocks[0].srcdata['feat'])

    train_labels = labels[output_nodes].squeeze(1).to(try_gpu())

    for i, Model in enumerate(models):
        opts[i].zero_grad()

    if type == 'average':
        embedding /= num_workers
    loss = Loss(F.log_softmax(embedding, dim=-1), train_labels)
    loss.backward()
    gradients1 = extract_gradient(models[0])
    for Model in models[1:]:
        gradients = extract_gradient(Model)
        for i in range(gradients.__len__()):
            gradients1[i] += gradients[i]
    for i in range(gradients1.__len__()):
        gradients1[i] -= gradients2[i]
    return cal_norm(gradients1)

step, type, is_correct, correct_step = 10, 'summing', True, 64
norm_list, diff_norm_list = [], []
cnt = 0
for epoch in range(num_epochs):
    for input_nodes, output_nodes, blocks in dataloader:
        cnt += 1
        blocks = [b.to(try_gpu()) for b in blocks]
        # loss = Stochastic_train(models[0], Loss, blocks, output_nodes, labels, opts[0])
        cal_gradient(models[0], Loss, blocks, output_nodes, labels, opts[0])
        gradients2 = extract_gradient(models[0])    
        norm_list.append(cal_norm(gradients2)/2)

        # Model = Model.to(torch.device('cpu'))
        diff_norm_list.append(cal_diff_norm(models, opts, Loss, blocks, gradients2, input_nodes, output_nodes, type))

        loss = 0
        for i, Model in enumerate(models):
            blocks = [b.to(torch.device('cpu')) for b in blocks]
            blocks[0].srcdata['feat'][:, :] = 0
            blocks[0].srcdata['feat'][:, split_list[i]:split_list[i+1]] = node_features[input_nodes, split_list[i]:split_list[i+1]]*num_workers
            blocks = [b.to(try_gpu()) for b in blocks]
            loss += Stochastic_train(Model, Loss, blocks, output_nodes, labels, opts[i])/num_workers
        avgModel(models)

        # if is_correct and cnt%correct_step==0:  #wrong
            
            

    if (epoch+1)%(num_epochs//step) == 0:
        train_acc, valid_acc, test_acc = Stochastic_test(models[0], graph, labels, split_idx, evaluator)
        print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
        print(f'loss: {loss:.6}')
        print(f'train_acc: {train_acc:.2}')
        print(f'valid_acc: {valid_acc:.2}')
        print(f'test_acc: {test_acc:.2}')

pltx = [i+1 for i in range(norm_list.__len__())]
plt.xlabel('iteration')
plt.ylabel('norm')

plt.plot(pltx, norm_list)
plt.plot(pltx, diff_norm_list)
plt.legend(['gradient', f'diff({type})'])
plt.savefig(f'./image/gradientTest({type}).jpg')