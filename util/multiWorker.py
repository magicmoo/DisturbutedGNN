import torch
from util import try_gpu, Stochastic_train, Stochastic_test

@torch.no_grad()
def avgModel(models):
    for model in models:
        model.eval()

    params = list(models[0].parameters())

    for model in models[1:]:
        params2 = list(model.parameters())
        for i, param in enumerate(params):
            param += params2[i]
    
    num_models = models.__len__()
    for param in params:
        param /= num_models

    for model in models[1:]:
        params2 = list(model.parameters())
        for i, param in enumerate(params):
            params2[i][:] = param[:]
            
def multi_Stochastic_run_graph(graph, labels, dataloader, split_idx, evaluator, num_epochs, Model, Loss, Opt, is_output=False):
    loss_list = []
    step = 10   # the step program output train's data
    for epoch in range(num_epochs):
        loss = 0
        for _, output_nodes, blocks in dataloader:
            blocks = [b.to(try_gpu()) for b in blocks]
            loss += Stochastic_train(Model, Loss, blocks, output_nodes, labels, Opt)
        
        loss_list.append(loss)
        if is_output and (epoch+1)%(num_epochs//step) == 0:
            train_acc, valid_acc, test_acc = Stochastic_test(Model, graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}') 
    train_acc, valid_acc, test_acc = Stochastic_test(Model, graph, labels, split_idx, evaluator)
    return loss_list, train_acc, valid_acc, test_acc
