import torch
from util.util import try_gpu, Stochastic_train, Stochastic_test

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
            
def multi_Stochastic_run_graph(graph, labels, dataloader, split_idx, evaluator, num_epochs, Models, Loss, Opts, is_output=False):
    node_features = graph.ndata['feat']
    num_workers, num_features = Models.__len__(), node_features.shape[-1]
    split_list = [0]
    for i in range(num_workers):
        split_list.append(split_list[-1] + (num_features//num_workers))
        if i < num_features%num_workers:
            split_list[-1] += 1

    loss_list = []
    step = 10   # the step program output train's data
    tmp_features = torch.zeros(node_features.shape)
    tmp_features[:, :] = node_features[:, :]

    for epoch in range(num_epochs):
        loss = 0
        for i, Model in enumerate(Models):
            node_features[:, :] = 0
            node_features[:, split_list[i]:split_list[i+1]] = tmp_features[:, split_list[i]:split_list[i+1]]*num_workers
            for _, output_nodes, blocks in dataloader:
                blocks = [b.to(try_gpu()) for b in blocks]
                loss += Stochastic_train(Model, Loss, blocks, output_nodes, labels, Opts[i])

        # for _, output_nodes, blocks in dataloader:
        #     tmp_blocks = blocks.copy()
        #     for i, Model in enumerate(Models):
        #         blocks = [b.to(torch.device('cpu')) for b in blocks]
        #         for j, block in enumerate(blocks):
        #             block.srcdata['feat'][:, :] = 0
        #             block.srcdata['feat'][:, split_list[i]:split_list[i+1]] = tmp_blocks[j].srcdata['feat'][:, split_list[i]:split_list[i+1]]
        #             print(block.srcdata['feat'])
        #         blocks = [b.to(try_gpu()) for b in blocks]
        avgModel(Models)

        loss_list.append(loss)
        if is_output and (epoch+1)%(num_epochs//step) == 0:
            train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}') 
    train_acc, valid_acc, test_acc = Stochastic_test(Models[0], graph, labels, split_idx, evaluator)
    return loss_list, train_acc, valid_acc, test_acc
