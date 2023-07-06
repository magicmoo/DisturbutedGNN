import torch

def try_gpu(i=0):  #@save
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

@torch.no_grad()
def test(model, graph, labels, split_idx, evaluator):
    model.eval()
    features = graph.ndata['feat']
    y_pred = model(graph, features).argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': labels[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': labels[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': labels[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

@torch.no_grad()
def Stochastic_test(model, graph, labels, split_idx, evaluator):
    model.eval()
    
    features = graph.ndata['feat']
    y_pred = model.cal(graph, features).argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': labels[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': labels[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': labels[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def train(Model, Loss, graph, labels, train_idx, Opt):
    Model.train()
    Opt.zero_grad()

    node_features = graph.ndata['feat']
    pred_labels = Model(graph, node_features)
    pred_train = pred_labels[train_idx]
    # train_output = nn.functional.one_hot(labels[train_idx], num_classes=pred_train.shape[-1]).squeeze()
    train_output = labels[train_idx].squeeze(1)
    loss = Loss(pred_train, train_output)
    loss.backward()
    Opt.step()
    
    return loss.item()
    
def run_graph(graph, labels, split_idx, evaluator, num_epochs, Model, Loss, Opt, is_output=False):
    graph, labels = graph.to(try_gpu()), labels.to(try_gpu())

    loss_list = []
    step = 10   # the step program output train's data
    for epoch in range(num_epochs):
        loss = train(Model, Loss, graph, labels, split_idx['train'], Opt)
        loss_list.append(loss)
        if is_output and (epoch+1)%(num_epochs//step) == 0:
            train_acc, valid_acc, test_acc = test(Model, graph, labels, split_idx, evaluator)
            print(f'---------------------{(epoch+1)//(num_epochs//step)}---------------------')
            print(f'loss: {loss:.6}')
            print(f'train_acc: {train_acc:.2}')
            print(f'valid_acc: {valid_acc:.2}')
            print(f'test_acc: {test_acc:.2}')
    train_acc, valid_acc, test_acc = test(Model, graph, labels, split_idx, evaluator)
    return loss_list, train_acc, valid_acc, test_acc

def Stochastic_train(Model, Loss, blocks, output_nodes, labels, Opt):

    Model.train()
    Opt.zero_grad()

    Model = Model.to(try_gpu())
    node_features = blocks[0].srcdata['feat']
    train_labels = labels[output_nodes].squeeze(1).to(try_gpu())
    pred_labels = Model(blocks, node_features)
    # train_output = nn.functional.one_hot(labels[train_idx], num_classes=pred_train.shape[-1]).squeeze()
    loss = Loss(pred_labels, train_labels)
    loss.backward()
    Opt.step()
    
    Model = Model.to(torch.device('cpu'))
    return loss.item()

def Stochastic_run_graph(graph, labels, dataloader, split_idx, evaluator, num_epochs, Model, Loss, Opt, is_output=False):
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
