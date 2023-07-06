from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
import dgl
import torch
from dgl.dataloading import GraphDataLoader

def _collate_fn(batch):
    # 小批次是一个元组(graph, label)列表
    graphs = [e[0] for e in batch]
    g = dgl.batch(graphs)
    labels = [e[1] for e in batch]
    labels = torch.stack(labels, 0)
    return g, labels

# Download and process data at './dataset/ogbg_molhiv/'
dataset = DglGraphPropPredDataset(name = 'ogbg-molhiv')
split_idx = dataset.get_idx_split()
# dataloaderS
data = GraphDataLoader(dataset, batch_size=1, collate_fn=_collate_fn)

cnt, graph_size, feature_size = 0, 0, 0

for g, labels in data:
    t = g.edges()
    n_data = g.ndata['feat']
    e_data = g.edata['feat']
    graph_size += t[0].element_size() * t[0].nelement() * 2
    feature_size += n_data.element_size() * n_data.nelement()
    feature_size += e_data.element_size() * e_data.nelement()
    feature_size += labels.element_size() * labels.nelement()
    cnt += 1

print(f'cnt: {cnt}\ngraph_size: {graph_size}\nfeature_size: {feature_size}\n')
