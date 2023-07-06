from ogb.nodeproppred import DglNodePropPredDataset

print("-----------------------------------")
dataset = DglNodePropPredDataset(name='ogbn-arxiv')
print("-----------------------------------")
split_idx = dataset.get_idx_split()

# there is only one graph in Node Property Prediction datasets
# 在Node Property Prediction数据集里只有一个图

cnt, graph_size, feature_size = 0, 0, 0
g, labels = dataset[0]

t = g.edges()
n_data = g.ndata['feat']
n2_data = g.ndata['year']
graph_size += t[0].element_size() * t[0].nelement() * 2
feature_size += n_data.element_size() * n_data.nelement()
feature_size += n2_data.element_size() * n2_data.nelement()

print(f'graph_size: {graph_size}\nfeature_size: {feature_size}\n')