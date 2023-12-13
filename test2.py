import dgl
from dgl.data import karate

# 加载Karate Club数据集
dataset = karate.KarateClubDataset()

# 获取第一个图
g = dataset[0]

# 打印图的基本信息
print(f"Number of nodes: {g.number_of_nodes()}")
print(f"Number of edges: {g.number_of_edges()}")

# 打印节点特征和标签
print(f"Node features:\n{g.ndata}")
print(f"Node labels:\n{g.ndata['label']}")

# 打印边的信息
print(f"Edge source nodes: {g.edges()[0]}")
print(f"Edge destination nodes: {g.edges()[1]}")
