import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import dgl
import numpy as np
import time

from util.model import StochasticSAGE, StochasticGATNet
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

dist.init_process_group('nccl', init_method='env://')

rank = dist.get_rank()
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
graph.add_edges(*graph.all_edges()[::-1])
graph = graph.remove_self_loop().add_self_loop()

dgl.distributed.initialize("output/ip_config.json")
# dgl.distributed.partition_graph(graph, "graph111", 1, "output/")
graph = dgl.distributed.DistGraph("graph111", part_config='output/graph111.json')

num_epochs, num_hidden, num_layers, dropout, lr = 10, 256, 2, 0.5, 0.005
batch_size = 1024
max_time = 60 * 1
node_features = graph.ndata['feat']
num_nodes = graph.num_nodes()
num_input, num_output = node_features.shape[1], int(labels.max().item()+1)
neighbor_samples = [5, 10]

model = StochasticSAGE(num_input, num_hidden, num_output, num_layers, dropout).cuda()
def sample(seeds):
    blocks = []
    seeds = torch.LongTensor(np.asarray(seeds))
    for i in range(num_layers):
        frontier = dgl.distributed.sample_neighbors(graph, seeds, neighbor_samples[i])
        blocks.append(dgl.to_block(frontier, seeds))
        seeds = blocks[-1].srcdata['_ID']
    blocks = blocks[::-1]
    return blocks

dataloader = dgl.distributed.DistDataLoader(dataset=torch.tensor([i for i in range(graph.num_nodes())]), batch_size=1000, collate_fn=sample, shuffle=True)

# if rank == 0:
#     blocks = [next(iter(dataloader))]
# else:
#     blocks = [None]
# t1 = time.perf_counter()
# dist.broadcast_object_list(blocks, src=0)
# t2 = time.perf_counter()
# print(f'rank{rank}:{t2-t1}')
# print(f'rank={rank}\nblocks={blocks}')

#   torchrun --nproc_per_node=2 test.py
# for epoch in range(num_epochs):
#     for iteration in range((num_nodes+batch_size-1)//batch_size):
#         if rank == 0:
#             blocks = [next(iter(dataloader))]
#         else:
#             blocks = [None]
#         t1 = time.perf_counter()
#         dist.broadcast_object_list(blocks, src=0)
#         t2 = time.perf_counter()
#         blocks = blocks[0]
#         # print(f'rank{rank}:{t2-t1}')
        
#         dist.barrier()
#         input_ID = blocks[0].srcdata['_ID']
#         print(type(node_features))
#         print(node_features[:][:100])

#         break
#     break
for p in model.parameters():
    print(f'before: {p}, rank: {rank}')
    dist.all_reduce(p, op=dist.ReduceOp.AVG)
    dist.barrier()

for p in model.parameters():
    print(f'after: {p}, rank: {rank}')
    dist.barrier()
