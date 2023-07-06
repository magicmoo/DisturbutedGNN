import torch
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean'))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hid_feats))
        for _ in range(num_layers - 2):
            self.convs.append(dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean'))
            self.bns.append(nn.BatchNorm1d(hid_feats))
        self.convs.append(dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean'))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, g, h):
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(g, h)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](g, h)
        return F.log_softmax(h, dim=-1)

class GATNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=3, heads=8, dropout=0.5):
        super(GATNet,self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(dglnn.GATConv(in_dim, hidden // heads, heads, dropout))
        self.bns.append(nn.BatchNorm1d(hidden))
        
        for _ in range(self.num_layers - 2):
            self.convs.append(dglnn.GATConv(hidden, hidden // heads, heads, dropout))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.convs.append(dglnn.GATConv(hidden,  out_dim, 1, dropout))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, g, h):
        for i in range(self.num_layers - 1):
            h = self.convs[i](g, h)
            h = self.flatten(h)
            h = self.bns[i](h)
        h = self.convs[-1](g, h)
        h = self.flatten(h)
        h =  F.log_softmax(h, dim=1)
        return h

class StochasticSAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean'))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hid_feats))
        for _ in range(num_layers - 2):
            self.convs.append(dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean'))
            self.bns.append(nn.BatchNorm1d(hid_feats))
        self.convs.append(dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean'))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, blocks, h):
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(blocks[i], h)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](blocks[-1], h)
        return F.log_softmax(h, dim=-1)
    
    def cal(self, g, h):
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(g, h)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](g, h)
        return F.log_softmax(h, dim=-1)
    
class StochasticGATNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=3, heads=8, dropout=0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(dglnn.GATConv(in_dim, hidden // heads, heads, dropout))
        self.bns.append(nn.BatchNorm1d(hidden))
        
        for _ in range(self.num_layers - 2):
            self.convs.append(dglnn.GATConv(hidden, hidden // heads, heads, dropout))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.convs.append(dglnn.GATConv(hidden,  out_dim, 1, dropout))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, blocks, h):
        for i in range(self.num_layers - 1):
            h = self.convs[i](blocks[i], h)
            h = self.flatten(h)
            h = self.bns[i](h)
        h = self.convs[-1](blocks[-1], h)
        h = self.flatten(h)
        h =  F.log_softmax(h, dim=1)
        return h
    
    def cal(self, g, h):
        for i in range(self.num_layers - 1):
            h = self.convs[i](g, h)
            h = self.flatten(h)
            h = self.bns[i](h)
        h = self.convs[-1](g, h)
        h = self.flatten(h)
        h =  F.log_softmax(h, dim=1)
        return h