import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.conv import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)
        self.h_ = list()

    def forward(self, graph, feat, eweight=None):
        h = self.conv1(graph, feat)
        self.h_ = h
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
