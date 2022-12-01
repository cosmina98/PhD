from torch_geometric.utils import softmax, add_remaining_self_loops
import torch
from torch import nn
from torch_scatter import scatter_add
from torch.nn import Sequential, Linear, ReLU,Parameter
from torch_geometric.utils import add_self_loops, degree
from Convs.MessagePass import MessagePass
import torch.nn.functional as F

class SSGConv1(MessagePass):
   
    def __init__(self, in_channels, out_channels,alpha=0.05, K = 1,
                  add_self_loops = True,
                 bias  = True, dropout = 0.05, **kwargs):
        super(SSGConv1, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.alpha=alpha
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight):
        if self.add_self_loops:
            edge_index, edge_weight = add_self_loops(edge_index,edge_weight)
       
       
        output = self.alpha * x
        for k in range(self.K):
            x = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
            output = output + (1. / self.K) * x
        x = output
        return self.lin(x)

    def message(self, x_j, edge_weight) :
        return edge_weight.view(-1, 1) * x_j


    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)