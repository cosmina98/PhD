from torch_geometric.utils import softmax, add_remaining_self_loops
import torch
from torch import nn
from torch_scatter import scatter_add
from torch.nn import Sequential, Linear, ReLU,Parameter
from torch_geometric.utils import add_self_loops, degree
from Convs.MessagePass import MessagePass
import torch.nn.functional as F
from typing import Optional, Tuple
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor

class APPNP1(MessagePass):
    def __init__(self, K: int, alpha: float, dropout: float = 0.,
                 add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.node_dim=0
        self.aggr='add'
        
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)
        h = x
        for k in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h
        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, x, edge_index, edge_weight=None):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        row, col = edge_index[j],edge_index[i] 
        value = edge_weight
        adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(x.size(0), x.size(0)))
        fin=matmul(adj2.t(), x, reduce=self.aggr)
        return fin

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'