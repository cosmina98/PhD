
import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor
from Convs.MessagePass import MessagePass

class TAGConv1(MessagePass):
    def __init__(self, in_channels: int, out_channels: int, K: int = 3,
                 bias: bool = True, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalize = normalize
        self.aggr='add'

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False) for _ in range(K + 1)
        ])

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    improved=False, add_self_loops=False, flow=self.flow,
                    dtype=x.dtype)

            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    add_self_loops=False, flow=self.flow, dtype=x.dtype)

        out = self.lins[0](x)
        for lin in self.lins[1:]:
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight,
                              )
            out = out + lin.forward(x)

        if self.bias is not None:
            out = out + self.bias

        return out

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
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')