import torch
from torch import nn
from torch_scatter import scatter_add
from torch.nn import Sequential, Linear, ReLU,Parameter
from torch_geometric.utils import add_self_loops, degree
from Convs.MessagePass import MessagePass

class GCNConv1(MessagePass):
    
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  
        #initialisation
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        num_nodes=len(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out= self.propagate(x,edge_index,norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

   
    def message(self, x_j,norm):
        # Step 4: Normalize node features.
        if norm!= None:
            return norm.view(-1, 1) * x_j
        else: return x_j
    
    #'not needed'
    def update(self, aggr_out):
    # aggr_out has shape [num_nodes, out_channels]

    # Step 5: Return new node embeddings.
        return aggr_out