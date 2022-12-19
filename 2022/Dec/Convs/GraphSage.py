import torch
from torch import nn
from torch_scatter import scatter_add
from torch.nn import Sequential, Linear, ReLU,Parameter
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_mean
from Convs.MessagePass import MessagePass

class GraphSage1(MessagePass):
    
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')  
        #initialisation
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.out_channels=out_channels
        self.reset_parameters()
        self.W1 = nn.Parameter(torch.zeros(size=(in_channels, out_channels))) #xavier paramiter inizializator
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
      
    def reset_parameters(self):
        self.lin.reset_parameters()
       

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        num_nodes=len(x)
        # Step 1: Linearly transform node feature matrix.
        #first term
        first_term = self.lin(x)
        second_term= self.propagate(x,edge_index)
        out=first_term + self.lin(second_term)
        return out

    
    #'not needed'
    def update(self, aggr_out):
    # aggr_out has shape [num_nodes, out_channels]

    # Step 5: Return new node embeddings.
        return aggr_out