import torch
from torch import nn
from torch_scatter import scatter_add
from torch.nn import Sequential, Linear, ReLU,Parameter
from torch_geometric.utils import add_self_loops, degree
from torch import Tensor
from torch_geometric.nn.inits import uniform
from Convs.MessagePass import MessagePass
class GatedGraphConv1(MessagePass):
    
    def __init__(self, in_channels, out_channels,num_layers=5,bias=True):
        super().__init__()  
        #initialisation
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.out_channels=out_channels
        self.num_layers = num_layers
        self.W = nn.Parameter(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.reset_parameters()


    def reset_parameters(self):
        uniform(self.out_channels, self.W)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index,edge_weight):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Check of embedding size is larger than the target output channels 
        #Otherwise error; Trying to create tensor with negative dimension 
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')
        #Fill x with 0 along the y axis to match the out_channels
        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)
   
        
        for i in range(self.num_layers):
            #Multiply x by their weights 
            m = torch.matmul(x, self.W[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
           # m = self.propagate(edge_index, x=m, edge_weight=edge_weight,
                               #size=None)

            m=self.propagate(m, edge_index,edge_weight=edge_weight)
            x = self.rnn(m, x)

        return x


    def message(self, x_j,edge_weight=None):
        # x_j has shape [E, out_channels]
        return  x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)
    
    #'not needed'
    def update(self, aggr_out):
    # aggr_out has shape [num_nodes, out_channels]

    # Step 5: Return new node embeddings.
        return aggr_out