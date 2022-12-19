
import torch
from torch import nn
from torch_scatter import scatter_add
from torch.nn import Sequential, Linear, ReLU,Parameter
from torch_geometric.utils import add_self_loops, degree
from Convs.MessagePass import MessagePass

class GIN1(MessagePass):
    
    def __init__(self, in_channels, out_channels,epsilon=0.0):
        super().__init__()  
        #initialisation
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.out_channels=out_channels
        self.in_channels=in_channels
        self.hid_dim=64
        self.epsilon=epsilon
        self.mlp=MLP(self.in_channels,self.out_channels,self.hid_dim)
        self.reset_parameters()


    def reset_parameters(self):
        self.lin.reset_parameters()
       

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        num_nodes=len(x)
        # Step 1: Linearly transform node feature matrix.
        #first term
        first_term = torch.mul((1+self.epsilon),x )
        
        # Step 4-5: Start propagating messages.

        out = self.propagate(x=x,edge_index=edge_index)
        
        out=first_term + out
        out= self.mlp(out)
        return out

    #'not needed'
    def update(self, aggr_out):
    # aggr_out has shape [num_nodes, out_channels]

    # Step 5: Return new node embeddings.
        return aggr_out



class MLP(nn.Module):
  def __init__(self,input_shape,output_shape,hid_dim):
    super(MLP,self).__init__()
    self.fc1 = nn.Linear(input_shape,32)
    self.fc2 = nn.Linear(32,output_shape)
  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.sigmoid(self.fc2(x))
    return x




