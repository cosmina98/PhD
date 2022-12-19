from torch_geometric.utils import softmax, add_remaining_self_loops
import torch
from torch import nn
from torch_scatter import scatter_add
from torch.nn import Sequential, Linear, ReLU,Parameter
from torch_geometric.utils import add_self_loops, degree
from Convs.MessagePass import MessagePass
import torch.nn.functional as F

class GATLayer(MessagePass):
    
    def __init__(self, in_channels, out_channels, alpha, dropout=0.0):
        super().__init__(aggr='add')  
        #initialisation
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.out_channels=out_channels
        self.drop_prob = dropout
        self.leakrelu = nn.LeakyReLU(alpha)
        self.a = nn.Parameter(torch.zeros(size=(2*out_channels, 1)))
        nn.init.xavier_uniform_(self.a)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        #self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        num_nodes=len(x)
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_remaining_self_loops(edge_index)

        # Step 2: Linearly transform node feature matrix.
        z = self.lin(x)

        # Step 4: Start propagating messages.
        self.propagate(z,edge_index,alpha=1)
        
        #Step 5 Compute the attention score 
        e = self.leakrelu(torch.matmul((torch.cat([self._x_i, self._x_j], dim=-1)), self.a))
        
        #Step 6 Normalise the attention score
        alpha = softmax(e, self._edge_index_i)
        alpha = F.dropout(alpha, self.drop_prob, self.training)
        
        out=self.propagate(x=z,edge_index=edge_index, alpha=alpha)

        return out

   
    def message(self, x_j,alpha):
        # x_j has shape [E, out_channels]
        return x_j * alpha

    #'not needed'
    def update(self, aggr_out):
    # aggr_out has shape [num_nodes, out_channels]

    # Step 5: Return new node embeddings.
        return aggr_out
    
    
    
class GAT1(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super(GAT1, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATLayer(num_features, self.hid, dropout=0.6,alpha=0.2)
        self.conv2 = GATLayer(self.hid*self.in_head, num_classes,
                             dropout=0.6,alpha=0.2)

    def forward(self, x, edge_index):
                
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)