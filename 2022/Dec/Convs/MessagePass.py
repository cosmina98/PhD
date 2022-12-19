from torch_scatter import scatter
import torch
from torch import nn
from torch_scatter import scatter_add,scatter_mean
import os
class MessagePass(torch.nn.Module):
    """
    Args:
    dim -direction along  each to agregate 
    """
    def __init__(
        self,
        aggr="add",
        node_dim=0,
        flow: str = "source_to_target",
        **kwargs
    ):
        super().__init__()
        if aggr == "add":
            self.aggr=self.sum_aggregate
        elif aggr=="mean":
            self.aggr=self.mean_aggregate
        elif aggr=="max":
            self.aggr=self.max_aggregate
        elif aggr=="min":
            self.aggr=self.min_aggregate
        elif aggr=="mul":
            self.aggr=self.mul_aggregate
        self.node_dim=node_dim
        self.flow=flow
        
    def propagate(self ,x,edge_index,**kwargs):
        out=self.collect(x,edge_index)
        self._x_j=out['x_j']
        self._x_i=out['x_i']
        self._edge_index_i= out['edge_index_i']
        self._edge_index_j= out['edge_index_j']
        message_and_aggregate = getattr(self, "message_and_aggregate", None)
        if callable(message_and_aggregate):
            out=self.message_and_aggregate(x, edge_index,**kwargs)    
        else:
            out = self.message(self._x_j,**kwargs)
            out= self.aggr(out,edge_index)
        return out


    def message(self,x_j, **kwarg):
        return x_j
    
    def collect(self ,x,edge_index):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        out={}
        # 2. construct message x_j, x_i. Both with shape [num_edge, embed_size]
        out['x_j'] = x.index_select(0, edge_index[i]) 
        out['x_i'] = x.index_select(0, edge_index[j])
        out['edge_index_i'] = edge_index[i] # Source node edges 
        out['edge_index_j'] = edge_index[j]   # Target node edges 
        return out  
    
    def sum_aggregate(self,x,edge_index):
        row,col=edge_index
        tmp = torch.index_select(x, 0, col) # shape [num_edges, embed_size)
        aggr2 = scatter_add(tmp,row,0)
        return aggr2

    
    def mean_aggregate(self,x,edge_index):
        row,col=edge_index
        tmp = torch.index_select(x, 0, col) # shape [num_edges, embed_size)
        aggr2 = scatter_mean(tmp,row,0)
        return aggr2
    
    
    def min_aggregate(self, x, edge_index):
        row,col=edge_index
        tmp = torch.index_select(x, 0, col) # shape [num_edges, embed_size)
        aggr2 = scatter(tmp,row,0,reduce='min')
        return aggr2
    
    
    def max_aggregate(self, x, edge_index):
        row,col=edge_index
        tmp = torch.index_select(x, 0, col) # shape [num_edges, embed_size)
        aggr2 = scatter(tmp,row,0,reduce='max')
        return aggr2  
    
    def mul_aggregate (self, x, edge_index):
        row,col=edge_index
        tmp = torch.index_select(x, 0, col) # shape [num_edges, embed_size)
        aggr2 = scatter(tmp,row,0,reduce='mul')
        return aggr2
    
   # def message_and_aggregate(self):
       
        #raise NotImplementedError

    
    