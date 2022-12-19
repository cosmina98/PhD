

import networkx as nx
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from torch_geometric.utils.convert import from_networkx
import torch 
class NxGraphstoPyggraphs(BaseEstimator, TransformerMixin):


    def fit(self, X,y=None):
        return self
    def transform( X, y=None):
       list_of_pyg=[]
       X=deepcopy(X)
       for i , g in enumerate(X):

            s=from_networkx(g)
            s.num_nodes=s.x.shape[0]
            s.num_edges=(s.edge_index[1])/2
            if y is  None:
                s.y=None
            else: 
               s.y=torch.tensor(y[i],dtype=torch.int64)
            list_of_pyg.append(s)
       return list_of_pyg