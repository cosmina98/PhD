from collections.abc import Iterable
import numpy as np
import networkx as nx
from numpy.linalg import inv,multi_dot
from scipy.linalg import expm
from Kernels.utils import get_a_floyd_S_graph

class Shortest_Path_Labelled():

        def __init__(self,normalise=True,node_label='node_label'):
              self.node_label=node_label
              self.phi=None
              self.normalise=normalise
              self.k=None
              self.graphs_list1=None
              self.fitted=False


        def compare(self,x, y):
            
            """Calculate shortests paths on attributes.
            Parameters
            ----------
            x, y : tuple
                Tuples of shortest path matrices and their attribute
                dictionaries.
            Returns
            -------
            kernel : number
                The kernel value.
            """
            # Initialise
            Sx, phi_x = x[0],x[1]
            Sy, phi_y = y[0],y[1]
            kernel = 0
            dimx = Sx.shape[0]
            dimy = Sy.shape[0]
            #matrix1 i row
            for i in range(dimx):
                #matrix1 j columns
                for j in range(dimx):
                    #skip the diagonal as it s 0
                    if i == j:
                        continue
                    #else for rows k in second matrix
                    for k in range(dimy):
                        #for   columns  m  second  matrix 
                        for m in range(dimy):
                            #agin skip the main diagonal 
                            if k == m:
                                continue
                            #else 
                            #if there s a matching distance at that entry, and it s not infinity
                            if (Sx[i, j] == Sy[k, m] and
                                    Sx[i, j] != float('Inf')):
                                #add the dot product  between the node labels to the final kernel
                                kernel = kernel + np.dot(phi_x[i], phi_y[k]) * \
                                    np.dot(phi_x[j], phi_y[m])
            return kernel
        

        def fit(self,graphs_list1):
                if  not  isinstance(graphs_list1, list):
                      graphs_list1=[graphs_list1]
                self.graphs_list1= graphs_list1
                self.fitted=True
                return self
          
        def transform(self,graphs_list2):
                 if  not self.fitted:
                         raise Exception("X Not fiited")
                 graphs_list1=self.graphs_list1
                 if not  isinstance(graphs_list1, list):
                  graphs_list1=[graphs_list1]
                 if   not  isinstance(graphs_list2, list):
                  graphs_list2=[graphs_list2]
                 self.kernel_matrix= np.zeros((len(graphs_list2), len(graphs_list1)))
                 for  i,g_1 in enumerate(graphs_list2):
                   for  j,g_2 in enumerate(graphs_list1):
                     dict1=nx.get_node_attributes(g_1, self.node_label)
                     dict2=nx.get_node_attributes(g_2, self.node_label)
                     adj1=np.array(nx.floyd_warshall_numpy(g_1))
                     adj2=np.array(nx.floyd_warshall_numpy(g_2))
                     x=adj1,dict1
                     y=adj2,dict2
                     if self.normalise:
                        self.kernel_matrix[i][j]= float(self.compare(x, y) )/ float(np.sqrt(self.compare(x, x) *
                                                         self.compare(y, y)))
                     else:  
                        self.kernel_matrix[i][j]=self.compare(x,y)
                 return self.kernel_matrix

        def fit_transform(self,X):
             self.fit(X)
             return  self.transform(X)
             
        def get_phi(self):
              return self.phi


        def __call__(self):
              pass