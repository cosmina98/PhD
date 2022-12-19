from collections.abc import Iterable
import numpy as np
import networkx as nx
from numpy.linalg import inv,multi_dot
from scipy.linalg import expm
class Shortest_Path():

        def __init__(self,normalise=True):
              
              self.phi=None
              self.normalise=normalise
              self.k=None
              self.graphs_list1=None
              self.fitted=False


        def compare(self,g_1, g_2):
                """Compute the kernel value (similarity) between two graphs.
                Parameters
                ----------
                g1 : networkx.Graph
                    First graph.
                g2 : networkx.Graph
                    Second graph.
                Returns
                -------
                k : The similarity value between g1 and g2.
                """
                # Diagonal superior matrix of the floyd warshall shortest
                # paths:
              
                s1 = np.array(nx.floyd_warshall_numpy(g_1))
                s1 = np.triu(s1, k=1)
                count1,val1 = np.unique(s1, return_counts=True)
                s2 = np.array(nx.floyd_warshall_numpy(g_2))
                s2 = np.triu(s2, k=1)
                count2,val2 =  np.unique(s2, return_counts=True)

                # Copy into arrays with the same length the non-zero shortests
                # paths:
                e1 = np.zeros(max(len(val1), len(val2)) - 1)
                e1[range(0, len(val1)-1)] = val1[1:]
                e2 = np.zeros(max(len(val1), len(val2)) - 1)
                e2[range(0, len(val2)-1)] = val2[1:]
                #print(e1,e2)
                self.phi=np.unique([[e1],[e2]],axis=0)
                #print(self.phi)
                k=np.sum(e1 * e2)
                return k

        def fit(self,graphs_list1):
                    if  not  isinstance(graphs_list1, list):
                          graphs_list1=[graphs_list1]
                    self.graphs_list1= graphs_list1
                    self.fitted=True
                    return self

        def transform(self,graphs_list):
            if  not self.fitted:
                     raise Exception("X Not fited")
            else:
                if not  isinstance(graphs_list, list):
                    graphs_list=[graphs_list]
                km= np.zeros((len(graphs_list), len(self.graphs_list1)))
                for  i,g_1 in enumerate(graphs_list):
                    for  j,g_2 in enumerate(self.graphs_list1):
                         if self.normalise:
                            km[i][j]= float(self.compare(g_1, g_2) )/ float(np.sqrt(self.compare(g_1, g_1) *
                                                             self.compare(g_2, g_2)))
                         else:  
                            km[i][j]=self.compare(g_1,g_2)
                return km

        def fit_transform(self,X):
             self.fit(X)
             return  self.transform(X)



        def get_phi(self):
              return self.phi


        def __call__(self):
              pass