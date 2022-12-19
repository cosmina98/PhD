from collections.abc import Iterable
import numpy as np
import networkx as nx
from numpy.linalg import inv,multi_dot
from scipy.linalg import expm

class Three_node_graphlet_kernel():

      def __init__(self,normalise=False,n_samples=200):
         
          self.phi_X=None
          self.phi_Y=None
          self.normalise=normalise
          self.k=None
          self.graphs_list1=None
          self.fitted=False
          self.n_samples=n_samples
          self.k_train=None
          self.graphlets=[nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]  
            
          self.graphlets[0].add_nodes_from(range(3))
        
          self.graphlets[1].add_nodes_from(range(3))
          self.graphlets[1].add_edge(0,1)
        
          self.graphlets[2].add_nodes_from(range(3))
          self.graphlets[2].add_edge(0,1)
          self.graphlets[2].add_edge(1,2)
            
          self.graphlets[3].add_nodes_from(range(3))
          self.graphlets[3].add_edge(0,1)
          self.graphlets[3].add_edge(1,2)
          self.graphlets[3].add_edge(0,2)

  
      def normalise_k( self,Kmatrix):
                    k_norm = np.zeros(Kmatrix.shape)
                    for i in range(Kmatrix.shape[0]):
                        for j in range( Kmatrix.shape[1]):
                            k_norm[i, j] = Kmatrix[i, j] / np.sqrt( Kmatrix[i, i] *  Kmatrix[j, j])
                    return k_norm
        
      def fit(self,graphs_list1):
        if  not  isinstance(graphs_list1, list):
              graphs_list1=[graphs_list1]
        self.graphs_list1= graphs_list1
        self.fitted=True
        return self
            
      def _compute_phi_x_and_k_train(self,graphs_list1):
        self.phi_X=np.zeros((len(graphs_list1), 4))
        for  i in range(len(graphs_list1)):
            vect = np.zeros(4)
            l = np.random.choice(graphs_list1[i].nodes,(self.n_samples,3))
            for j  in  range(self.n_samples): 
                Sg = graphs_list1[i].subgraph(l[j,])
                for k in range(len(self.graphlets)) :
                    if (nx.is_isomorphic(Sg,self.graphlets[k])) :
                        #print('Isomorphic subgraph found', Sg.nodes(data=True),Sg.edges(data=True))
                        vect[k] = vect[k] + 1
            self.phi_X[i,] = vect
        self.k_train=np.dot(self.phi_X, self.phi_X.T)
        return self
        
      def transform(self,graphs_list2):
         if  not self.fitted:
                 raise Exception("X Not fited")
         if   not  isinstance(graphs_list2, list):
          graphs_list2=[graphs_list2]
         self._compute_phi_x_and_k_train(self.graphs_list1)
         self.phi_Y=np.zeros((len(graphs_list2), 4))
         for  i,g_1 in enumerate(graphs_list2):
            vect = np.zeros(4)
            l = np.random.choice(g_1.nodes,(self.n_samples,3))
            for j  in  range(self.n_samples): 
                Sg = g_1.subgraph(l[j,])
                for k in range(len(self.graphlets)) :
                    if (nx.is_isomorphic(Sg,self.graphlets[k])) :
                        vect[k] = vect[k] + 1
            self.phi_Y[i,] = vect
         k_test=np.dot(self.phi_Y, self.phi_X.T)
         k_test2= np.dot(self.phi_Y, self.phi_Y.T)
         if self.normalise:
            return k_test/np.sqrt(np.outer(np.diagonal(k_test2),np.diagonal(self.k_train)))
         return k_test
             
      def fit_transform(self,X):
         self.fit(X)
         return  self.transform(X)
             
      def get_phi(self):
          return self.phi_X, self.phi_Y
          

      def __call__(self):
              return self.k