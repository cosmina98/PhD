from collections.abc import Iterable
import numpy as np
import networkx as nx
from numpy.linalg import inv,multi_dot
from scipy.linalg import expm
from scipy.sparse.linalg import cg,LinearOperator

class Random_Walk():

      def __init__(self, kernel_type='geometric',normalise=True):
          self.kernel_type=kernel_type
          self.phi=None
          self.normalise=normalise
          self.k=None
          self.graphs_list1=None
          self.fitted=False
        
      def compare(self,g_1, g_2, verbose=False,kernel_type='geometric'):
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
              A = nx.to_numpy_matrix(g_1, nonedge=0)
              B= nx.to_numpy_matrix(g_2,nonedge=0)
              n,m = A.shape
              p,q=B.shape
              XY = np.kron(A, B)
              #print(XY.shape)
              # XY is a square matrix
              s = XY.shape[0]
              lamda=0.1
              if self.kernel_type=='geometric':
                  #replaced the standard kron with  a faster method due to errors
                        Ax, Ay = A, B
                        xs, ys = Ax.shape[0], Ay.shape[0]
                        mn = xs*ys

                        def lsf(x):
                            xm = x.reshape((xs, ys), order='F')
                            y = np.reshape(multi_dot((Ax, xm, Ay)), (mn,), order='F')
                            return x - lamda * y

                        # A*x=b
                        A = LinearOperator((mn, mn), matvec=lsf)
                        b = np.ones(mn)
                        x_sol, _ = cg(A, b, tol=1.0e-6, maxiter=20, atol='legacy')
                        S=abs(x_sol)
                       
              elif  self.kernel_type=='exponential': 
                    S = expm(lamda*XY).T
              
              self.phi=S
              return np.sum(S)

      def compare_normalise(self,g_1,g_2):
          return float(self.compare(g_1, g_2) )/ (float(np.sqrt(self.compare(g_1, g_1) * self.compare(g_2, g_2))))
        
      def fit(self,graphs_list1):
            if not self.fitted:
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
                        km[i][j]= float(self.compare_normalise(g_1, g_2) )
                     else:  
                        km[i][j]=self.compare(g_1,g_2)
                        
            return km
          
      def fit_transform(self,X):
         self.fit(X)
         return  self.transform(X)
             

      def get_phi(self):
          return self.phi
          

      def __call__(self):
              return self.k