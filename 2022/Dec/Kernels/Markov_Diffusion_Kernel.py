from collections.abc import Iterable
import numpy as np
import networkx as nx
from numpy.linalg import inv,multi_dot
from scipy.linalg import expm

class Markov_Diff():
    
#for cluustering graph nodes

    
    def __init__(self):
    
          self.t=3
          
          
    def markov_diff_kernel(graph):
        t=self.t
        adj_matrix = nx.to_numpy_matrix(graph)
        #degree matrix
        deg_matrix = np.diag(np.sum(np.array(adj_matrix) , axis=0))
        #the transition matrix p
        p_matrix = np.matmul(np.linalg.inv(deg_matrix), np.array(adj_matrix))

        p_matrix_power = p_matrix.copy()
        z_matrix = p_matrix.copy()

        for tau in range(2, int(t + 1), 1):
            p_matrix_power = np.dot(p_matrix_power, p_matrix)
            z_matrix += p_matrix_power

        z_matrix = z_matrix / t

        # pool = mp.Pool(n_cores)
        # out = pool.starmap(np.linalg.matrix_power, zip(repeat(p_matrix),list(np.arange(1, int(t + 1), step=1))))
        # z_matrix = np.sum(out, axis=0) / t

        kernel = np.matmul(z_matrix, z_matrix.transpose())


        return kernel
        
    
    