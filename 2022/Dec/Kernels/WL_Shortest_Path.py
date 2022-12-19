from collections.abc import Iterable
import numpy as np
import networkx as nx
from numpy.linalg import inv,multi_dot
from scipy.linalg import expm
import copy
from collections import Counter
from Kernels.utils import get_a_floyd_S_graph

class WL_Shortest_Path():
    
        def __init__(self,normalise=False,h=0,node_label=None,edge_label=None,edge_weight='weight'):
              self.phi=None
              self.normalise=normalise
              self.kernel_matrix=None
              self.h=h
              self.node_label=node_label
              self.graphs_list1=None
              self.fitted=False
              self.edge_label=edge_label
              self.edge_weight=edge_weight
              

        def compare(self,Gn):
                Kmatrix = np.zeros((len(Gn), len(Gn)))
                edge_label=self.edge_label
                node_label = self.node_label
                node_label_check=node_label != None
                height = self.h
                height = int(height)
                edge_weight=self.edge_weight
                new_GN=[]
                # initial for height = 0
                Gn = [ get_a_floyd_S_graph(G, edge_weight = edge_weight) for G in Gn ] # get shortest path graphs of Gn
                # initial for height = 0
                for i in range(0, len(Gn)):
                    for j in range(i, len(Gn)):
                        for e1 in Gn[i].edges(data = True):
                            for e2 in Gn[j].edges(data = True):
                                #if there s cost matching  and source node and target node or target node with source node 
                                if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
                                    Kmatrix[i][j] += 1
                        Kmatrix[j][i] = Kmatrix[i][j]
                # iterate each height

                for G in Gn:
                        
                    G=nx.convert_node_labels_to_integers(G)
                    # get the set of original labels
                    if node_label_check is True:
                        labels = list(nx.get_node_attributes(G, node_label).values())
                        if  isinstance(labels[0], np.ndarray):
                            labels=["".join(map(str, (item.astype(int)))) for item in labels]
                            for i,node in enumerate(G.nodes(data=True)):
                                node[1][node_label]=labels[i]
                                #print((node[1][node_label]))

                        #print(labels_ori)
                        #print('stop')
                    else: 

                        node_label= 'new_node_label'
                        #print(G.nodes(data=True))
                        for node in G.nodes():

                            d = G.degree(node)
                            nx.set_node_attributes(G,{node:str(d)}, name=node_label)
                        #print(G.nodes(data=True))


                    #print(labels_ori)


                    new_GN.append(G)
                 
                Gn=new_GN
                
                for h in range(1, height + 1):
                    
                    all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
                    num_of_labels_occured = 0 # number of the set of letters that occur before as node labels at least once in all graphs
                    for G in Gn: # for each graph
                        set_multisets = []
                        for node,attr in G.nodes(data = True):
                                # Multiset-label determination.
                                multiset = [str(G.nodes[n][node_label]) for n in G.neighbors(node)]
                                # sorting each multiset
                                multiset.sort()
                                multiset = G.nodes[node][node_label] + ''.join(multiset) # concatenate to a string and add the prefix 
                                set_multisets.append(multiset)
                                #print(set_multisets)
     

                        # label compression
                        set_unique = list(set(set_multisets)) # set of unique multiset labels
                        # a dictionary mapping original labels to new ones. 
                        set_compressed = {}
                        # if a label occured before, assign its former compressed label, else assign the number of labels occured + 1 as the compressed label 
                        for value in set_unique:
                            if value in all_set_compressed.keys():
                                set_compressed.update({ value : all_set_compressed[value] })
                            else:
                                set_compressed.update({ value : str(num_of_labels_occured + 1) })
                                num_of_labels_occured += 1

                        all_set_compressed.update(set_compressed)

                        # relabel nodes
                        for node in G.nodes(data = True):
                            node[1][node_label] = set_compressed[set_multisets[node[0]]]
                    
                    # calculate subtree kernel with h iterations and add it to the final kernel
                    for i in range(0, len(Gn)):
                        for j in range(i, len(Gn)):
                            for e1 in Gn[i].edges(data = True):
                                for e2 in Gn[j].edges(data = True): 
                                   
                                    if e1[2]['cost'] != 0 and e1[2]['cost'] == e2[2]['cost'] and ((e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])):
                                        
                                        Kmatrix[i][j] += 1
                                Kmatrix[j][i] = Kmatrix[i][j]
                if self.normalise:
                    return self.normalise_k(Kmatrix)
                else:
                    return Kmatrix

      

        def fit(self,graphs_list1):
            if  not  isinstance(graphs_list1, list):
                  graphs_list1=[graphs_list1]
            self.graphs_list1= graphs_list1
            self.fitted=True
            return 
            

        def normalise_k(self, Kmatrix):
                    k_norm = np.zeros(Kmatrix.shape)
                    for i in range(Kmatrix.shape[0]):
                        for j in range( Kmatrix.shape[1]):
                            k_norm[i, j] = Kmatrix[i, j] / np.sqrt( Kmatrix[i, i] *  Kmatrix[j, j])
                    return k_norm

        def transform(self, graphs_list2):
             if  not self.fitted:
                 raise Exception("X Not fiited")
             graphs_list1=self.graphs_list1
             if   not  isinstance(graphs_list1, list):
                  graphs_list1=[graphs_list1]
             if   not  isinstance(graphs_list2, list):
                  graphs_list2=[graphs_list2]
                    
             self.kernel_matrix= np.zeros((len(graphs_list2), len(graphs_list1)))
             #print( self.kernel_matrix)
             for  i,g_x in enumerate(graphs_list2):
                  for  j,g_y in enumerate(graphs_list1):
                            self.kernel_matrix[i][j]=self.compare([g_x,g_y])[0][1]
                       
             return self.kernel_matrix
            
                
        def fit_transform(self,X):
             self.fit(X)
             return  self.transform(X)

        def get_phi(self):
              return self.phi


        def __call__(self):
              pass