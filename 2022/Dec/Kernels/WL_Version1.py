from collections.abc import Iterable
import numpy as np
import networkx as nx
from numpy.linalg import inv,multi_dot
from scipy.linalg import expm
import copy
from collections import Counter

class Weisfeiler_Lehman():
       def __init__(self,normalise=False,h=0,node_label_check=True,node_label='node_label'):
              self.phi=None
              self.normalise=normalise
              self.kernel_matrix=None
              self.h=h
              self.node_label_check=node_label_check
              self.node_label=node_label
            
       def compare(self,Gn,h=0):        
            graph_list=Gn
            node_label_check=self.node_label_check
            #read the number of graphs 
            num_graphs=len(graph_list)
            #create a list opf 0s pf size n_graphs
            lists = [0] * num_graphs
            #set the number of ieterations
            n_iterations=self.h
            #create a matrix of 0s coressponding to the number of iteartions 
            k = [0] * (n_iterations + 1)
            n_nodes = 0
            n_max = 0
            node_label=self.node_label

            for i in range(num_graphs):
                    #convert each graph  to an adjacency matrix
                    lists[i] = nx.to_numpy_matrix(graph_list[i])
                    #get the total number of nodes 
                    n_nodes = n_nodes + graph_list[i].number_of_nodes()
                    #find the graph with the maximum amount of nodes and update the n_max no of nodes
                    if(n_max < graph_list[i].number_of_nodes()):
                        n_max = graph_list[i].number_of_nodes()
            #initialise phi of 0, of zise n_max and amount of graphs 
            phi = np.zeros((n_max, num_graphs), dtype=np.uint64)
            # INITIALIZATION: initialize the nodes labels for each graph
                    # with their labels or with degrees (for unlabeled graphs)
            #new_labels initialise them to 0
            labels = [0] * num_graphs
            #look_up dictionary
            label_lookup = {}
            label_counter = 0
            if node_label_check is True:
              #for graph in graph
              for i in range(num_graphs):
                #get the list of node labels
                list_of_node_labels_per_graph=list(nx.get_node_attributes(graph_list[i], node_label).values())
                #initialise the new labels list
                labels[i] = np.zeros(len(list_of_node_labels_per_graph), dtype=np.int32)
                #for label in labels
                for j in range(len(list_of_node_labels_per_graph)):
                   #if the label is not in the visited labels
                   if not (list_of_node_labels_per_graph[j] in label_lookup):
                     #add it ti the lookup dictionary
                     label_lookup[list_of_node_labels_per_graph[j]] = label_counter
                     #increase the counter by own 
                     labels[i][j] = label_counter
                     label_counter+=1
                   else:
                     #if the label has been encountered already
                     labels[i][j]= label_lookup[list_of_node_labels_per_graph[j]]
                  #update phi with the frequency of each label  encountered
              
                   try:  phi[labels[i][j], i] += 1
                   except:
                    phi=np.vstack([phi,np.zeros(num_graphs)])
                    phi[labels[i][j], i] += 1
                
            else:
              #if the labels are not provided 
                 for i in range(num_graphs): 
                   #labels for each node will be their degree
                    labels[i] = np.array([int(val) for (node, val) in sorted(graph_list[i].degree(), key=lambda pair: pair[0])])
                    for j in range(len(labels[i])):
                        phi[labels[i][j], i] += 1

            vectors = np.copy(phi.transpose())
            #print(vectors)
            #similarity matrix as the product of the degree counts and itself
            k = np.dot(phi.transpose(), phi)

            # MAIN LOOP
            it = 0
            new_labels = copy.deepcopy(labels)

            while it < n_iterations:
              # create an empty lookup table
              label_lookup = {}
              label_counter = 0
              phi = np.zeros((n_nodes, num_graphs), dtype=np.uint64)
              for i in range(num_graphs):
                  for v in range(len(lists[i])):
                      # form a multiset label of the node v of the i'th graph
                      # and convert it to a string  
                      long_label = np.concatenate(np.array(np.sort(labels[i][(lists[i][v]).astype(int)])))
                      long_label=np.concatenate( (np.array([labels[i][v]]),long_label))
                      long_label_string = str(long_label)
                      #print(long_label_string)
                      # if the multiset label has not yet occurred, add it to the
                      # lookup table and assign a number to it
                      if not (long_label_string in label_lookup):
                          label_lookup[long_label_string] = label_counter
                          new_labels[i][v] = label_counter
                          label_counter += 1
                      else:
                          new_labels[i][v] = label_lookup[long_label_string]
                  # fill the column for i'th graph in phi
                  #count vunique new labels
                  aux = np.bincount(new_labels[i])
                  phi[new_labels[i], i] += np.array(aux[new_labels[i]],dtype=np.uint64)
                 
              k += np.dot(phi.transpose(), phi)
              labels = copy.deepcopy(new_labels)
              it = it + 1
            self.phi=phi
            return k

       def fit(self,graphs_list):
             self.kernel_matrix=self.compare(graphs_list)
             if not self.normalise:
                return self.kernel_matrix
             else:
                # Compute the normalized version of the kernel
                k_norm = np.zeros( self.kernel_matrix.shape)
                for i in range( self.kernel_matrix.shape[0]):
                    for j in range( self.kernel_matrix.shape[1]):
                        k_norm[i, j] =  self.kernel_matrix[i, j] / np.sqrt( self.kernel_matrix[i, i] *  self.kernel_matrix[j, j])

                return k_norm  


       def get_phi(self):
              return self.phi


       def __call__(self):
             pass
