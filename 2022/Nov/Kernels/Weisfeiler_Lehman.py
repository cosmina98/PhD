from collections.abc import Iterable
import numpy as np
import networkx as nx
from numpy.linalg import inv,multi_dot
from scipy.linalg import expm
import copy
from collections import Counter

class Weisfeiler_Lehman():
    

        def __init__(self,normalise=False,h=0,node_label=None):
              self.phi=None
              self.normalise=normalise
              self.kernel_matrix=None
              self.h=h
              self.node_label=node_label
              self.graphs_list1=None
              self.fitted=False
             

        def compare(self,Gn):
                Kmatrix = np.zeros((len(Gn), len(Gn)))
                node_label = self.node_label
                edge_label = None
                height = self.h
                height = int(height)
                node_label_check=self.node_label != None
                all_num_of_labels_occured = 0
                node_label= self.node_label
                # initial for height = 0
                all_labels_ori = set() # all unique orignal labels in all graphs in this iteration
                all_num_of_each_label = [] # number of occurence of each label in each graph in this iteration
                all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
                num_of_labels_occured = all_num_of_labels_occured # number of the set of letters that occur before as node labels at least once in all graphs
                phi={}
                new_GN=[]
                
                

                #print("Iteration 0")
                # for each graph
                for G in Gn:
                        
                        G=nx.convert_node_labels_to_integers(G)
                        # get the set of original labels
                        if node_label_check is True:
                            labels_ori = list(nx.get_node_attributes(G, node_label).values())
                            if  isinstance(labels_ori[0], np.ndarray):
                                labels_ori=["".join(map(str, (item.astype(int)))) for item in labels_ori]
                                for i,node in enumerate(G.nodes(data=True)):
                                    node[1][node_label]=labels_ori[i]
                                    #print((node[1][node_label]))
                                
                            #print(labels_ori)
                            #print('stop')
                        else: 
                            labels_ori = np.array([str(val) for (node, val) in sorted(G.degree(), key=lambda pair: pair[0])])            
                            node_label= 'new_node_label'
                            #print(G.nodes(data=True))
                            for node in G.nodes():

                                d = G.degree(node)
                                nx.set_node_attributes(G,{node:str(d)}, name=node_label)
                            #print(G.nodes(data=True))


                        #print(labels_ori)

                        all_labels_ori.update(labels_ori)
                        num_of_each_label = dict(Counter(labels_ori)) # number of occurence of each label in graph
                        all_num_of_each_label.append(num_of_each_label)
                        num_of_labels = len(num_of_each_label) # number of all unique labels
                        all_labels_ori.update(labels_ori)
                        new_GN.append(G)
                #print("All unique labels found for the 2 graphs ", all_labels_ori)
                #print("List of frequency of each label per graph", all_num_of_each_label)



                all_num_of_labels_occured += len(all_labels_ori)
              

                # calculate subtree kernel with the 0th iteration and add it to the final kernel
                for i in range(0, len(Gn)):
                    for j in range(i, len(Gn)):
                        labels = set(list(all_num_of_each_label[i].keys()) + list(all_num_of_each_label[j].keys()))
                        vector1 = np.matrix([ (all_num_of_each_label[i][label] if (label in all_num_of_each_label[i].keys()) else 0) for label in labels ])
                        vector2 = np.matrix([ (all_num_of_each_label[j][label] if (label in all_num_of_each_label[j].keys()) else 0) for label in labels ])
                        
                        Kmatrix[i][j] += np.dot(vector1, vector2.transpose())
                        Kmatrix[j][i] = Kmatrix[i][j]
                    
                 
                #print(Kmatrix)
                Gn=new_GN

                # iterate each height
                for h in range(1, height + 1):
                    #print('iteration ' , h)
                    all_set_compressed = {} # a dictionary mapping original labels to new ones in all graphs in this iteration
                    num_of_labels_occured = all_num_of_labels_occured # number of the set of letters that occur before as node labels at least once in all graphs
                    all_labels_ori = set()
                    all_num_of_each_label = []

                    # for each graph
                    for idx, G in enumerate(Gn):
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
                            # if a label occured before, assign its former compressed label,
                            #else assign the number of labels occured + 1 as the compressed label 
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

                            # get the set of compressed labels
                            labels_comp = list(nx.get_node_attributes(G, node_label).values())
                            all_labels_ori.update(labels_comp)
                            num_of_each_label = dict(Counter(labels_comp))
                            all_num_of_each_label.append(num_of_each_label)
                            
                            #print('Graph ', idx, 'set_multisets', set_multisets)
                            #print('Compressed labels ', set_compressed)
                    all_num_of_labels_occured += len(all_labels_ori)
                    max_l=0
                    #print("Dictionary of frequency of each compressed label per graph " ,all_num_of_each_label)
                    #print("Set of unique compressed labels ",all_labels_ori)
                    #print("New labels overall " ,all_set_compressed)

                    # calculate subtree kernel with h iterations and add it to the final kernel
                    for i in range(0, len(Gn)):
                        for j in range(i, len(Gn)):
                            #get the set of labels
                            labels = set(list(all_num_of_each_label[i].keys()) + list(all_num_of_each_label[j].keys()))
                            #print(labels)
                            vector1 = np.matrix([ (all_num_of_each_label[i][label] if (label in all_num_of_each_label[i].keys()) else 0) for label in labels ])
                            vector2 = np.matrix([ (all_num_of_each_label[j][label] if (label in all_num_of_each_label[j].keys()) else 0) for label in labels ])
                            #print(vector1, vector2)
                          
                            Kmatrix[i][j] += np.dot(vector1, vector2.transpose())
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
            return self
            

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
             
            
        def get_the_largest_phi_between_graphs(self):

            import re
            def add_comma(match):
                return match.group(0) + ','

            max_l=0
            mylist=[]
            for key in self.phi.keys():
                key= re.sub(r'\[[0-9\.\s]+\]', add_comma, key)
                key = re.sub(r'([0-9\.]+)', add_comma, key)
                mylist.append(eval(key))
                if (len(eval(key)[0])> max_l):
                    max_l=len(eval(key)[0])
            mylist=[l for l in mylist if len(l[0])==max_l]
            self.phi= np.array([[a1, b1] for (a1, b1) in zip(mylist[0], mylist[1])])
            return self.phi

       
        def reset(self):
            pass


        def __call__(self):
              pass
