import random
import networkx as nx
import numpy as np
import os
from collections import Counter
from itertools import chain
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchnet as tnt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


from typing import (
    List,
    Dict,
    Tuple,
    Iterable,
    Union,
    Set,
)


def get_a_floyd_S_graph(G1, edge_weight=None):
    spMatrix = nx.floyd_warshall_numpy(G1,weight=edge_weight)
    S = nx.Graph()
    S.add_nodes_from(G1.nodes(data=True))
    ns = list(G1.nodes())
    for i in range(0, G1.number_of_nodes()):
        for j in range(i + 1, G1.number_of_nodes()):
            if spMatrix[i, j] != np.inf:
                S.add_edge(ns[i], ns[j], cost=spMatrix[i, j])
    return S


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def count_commons(a: Iterable, b: Iterable) -> int:
    'Return the number of common elements in the two iterables'
    uniques = set(a).intersection(set(b))
    counter_a = Counter(a)
    counter_b = Counter(b)
    commons = 0
    for u in uniques:
        commons += counter_a[u] * counter_b[u]
    return commons


def get_nxgraph_from_adjanecy_matrix(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    all_rows = range(0, adjacency_matrix.shape[0])
    for n in all_rows:
        gr.add_node(n)
    gr.add_edges_from(edges)
    gr.add_edges_from(edges)
    return gr


def transform_labels_to_integers_from_a_list_of_graphs(Gn,node_label='node_label'):
    le = preprocessing.LabelEncoder()
    new_g=[]
    all_labels_ori=set()
    for idx, G in enumerate(Gn):
        labels_ori=(list(nx.get_node_attributes(G, node_label).values()))
        if  isinstance(labels_ori[0], np.ndarray):
            labels_ori=["".join(map(str, (item.astype(int)))) for item in labels_ori]
            for i,node in enumerate(G.nodes(data=True)):
                node[1][node_label]=labels_ori[i]
        all_labels_ori.update(labels_ori)
        le.fit([i for i in list(all_labels_ori)])
    
        for i,node in enumerate(G.nodes(data=True)):#
            node[1][node_label]=le.transform([labels_ori[i]])[0]+1
        new_g.append(G)
        
    return new_g







