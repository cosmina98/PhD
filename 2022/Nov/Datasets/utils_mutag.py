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
import sys

def get_graph_signal(nx_graph):
    d = dict((k, v) for k, v in nx_graph.nodes.items())
    x = []
    invd = {}
    j = 0
    for k, v in d.items():
        x.append(v['attr_dict'])
        invd[k] = j
        j = j + 1
    return np.array(x)

#current = os.getcwd()
#parent = os.path.dirname(current)
#sys.path.append(parent)
def load_data(path, ds_name, use_node_labels=True,use_edge_labels=True, max_node_label=10,max_edge_label=4):
    node2graph = {}
    Gs = []
    data = []
    dataset_graph_indicator = f"{ds_name}_graph_indicator.txt"
    dataset_adj = f"{ds_name}_A.txt"
    dataset_node_labels = f"{ds_name}_node_labels.txt"
    dataset_edge_labels = f"{ds_name}_edge_labels.txt"
    dataset_graph_labels = f"{ds_name}_graph_labels.txt"
    path_graph_indicator = "../" + os.path.join(path, dataset_graph_indicator)
    path_adj =  "../" + os.path.join(path, dataset_adj)
    path_node_lab =  "../" + os.path.join(path, dataset_node_labels)
    path_edge_lab= "../" + os.path.join(path, dataset_edge_labels)
    path_labels =  "../" + os.path.join(path, dataset_graph_labels)
    print(path_graph_indicator)
    with open(path_graph_indicator, "r") as f:
        c = 1
        for line in f:
            node2graph[c] = int(line[:-1])
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1
    with open(path_adj, "r") as f:
        i=1
      
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            if use_edge_labels:
                with open(path_edge_lab, "r") as f:
                    j=0
                    for line2 in f:
                        j+=1
                        if (i == j):
                            edge_label = indices_to_one_hot(int(line2[:-1]), max_edge_label)
                            Gs[node2graph[int(edge[0])] -
                   1].add_edge(int(edge[0]), int(edge[1]),edge_label=edge_label)
            else:
                Gs[node2graph[int(edge[0])] -
               1].add_edge(int(edge[0]), int(edge[1]))
            i+=1
                      


    if use_node_labels:
        with open(path_node_lab, "r") as f:
            c = 1
            for line in f:
                node_label = indices_to_one_hot(int(line[:-1]), max_node_label)
                Gs[node2graph[c] - 1].add_node(c, attr_dict=node_label)
                c += 1
                

    labels = []
    with open(path_labels, "r") as f:
        for line in f:
            labels.append(int(line[:-1]))

    return list(zip(Gs, labels))


def create_loaders(dataset, batch_size, split_id, offset=-1):
    train_dataset = dataset[:split_id]
    val_dataset = dataset[split_id:]
    return to_pytorch_dataset(train_dataset, offset, batch_size), to_pytorch_dataset(val_dataset, offset, batch_size)
            
def to_pytorch_dataset(dataset, label_offset=0, batch_size=1):
    list_set = []
    for graph, label in dataset:
        F, G = get_graph_signal(graph), nx.to_numpy_matrix(graph)
        numOfNodes = G.shape[0]
        F_tensor = torch.from_numpy(F).float()
        G_tensor = torch.from_numpy(G).float()

        # fix labels to zero-indexing
        if label == -1:
            label = 0

        label += label_offset

        list_set.append(tuple((F_tensor, G_tensor, label)))

    dataset_tnt = tnt.dataset.ListDataset(list_set)
    data_loader = torch.utils.data.DataLoader(
        dataset_tnt, shuffle=True, batch_size=batch_size)
    return data_loader

def indices_to_one_hot(number, nb_classes, label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""
    if number == label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]