from copy import deepcopy

def get_unique_edge_labels(graph_list,label='bond_type'):
    """Pass in a list of a graphs and the name  of edge labels
    and  get a list of unique labels"""
    unique_labels={}
    for i,g in enumerate(graph_list):
        for j,edge in enumerate(g.edges(data=True)):
            try:
                unique_labels[edge[2][label]]=i
            except:continue
    return sorted(unique_labels.keys())


def one_hot_encode_edge_labels(graph_list,label='bond_type', label_new='edge_attr', drop_label=False):
    "gets an edge attribute, encodes it  and creates a  new edge attribute called edge_attr by default"
    list_of_unique_labels=get_unique_edge_labels(graph_list,label)
    list_of_unique_labels=[x.lower().strip() for x in list_of_unique_labels]
    enc_edges  = preprocessing.LabelBinarizer()
    enc_edges.fit_transform(list_of_unique_labels)
    graph_list=deepcopy(graph_list)
    for i,G in enumerate(graph_list):
     #create a new attribute called edge_attr set to None
     nx.set_edge_attributes(G, 'None', label_new)
     for edge in G.edges(data=True):
        try:
            edge[2][label_new]= enc_edges.transform([edge[2][label].lower().strip()])[0].astype(float)
            if drop_label:
             del edge[2][label]
        except: 
           edge[2][label_new]=enc_edges.transform('None')
          
    return graph_list


def get_unique_node_labels(graph_list,label='atomic_num'):
    """Pass in a list of a graphs and the name  of edge labels
    and  get a list of unique labels"""
    unique_labels={}
    graph_list=deepcopy(graph_list)
    for i,g in enumerate(graph_list):
        for j,node in enumerate(g.nodes(data=True)):
            try:
                unique_labels[node[1][label]]=i
            except:continue
    return sorted(unique_labels.keys())

def one_hot_encode_node_labels(graph_list,labels, label_new='x', drop_label=False):
    "If  a list is given the labels will be one hot encoded and hstack by default"
    graph_list= deepcopy(graph_list)
    encoders=[  preprocessing.LabelBinarizer()  for i in range(len(labels))]
    for i, label in enumerate(list(labels)):
        list_of_unique_labels=get_unique_node_labels(graph_list,label)
        list_of_unique_labels=[x for x in list_of_unique_labels]
        encoders[i].fit_transform(list_of_unique_labels)
        
    for i,G in enumerate(graph_list):
        for node in G.nodes(data=True):
            z=[]
            for i,label in enumerate(labels):
                a=encoders[i].transform([node[1][label]])
                z=np.hstack((z,a[0]))
                if drop_label:
                   del node[1][label]
            node[1][label_new]=z.astype(float)
    return graph_list


