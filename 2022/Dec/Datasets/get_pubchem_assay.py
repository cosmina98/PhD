import pandas as pd
import numpy as np
import requests
import io
from io import StringIO
import time
import pydot
from rdkit.Chem import Draw
from colour import Color
from numpy.random import randint
import matplotlib as mpl
from matplotlib import pyplot as plt
from pylab import rcParams
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout,pygraphviz_layout
import random

def load_csv_data_from_a_PubChem_assay(assay_id):
    #url = f'https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&actvty=all&response_type=save&aid={assay_id}'
    url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{assay_id}/concise/csv'
    #df_raw=pd.read_csv(url)
    df_raw=pd.read_csv(url)
    #print(df_raw.head())
    return(df_raw)

def drop_sids_with_no_cids(df):
    df = df.dropna( subset=['cid'] )
    #Remove CIDs with conflicting activities
    cid_conflict = []
    idx_conflict = []

    for mycid in df['cid'].unique() :
        
        outcomes = df[ df.cid == mycid ].activity.unique()
        
        if len(outcomes) > 1 :
            
            idx_tmp = df.index[ df.cid == mycid ].tolist()
            idx_conflict.extend(idx_tmp)
            cid_conflict.append(mycid)

    #print("#", len(cid_conflict), "CIDs with conflicting activities [associated with", len(idx_conflict), "rows (SIDs).]")
    df = df.drop(idx_conflict)

    #Remove redundant data

    df = df.drop_duplicates(subset='cid')  # remove duplicate rows except for the first occurring row.
    #print(len(df['sid'].unique()))
    #print(len(df['cid'].unique()))
    return df
    
     

def download_smiles_given_cids_from_pubmed(list_of_cids,chunk_size = 200): #returns df of smiles and cids
    df_smiles = pd.DataFrame()

    num_cids = len(list_of_cids)
    list_dfs = []
    if num_cids % chunk_size == 0 :
        num_chunks = int( num_cids / chunk_size )
    else :
        num_chunks = int( num_cids / chunk_size ) + 1

    #print("# CIDs = ", num_cids)
    #print("# CID Chunks = ", num_chunks, "(chunked by ", chunk_size, ")")

    for i in range(0, num_chunks) :
        idx1 = chunk_size * i
        idx2 = chunk_size * (i + 1)
        cidstr = ",".join( str(x) for x in list_of_cids[idx1:idx2] )

        url = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/' + cidstr + '/property/IsomericSMILES/TXT')
        res = requests.get(url)
        data = pd.read_csv( StringIO(res.text), header=None, names=['smiles'] )
        list_dfs.append(data)
        
        time.sleep(0.2)
        
        #if ( i % 5 == 0 ) :
            #print("Processing Chunk ", i)
    df_smiles = pd.concat(list_dfs,ignore_index=True)
    df_smiles[ 'cid' ] = list_of_cids   

    return df_smiles


def load_smiles_and_activity_targets_from_a_PubChem_assay(assay_id):
    df_raw=load_csv_data_from_a_PubChem_assay(assay_id=assay_id)
    print(len(df_raw))
    #Drop substances without Inconclusive activity
    df_raw=df_raw[df_raw['Activity Outcome']!='Inconclusive']
    #Select active/inactive compounds for model building
    df=df_raw[ (df_raw['Activity Outcome'] == 'Active' ) | 
             (df_raw['Activity Outcome'] == 'Inactive' ) ].rename(columns={"CID": "cid", "SID":"sid","Activity Outcome": "activity"})
    #drop duplicates, and comnflicting activities, and substances with no cids
    df=drop_sids_with_no_cids(df)
    #label encoding
    df['activity'] = [ 0 if x == 'Inactive' else 1 for x in df['activity'] ]
    df_smiles=download_smiles_given_cids_from_pubmed(df.cid.astype(int).tolist())
    X=df_smiles.smiles.tolist()
    y=df.activity.astype(int).tolist()
    return(X,y)


def Hex_color(bond_type):
    random.seed( np.sum([ord(c) for c in bond_type]) -30+ ord(bond_type[0]))
    L = '0123456789ABCDEF'
    x= Color('#'+ ''.join([random.choice(L) for i in range(6)][:]))
    return x.get_hex()

def get_edge_color_list(mol_nx):
    edge_color=[ Hex_color(data[2]) for data in mol_nx.edges(data = 'bond_type')] 
    #print(edge_color)
    return edge_color

def return_colors_for_atoms(mol_nx):
    random.seed(767)

    color_map = {}
 
    for idx in mol_nx.nodes():
      if mol_nx.nodes[idx]['atom_symbol'] not in color_map:
         
          color_map[mol_nx.nodes[idx]['atom_symbol']] ="#%06x" % random.randint(sum([ord(c) for c in mol_nx.nodes[idx]['atom_symbol']]), 0xFFFFFF) 

    mol_colors = []
    for idx in mol_nx.nodes():
        if (mol_nx.nodes[idx]['atom_symbol'] in color_map):
            mol_colors.append(color_map[mol_nx.nodes[idx]['atom_symbol']])
        else:
            mol_colors.append('gray')
    return mol_colors

def get_labels(mol_nx):
    return nx.get_node_attributes(mol_nx, 'atom_symbol')
#set(nx.get_node_attributes(mol_nx, 'atom_symbol').values())
#colors=[i/len(mol_nx.nodes) for i in range(len(mol_nx.nodes))]

def draw_one_mol(G, ax=None):
    
    rcParams['figure.figsize'] = 7.5,5
    #color_lookup = {k:v for v, k in enumerate(sorted((nx.get_node_attributes(G, "atom_symbol"))))}
    selected_data = dict( (n, ord(d['atom_symbol'][0])**3 ) for n,d in G.nodes().items() )
    selected_data=[v[1] for k, v in enumerate(selected_data.items())]
    low, *_, high = sorted(selected_data)
    seed=123
    random.seed(seed)
    np.random.seed(seed)    
    pos=pygraphviz_layout(G)
    nx.draw_networkx_nodes(G,pos,
            #labels= get_labels(G),
            node_size=100,
            #edgecolors='black',
            cmap='tab20c_r',
            vmin=low,
            vmax=high,
            node_color=[selected_data],
            ax=ax)
    nx.draw_networkx_labels(G,pos, get_labels(G),ax=ax)
    nx.draw_networkx_edges(G, pos,
          
            width=3,
            edge_color=get_edge_color_list(G),
         ax=ax)
    #nx.draw_networkx_edge_labels(G, pos, get_edge_labels(G), font_size=10,alpha=0.8,verticalalignment='bottom',
                                #horizontalalignment='center',clip_on='True',rotate='True' )
   


def get_edge_labels(G):
  return nx.get_edge_attributes(G,'bond_type')
   
def get_adjency_matrix(mol_nx):
    # print out the adjacency matrix ---------------------------------------------- 
    matrix = nx.to_numpy_matrix(mol_nx)
    print(matrix)
    return matrix



def draw_graphs(list_of_graph_molecules, num_per_line=3,labels=None):
      ixs=[]
      num_per_line_ax_n=0
      if len(list_of_graph_molecules)%num_per_line==0:
       lines=int(len(list_of_graph_molecules)/num_per_line)
      else:
        lines=len(list_of_graph_molecules)//num_per_line+1
        num_per_line_ax_n=len(list_of_graph_molecules) % num_per_line
       
        for i in range(num_per_line-num_per_line_ax_n):
          ix=lines-1,num_per_line-i-1
          ixs.append(ix)
          
      #print(lines)
      fig, ax = plt.subplots(lines, num_per_line)
      fig.set_figheight(10)
      fig.set_figwidth(50)
      for i, mol_nx in enumerate(list_of_graph_molecules):
     
        ix = np.unravel_index(i, ax.shape)
        draw_one_mol(mol_nx, ax=ax[ix])
        if labels is not None:
            ax[ix].set_title(labels[i], fontsize=8)
      for ix in ixs:
           ax[ix].set_axis_off()
 
          
    
