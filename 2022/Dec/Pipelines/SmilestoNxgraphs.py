from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw
import networkx as nx
def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                  bond_type=str(bond.GetBondType())[0:])
        
    return G

def list_of_smiles_to_list_of_graph_molecules(list_of_smiles):
   list_of_molecules=[Chem.MolFromSmiles(smiles) for smiles in list_of_smiles] 
   #print(len(list_of_molecules),len(list_of_smiles ))
   i=0
   list_of_graph_molecules=[]
   for mol in list_of_molecules:
      try:
          list_of_graph_molecules.append(mol_to_nx(mol))
          i=i+1
      except:
          print(i)
          list_of_graph_molecules.append('None')
          i=i+1
   return list_of_graph_molecules
        
        

class SmitoNxGraphs(BaseEstimator, TransformerMixin):


    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
        X=list_of_smiles_to_list_of_graph_molecules(X)
        if any(a == 'None' for a in X):
          return None
          print('Error')
        else: return np.array(X)