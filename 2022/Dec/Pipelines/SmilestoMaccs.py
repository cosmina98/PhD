import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.base import BaseEstimator, TransformerMixin

def generate_maccs_keys_from_smiles_and_cids(smiles): #takes a list of smiles and returns df of cids and associated mac
    list_of_maccs=[]

    for idx, row in enumerate(smiles):
  
        mol = Chem.MolFromSmiles(row)
        
        if mol == None :
            print("Can't generate MOL object:", idx)
            list_of_maccs.append('None')
        else:
            list_of_maccs.append(list(MACCSkeys.GenMACCSKeys(mol).ToBitString()))

    return(list_of_maccs)




class SmitoMACCS(BaseEstimator, TransformerMixin):
     

    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
        X=generate_maccs_keys_from_smiles_and_cids(X) #takes a df and returns df of cids and associated mac
        if any(a == 'None' for a in X):
          return None
        return np.array(X)