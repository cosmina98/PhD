from karateclub import Graph2Vec
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NxGraphstoGraph2Vec(BaseEstimator, TransformerMixin):


    def fit(self, X,y=None):
        return self
    def transform(self, X, y=None):
       model = Graph2Vec()
       model.fit(np.array(X))
       X = model.get_embedding()
       return np.array(X)