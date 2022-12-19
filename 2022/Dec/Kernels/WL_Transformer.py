from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from Kernels.WL_Version1 import Weisfeiler_Lehman

class WL_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self,h=0):
        self.h=h
    def fit(self, graph_list, y=None):
        return self
    def transform(self, graph_list, y=None):
        kernel=Weisfeiler_Lehman(normalise=True,h=self.h)
        kernel.fit(graph_list)
        #dictionary=kernel.get_phi(
        return kernel.get_phi()