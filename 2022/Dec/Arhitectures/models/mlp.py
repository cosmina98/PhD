import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU,Parameter
import torch.nn.functional as F
from typing import Optional, Tuple
import torch.nn.functional as F
from torch import Tensor


class MLPNN(nn.Module):
      '''
        Multilayer Perceptron.
      '''
      def __init__(
        self,
        input_dim,
        output_dim,
        #if you only want to add  a single hidden layer you do the following:
        hidden_layers=((64)),
        dropout=0.2
        #hidden_layers=(),
        ):
        super().__init__()
        #self.modules1 = torch.nn.ModuleList([])
        self.output_dim=output_dim
        first_hidden_dimension=hidden_layers[0]
        self.hidden_layer_num=len(hidden_layers)-1
        self.d_in=input_dim
        if self.hidden_layer_num>0:
            self.h_0=first_hidden_dimension
            self.h_n=hidden_layers[-1]
 
        else: 
            self.h_0=self.h_n=first_hidden_dimension
         
        activation=torch.nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.d_in,  self.h_0))
        self.layers.append( self.dropout)
        self.layers.append(activation)

        if self.hidden_layer_num>0:
          in_channels=self.h_0
          for h_dim in list(hidden_layers[1:]):
              self.layers.append(torch.nn.Linear(in_channels, h_dim))
              self.layers.append( self.dropout)
              self.layers.append(activation)
              in_channels = h_dim
        self.layers.append(nn.Linear( self.h_n,  self.output_dim))
        #self.modules1.append(nn.Linear( self.h_n, output_dim))
 
      def forward(self, input_data):
        input_data=input_data.view(-1,self.d_in)
        
        for i,layer in enumerate(self.layers):
            input_data=layer(input_data)

        if self.output_dim==1:
           return  F.sigmoid(input_data)
        else: return F.log_softmax(input_data)
        
        
        
      def  update(self,num_h_layers):
           #clear the layers
           self.layers=None
           self.layers = nn.ModuleList()
           activation=torch.nn.ReLU()
           self.layers.append(nn.Linear(self.d_in,  self.h_0))
           self.layers.append( self.dropout)
           self.layers.append(activation)
           for i in range((num_h_layers)-1):
              self.layers.append(torch.nn.Linear(self.h_0, self.h_0))
              self.layers.append( self.dropout)
              self.layers.append(activation)
           self.layers.append(nn.Linear( self.h_0,  self.output_dim))
           