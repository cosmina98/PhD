
import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU,Parameter
import torch.nn.functional as F
from typing import Optional, Tuple
import torch.nn.functional as F
from torch import Tensor
import torch.utils.data as data
import torch.utils.data as data_utils
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from hyperopt import hp
from sklearn.metrics import mean_squared_error,make_scorer

class Classifier(object):
    
    def __init__ (self,model=None,optimiser=None,loss_function = nn.CrossEntropyLoss(),batch_size=150,epochs=2,verbose=False,
     lr=1e-4):
    
      self.batch_size=batch_size
      self.epochs=epochs
      self.num_h_layers=2
      self.lr=lr
      self.verbose=verbose
      self.shuffle=False
      self.seed=42
      self.running_loss = 0
      self.loss_function = loss_function
      self.sm = torch.nn.Softmax()
      #default optimiser: adam 
      self.h_dim=32
      if model==None:
        model=MLPNN(input_dim=2,output_dim=1,hidden_layers=((32,64,65)))
      self.model=model
      if optimiser==None:
        self.optimiser=torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
      else:self.optimiser=optimiser
      for g in self.optimiser.param_groups:
           g['lr'] = self.lr
    
    def get_params(self): #get parameters
       return self.model.state_dict() 
  
    def fit(self,X,y,eval_set=[]):
        train = [*zip(X,y)]
        trainloader =  torch.utils.data.DataLoader(train,batch_size=self.batch_size, shuffle=False, num_workers=2)
        for epoch in range(self.epochs):  # loop over the dataset 2 times
          running_loss =self.running_loss
          for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            self.optimiser.zero_grad()
            # forward + compute loss+ backward + optimize
            outputs = self.model(torch.tensor(inputs,dtype=torch.float32))
            #print(outputs)
            if isinstance(self.loss_function,torch.nn.modules.loss.BCELoss):
                labels=torch.tensor(labels,dtype=torch.float32).reshape(-1,1)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimiser.step()
            # print statistics
            running_loss += loss.item()
            if i % self.batch_size == (self.batch_size-1):    # print every "batch_size" mini-batches
                if self.verbose==True:
                  print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / self.batch_size:.3f}')
                running_loss = 0.0

    def predict(self, X_test): 
      if isinstance(X_test, (pd.Series, list, np.ndarray)):
        X_test=(torch.tensor(X_test,  requires_grad = False,dtype=torch.float32))
      outputs = self.model(X_test)
      with torch.no_grad():
        if self.model.output_dim==1:
            predictions = torch.round(outputs).detach()
        #if we have multi_class classification 
        else:
            log_prob, predictions = torch.max(outputs, 1)
      return predictions #torch.stack(prediction_list)
  
    def predict_log_proba(self,X):  #	Return the log of probability estimates.
        y_prob=(self.predict_proba(X))
        log_proba=np.log(y_prob.detach().numpy())
        return log_proba
 

    def predict_proba(self,X_test):	#Probability estimates.
      probabilities_list=[]
      if isinstance(X_test, (pd.Series, list, np.ndarray)):
        X_test=(torch.tensor(X_test,  requires_grad = False,dtype=torch.float32))
      with torch.no_grad():
            outputs = self.model(X_test)
            predicted=self.sm( outputs) 
            if self.model.output_dim==1:
                predicted=outputs
            return  predicted

    def score(self,X_test,y_test): #Return the mean accuracy on the given test data and labels.
        return accuracy_score(y_test, self.predict(X_test).numpy(), sample_weight=None)
        
    def score_per_class(self,X_test,y_test):
      predictions= self.predict(X_test).numpy()
      if isinstance(X_test, (pd.Series, list, np.ndarray)):
        num_classes=len(torch.unique(torch.tensor(y_test)))
      else:
        num_classes=len(torch.unique((y_test)))
      if (self.model.output_dim==1):
        num_classes=2
       #Return the mean accuracy on the given test data and labels.
      classes=tuple(np.arange(num_classes))
      correct_pred = {classname: 0 for classname in np.arange(num_classes)}
      total_pred= {classname: 0 for classname in np.arange(num_classes)}
      if (self.model.output_dim==1):
          predictions=predictions.reshape(-1)
      for label, prediction in zip(y_test, predictions):
          if label == prediction:
              correct_pred[classes[label]] += 1
          total_pred[classes[label]] += 1
          accuracies={}
        # print accuracy for each class
      for classname, correct_count in correct_pred.items():
          accuracy = 100 * float(correct_count) / total_pred[classname]
          accuracies[classname]=accuracy
          if self.verbose:
            print(f'Accuracy for class: {classname} is {accuracy:.1f} %')

        

    def set_params(self, attr, value): #sets parameters
        setattr(self, attr, value)
        
    def props(cls):   
        return [i for i in cls.__dict__.items() if i[:1] != '_']

    def get_classifier(self):
        return self.model()



    def optimize_hyperparam(self, X, y,test_size=0.2, n_eval=10, **params):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=42,shuffle=self.shuffle)
        space= {}
        if params is not None:
          for k in params.items():
            if not isinstance(k[1], tuple):
              exec(f'self.{k[0]} = k[1]')
       
        if 'learning_rate' in params.keys():
            space['learning_rate']=hp.loguniform('learning_rate',params['learning_rate'][0],params['learning_rate'][1])
        if 'num_h_layers' in params.keys():
            space['num_h_layers']=hp.uniform('num_h_layers',params['num_h_layers'][0],params['num_h_layers'][1])
        if 'batch_size' in params.keys():
            space['batch_size']=hp.uniform('batch_size',params['batch_size'][0],params['batch_size'][1])
        
        def objective(hyperparams):
          if 'learning_rate' in hyperparams.keys():
            self.lr=int(hyperparams['learning_rate'])
          if 'num_h_layers' in hyperparams.keys():
            self.num_h_layers=int(hyperparams['num_h_layers'])
            self.model.update(self.num_h_layers)
          if 'batch_size' in hyperparams.keys():
            self.batch_size=int(hyperparams['batch_size'])
          for g in self.optimiser.param_groups:
               g['lr'] = self.lr
          self.fit(X=X_train, y=y_train,
                    eval_set=[])
          y_pred=self.predict(X_test)
          score=mean_squared_error(y_test,y_pred)
          return score
        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=n_eval, verbose=self.verbose,
                             rstate=np.random.default_rng(self.seed))

        hyperparams = space_eval(space, best)
        return hyperparams, trials