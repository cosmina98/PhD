class Trainer():
    def __init__ (self):
       
      self.batch_size=85
      self.verbose=True
        
      self.loss_function = nn.CrossEntropyLoss()
       
      #common  to most optimsers
      self.learning_rate=1e-3
      self.weight_decay=0
      self.running_loss = 0.0
    
      #for sgd  type of optimisers
     
      #error when trying to set the momentum using self.momentum
      self.nesterov= True,
      self.dampening=0
     
      #for adam optimiser
      self.betas=(0.9,0.99)
      self.eps=1e-8
      self.amsgrad=False

      self.mlp=MLP()
      self.optimiser=None
    class MLP(nn.Module):
      '''
        Multilayer Perceptron.
      '''
      def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          nn.Linear(32 * 32 * 3, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, 10)
        )


      def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

    def print_self(self):
        print(mlp)
          
    def set_optimiser(self,optimiser_choice):
        #playing around with adam and sgd only at the moment
        #there is only two of them 
        
        if optimiser_choice.lower().strip()=='adam':
          self.optimiser=torch.optim.Adam(self.mlp.parameters(),lr=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        elif optimiser_choice.lower().strip()=='sgd':
          self.optimiser=torch.optim.SGD(self.mlp.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9,nesterov=self.nesterov , dampening=self.dampening)
          return self.optimiser



    def get_params(self): #get parameters
       return  (self.optimiser.state_dict(), self.mlp.state_dict)

    def fit(self):
        pass
    def partial_fit(X, y): #Update the model with a single iteration over the given data.
        pass

    def predict(X): #Predict using the multi-layer perceptron classifier.
        pass 

    def predict_log_proba(X):  #	Return the log of probability estimates.
        pass 

    def  predict_proba(X):	#Probability estimates.
        pass 


    def score(X, y): #Return the mean accuracy on the given test data and labels.
        pass 
    def set_params(**params): #sets parameters
        pass

