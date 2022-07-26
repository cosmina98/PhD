class Trainer():
    self.batch_size=85
    def _init_ (self):
      self.mlp= self.MLP()
    #definining the MLP class
    class MLP(nn.Module):
      '''
        Multilayer Perceptron.
      '''
      def __init__():
        super().__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          nn.Linear(32 * 32 * 3, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, 10)
        ),

        self.max_iter=200,
        self.momentum=0.9, 
        self.nesterovs_momentum=True,
        self.verbose=True
        self.running_loss = 0.0


      def forward(self, x):
        '''Forward pass'''
        return self.layers(x)




    def get_params(deep): #get parameters
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
