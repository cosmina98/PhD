class Trainer():
    def __init__ (self):
       
      self.batch_size=2000
      self.epochs=2
      #used for label count 
      self.num_classes=0
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

      self.mlp=self.MLP()
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
       if (self.optimiser != None) :
         return  (self.props(),self.optimiser.state_dict(), self.mlp.state_dict)
       else: return self.mlp.state_dict() 
  
    def fit(self,trainloader):
        for epoch in range(self.epochs):  # loop over the dataset 2 times

          running_loss =self.running_loss
          for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
             
            # zero the parameter gradients
            self.optimiser.zero_grad()

            # forward + compute loss+ backward + optimize
            outputs = self.mlp(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimiser.step()
            # print statistics
            running_loss += loss.item()
            if i % self.batch_size == 1999:    # print every 2000 mini-batches
                if self.verbose==True:
                  print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / self.batch_size:.3f}')
                running_loss = 0.0


    def predict(self,testloader): #Predict using the multi-layer perceptron classifier.
      prediction_list=[]
      # again no gradients needed
      with torch.no_grad():
          for data in testloader:
              images, labels = data
              outputs = self.mlp(images)
              estimation, predictions = torch.max(outputs, 1)
              prediction_list.append(predictions)
      return prediction_list #torch.stack(prediction_list)


    def predict_log_proba(self,loader):  #	Return the log of probability estimates.
        y_prob=self.predict_proba(loader)
        log_proba=np.log(y_prob)
        return log_proba
 

    def predict_proba(self,loader):	#Probability estimates.
      probabilities_list=[]
      # again no gradients needed
      with torch.no_grad():
          for data in loader:
              images, labels = data
              outputs = self.mlp(images)
              estimation, predictions = torch.max(outputs, 1)
              probabilities_list.append(estimation)
      return probabilities_list
      

    def score(self,loader): #Return the mean accuracy on the given test data and labels.
        prediction_list=[]
        if self.num_classes==0:
            classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            # prepare to count predictions for each class
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
        #use the class number as key
        else: 
            classes=tuple(np.arange(self.num_classes))  
            correct_pred = {classname: 0 for classname in np.arange(num_classes)}
            total_pred= {classname: 0 for classname in np.arange(num_classes)}

        # again no gradients needed
        with torch.no_grad():
            for data in loader:
                images, labels = data
                outputs = self.mlp(images)
                estimation, predictions = torch.max(outputs, 1)
                prediction_list.append(predictions)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
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


