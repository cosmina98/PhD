from torch import nn
from torch.nn import functional as F
class Net(nn.Module):
  def __init__(self,input_shape,output_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,64)
    self.fc2 = nn.Linear(32,32)
    self.fc3 = nn.Linear(32,output_shape)  
  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x