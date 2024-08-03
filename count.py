
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
#import data_download
import functions
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import torch
import polars as pl
from torch.nn import functional as F
import torch.optim as optim
import warnings
from torch.optim.lr_scheduler import _LRScheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#warnings for device
if not torch.cuda.is_available():
    raise Exception('Torch.cuda is not available')
elif device!=torch.device('cuda'):
    warnings.warn('Cuda is not selected as device even if it is available')

dbname='ai_dataset'
dbuser='postgres'
dbpassword='parola'
dbport=5432
dbhost='127.0.0.1'
table_name='tabela'

tickers=["AAPL","MSFT",'SPY','XAUUSD.OANDA','BCO.ICMTRADER']
for i in range(len(tickers)):
    tickers[i]=tickers[i].replace('.','_')
net_functions=functions.Protected_execution


query=net_functions.create_query(tickers[0],table_name)
cursor,conn=net_functions.create_cursor(dbname,dbuser,dbpassword,dbport,dbhost)
data=pl.read_database(query=query,connection=conn)
#data=data.with_columns(pl.col('aapl_adj_close').pct_change().alias('labels'))
#normalize with pandas
#data= data.select((pl.all()-pl.all().mean()) / pl.all().std())

data=data.drop_nulls()
data


#get data from polars
#labels=data['labels']
train_data=data['aapl_high']
#make polars data into torch tensors
train_data=torch.tensor(train_data)

batch_size=32

#shorten the tensor
shortened_tensor=net_functions.tensor_shortner(train_data, batch_size)
#create batches of 32
image_tensor=net_functions.image_builder(shortened_tensor,batch_size=batch_size)
#create tensor
labels=net_functions.change(image_tensor, batch_size)
#cut the last image from data variable to adjust for change function
image_tensor=image_tensor[:-1]


#make labels and data into tensors
labels=torch.tensor(labels,dtype=torch.int32)

labels=labels.type(torch.LongTensor)
image_tensor=torch.tensor(image_tensor)

image_tensor=net_functions.make_tensor(data,image_tensor,tickers[0])


#normalize data tensor
#image_tensor=net_functions.fit_to_range_tensor(image_tensor)


#move them to gpu
labels=labels.to(device)
image_tensor=image_tensor.to(device)

def min_max_normalize(tensor):
    for c in range(tensor.size(1)):
        channel_min = tensor[:, c, :].min()
        channel_max = tensor[:, c, :].max()
        tensor[:, c, :] = (tensor[:, c, :] - channel_min) / (channel_max - channel_min)
    return tensor

print(image_tensor.shape)
image_tensor = min_max_normalize(image_tensor)  # Use clone() to avoid modifying the original tensor


train_dataset,test_dataset,train_labels,test_labels=train_test_split(image_tensor,labels,test_size=.1)


class SGDRScheduler(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_cur = 0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super(SGDRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == 0:
            self.T_i = self.T_0
        else:
            self.T_i = self.T_0 * (self.T_mult ** (self.last_epoch // self.T_0))
        
        cos_inner = np.pi * (self.T_cur % self.T_i)
        cos_inner /= self.T_i
        cos_out = np.cos(cos_inner) + 1
        return [self.eta_min + (base_lr - self.eta_min) / 2 * cos_out for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is not None:
            self.T_cur = epoch % self.T_i
        super(SGDRScheduler, self).step(epoch)
        if epoch is not None and epoch % self.T_i == 0:
            self.T_cur = 0




def createTheMNISTNet(printtoggle=False):

  class mnistNet(nn.Module):
    def __init__(self, printtoggle=False):
        super().__init__()

        # Input layer for 1D data of size 32
        self.conv1a = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=2)
        self.conv2a = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=2)

        self.conv1b = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=2, stride=1, padding=2)
        self.conv2b = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=2, stride=1, padding=2)

        self.conv1c = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=2)
        self.conv2c = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=2)

        self.conv1d = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=4, stride=1, padding=2)
        self.conv2d = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=4, stride=1, padding=2)

        self.conv1e = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.conv2e = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=2)

        self.conv1f = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=6, stride=1, padding=2)
        self.conv2f = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=6, stride=1, padding=2)
        
        self.pool=nn.MaxPool1d(kernel_size=2, stride=1)
        # fully-connected layers
        self.fc1 = nn.Linear(832, 460) 
        self.fc2 = nn.Linear(460, 280)
        #self.fc3 = nn.Linear(416, 208)
        self.fc4 = nn.Linear(280, 128)  # Assuming 10 output classes
        self.fc5 = nn.Linear(128, 41)  # Assuming 10 output classes

        # toggle for printing out tensor sizes during forward prop
        #self.print = printtoggle
        self.dropout = nn.Dropout(p=0.25)
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)  # Move the model to the chosen device

    # forward pass
    def forward(self, x):
        # Move input to device
        #x = x.to(self.device)
        x_context=x[0]
        x_a=x
        x_b=x
        x_c=x
        x_d=x
        x_e=x
        x_f=x
        
        #print(f' inainte de conv: {x.shape}')

        #next window
        # Convolution -> relu
        x_a = F.relu(self.conv1a(x))
        x_a = self.pool(x)
        x_a = self.dropout(x)
        
        # Convolution -> relu
        x_a = F.relu(self.conv2a(x))
        x_a = self.pool(x)
        x_a = self.dropout(x)
        '''####################################################'''

        
        #next window
        # Convolution -> relu
        x_b = F.relu(self.conv1b(x))
        x_b = self.pool(x)
        x_b = self.dropout(x)
        
        # Convolution -> relu
        x_b = F.relu(self.conv2b(x))
        x_b = self.pool(x)
        x_b = self.dropout(x)


        
        '''####################################################'''
        #next window
        # Convolution -> relu
        x_c = F.relu(self.conv1c(x))
        x_c = self.pool(x)
        x_c = self.dropout(x)
        
        # Convolution -> relu
        x_c = F.relu(self.conv2c(x))
        x_c = self.pool(x)
        x_c = self.dropout(x)


        
        '''####################################################'''
        #next window
        # Convolution -> relu
        x_d = F.relu(self.conv1d(x))
        x_d = self.pool(x)
        x_d = self.dropout(x)
        
        # Convolution -> relu
        x_d = F.relu(self.conv2d(x))
        x_d = self.pool(x)
        x_d = self.dropout(x)


        
        '''####################################################'''
        #next window
        # Convolution -> relu
        x_e = F.relu(self.conv1e(x))
        x_e = self.pool(x)
        x_e = self.dropout(x)
        
        # Convolution -> relu
        x_e = F.relu(self.conv2e(x))
        x_e = self.pool(x)
        x_e = self.dropout(x)


        
        '''####################################################'''
        #next window
        # Convolution -> relu
        x_f = F.relu(self.conv1f(x))
        x_f = self.pool(x)
        x_f = self.dropout(x)
        
        # Convolution -> relu
        x_f = F.relu(self.conv2f(x))
        x_f = self.pool(x)
        x_f = self.dropout(x)


        
        '''####################################################'''
        x=torch.cat([x_a,x_b,x_c,x_d,x_e,x_f],0)
        x = x.view(-1)  # Flatten the tensor
        x = torch.cat([x_context,x])
        # Fully connected layer

        
        x=torch.cat([x_context,x])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        #x = F.relu(self.fc3(x))
        #x = self.dropout(x)

        x = F.relu(self.fc4(x))
        x = self.dropout(x)

        x = self.fc5(x)
        return x





  # create the model instance
  net = mnistNet(printtoggle)
  
  # loss function (assuming classification task)
  lossfun = nn.CrossEntropyLoss()

  # optimizer
  optimizer = optim.SGD(net.parameters(),lr=.01)
  T_0=50
  scheduler=SGDRScheduler(optimizer, T_0=T_0,T_mult=2,eta_min=0.005)  

  return net,lossfun,optimizer



def function2trainTheModel(train_dataset, train_labels, test_dataset, test_labels,numepochs = 50):

  # number of epochs
  
  
  # create a new model
  net, lossfun, optimizer = createTheMNISTNet()

  # initialize losses
  losses = torch.zeros(numepochs).to(device)
  trainAcc = []
  testAcc = []
  z=0
  # loop over epochs
  for epochi in range(numepochs):
    z+=1
    # initialize batch losses and accuracies
    epochLoss = 0.0
    epochAcc = 0.0
    print(z)
    # loop over training data batches
    for i in range(train_dataset.size()[0]):
      X=train_dataset[i]
      y=train_labels[i]
      
      #X = X.unsqueeze(1)  # Add a singleton dimension for the channel axis
     
      # forward pass and loss
      yHat = net(X)
        
      loss = lossfun(yHat, y)
      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # accumulate batch loss
      epochLoss += loss.item()

      # compute accuracy
      matches = torch.argmax(yHat, axis=0) == y
      accuracyPct = 100 * torch.mean(matches.float())
      epochAcc += accuracyPct.item()

    # end of batch loop

    # average loss and accuracy for the epoch
    epochLoss /= train_dataset.size()[0]
    epochAcc /= train_dataset.size()[0]

    # append to lists
    losses[epochi] = epochLoss
    trainAcc.append(epochAcc)

    # test accuracy (evaluate on test set)
    test_epochAcc = 0
    total_correct = 0
    total_samples = 0
    
    '''with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)  # Move data to device
            yHat_test = net(X_test)
            matches_test = torch.argmax(yHat_test, dim=1) == y_test
            total_correct += matches_test.sum().item()
            total_samples += y_test.size(0)
    
    test_epochAcc = 100 * total_correct / total_samples
    testAcc.append(test_epochAcc)

     ''' 
    test_epochAcc = 0.0
    with torch.no_grad():
      for i_test in range(test_dataset.size()[0]):
        X_test=test_dataset[i_test]
        y_test=test_labels[i_test]
        yHat_test=net(X_test)
        matches_test = torch.argmax(yHat_test, axis=0) == y_test
        accuracyPct_test = 100 * torch.mean(matches_test.float())
        test_epochAcc += accuracyPct_test.item()
    test_epochAcc /= train_dataset.size()[0]
    testAcc.append(test_epochAcc)

  # end epochs loop

  # function output
  return trainAcc, testAcc, losses, net




trainAcc,testAcc,losses,net = function2trainTheModel(train_dataset, train_labels, test_dataset, test_labels,numepochs = 250)




nploss=losses.to("cpu")
nploss=nploss.cpu()
losses=losses.cpu().numpy()
plt.plot(losses)

plt.title('losses over time')


print(testAcc)
