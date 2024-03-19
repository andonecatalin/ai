
import pandas as pd
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
dbname='ai_dataset'
dbuser='postgres'
dbpassword='parola'
dbport=5432
dbhost='127.0.0.1'
table_name='tabela'
tickers=["AAPL","MSFT",'SPY','XAUUSD.OANDA','BCO.ICMTRADER']
for i in range(len(tickers)):
    tickers[i]=tickers[i].replace('.','_')
            
data=functions.request_data(tickers[0],table_name,dbname,dbuser,dbpassword,dbhost,dbport)

data_column1=[row[6] for row in data]
data_column1=np.array(data_column1)
data_column1=data_column1[:33]

data_column=[row[0]for row in data]
data_column=np.array(data_column)
data_column=data_column[:33]

y=functions.graph(data_column1,data_column)
np.set_printoptions(threshold=np.inf)
print(y)