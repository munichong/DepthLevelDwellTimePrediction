'''
Created on Aug 27, 2016

@author: munichong
'''
import pandas as pd  
from random import random

flow = (list(range(1,10,1)) + list(range(10,1,-1)))*10
pdata = pd.DataFrame({"a":flow, "b":flow})  
pdata.b = pdata.b.shift(9)  
data = pdata.iloc[10:] * random()  # some noise  

import numpy as np

def _load_data(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)


(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data

print(type(X_train), X_train.shape)
print(type(y_train), y_train.shape)
print(X_train)
print("***********************************")
print(y_train)


