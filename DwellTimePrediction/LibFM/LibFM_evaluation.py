'''
Created on Mar 25, 2016

@author: Wang
'''
from math import sqrt
from sklearn.metrics import mean_squared_error

y_pred = []
for line in open('J:/Dropbox/Yi Chen Financial Informatics/libfm-1.40.windows/data_bs/output', 'r'):
    y_pred.append(float(line.strip()))

y_test = []
for line in open('J:/Dropbox/Yi Chen Financial Informatics/libfm-1.40.windows/data_bs/y.test', 'r'):
    y_test.append(float(line.strip()))
    
print('RMSD_libfm =', sqrt(mean_squared_error(y_test, y_pred)))