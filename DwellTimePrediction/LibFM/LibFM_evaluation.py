'''
Created on Mar 25, 2016

@author: Wang
'''
from math import sqrt
from sklearn.metrics import log_loss, mean_squared_error
import scipy as sp


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


y_pred = []
for line in open('I:/Desktop/output_ffm_3s', 'r'):
    y_pred.append(float(line.strip()))

y_test = []
for line in open('I:/Desktop/ground_truth_3s.csv', 'r'):
    y_test.append(float(line.strip()))
    
# print('RMSD_libfm =', sqrt(mean_squared_error(y_test, y_pred)))
print('Logloss_libfm =', logloss(y_test, y_pred))