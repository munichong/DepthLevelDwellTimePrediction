'''
Created on Mar 25, 2016

@author: Wang
'''
from math import sqrt
from numpy import mean, median, array
from pymongo import MongoClient
from LibFM import Training_test_generator as ttg
from sklearn.metrics import log_loss
from _collections import defaultdict

client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']
article_DB = client['Forbes_Dec2015']['ArticleInfo']


required_dwell_time = 7


train_pv_num = 0; train_dep_num = 0
test_pv_num = 0; test_dep_num = 0
X_train = []; y_train = []
X_test = []; y_test = []
dwell_depth_avg_gloabl = defaultdict(list)
dwell_depth_avg_channel = defaultdict(dict)

for train_data in ttg.training_set:
    dwell_time = 1 if train_data[0] >= required_dwell_time else 0
    depth = train_data[3]
    channel = train_data[8]
    
    dwell_depth_avg_gloabl[depth].append(dwell_time) 
    if depth in dwell_depth_avg_channel[channel]:
        dwell_depth_avg_channel[channel][depth].append(dwell_time) 
    else:
        dwell_depth_avg_channel[channel][depth] = [dwell_time]
      





for depth in dwell_depth_avg_gloabl:
    dwell_depth_avg_gloabl[depth] = mean(dwell_depth_avg_gloabl[depth]) # can change from "mean" to "median", "mean" has higher RMSD

for cha in dwell_depth_avg_channel:
    for dep in dwell_depth_avg_channel[cha]:
        dwell_depth_avg_channel[cha][dep] = mean(dwell_depth_avg_channel[cha][dep])



''' Classification - Log-loss '''
y_pred_global = []
y_pred_channel = []
y_true = []
for test_data in ttg.test_set:
    dwell_time = 1 if test_data[0] >= required_dwell_time else 0
    depth = test_data[3]
    channel = train_data[8]
    y_pred_global.append( dwell_depth_avg_gloabl[depth] )
    y_pred_channel.append( dwell_depth_avg_channel[channel][depth] )
    y_true.append( dwell_time ) 


print('Log-loss_gloablAVG =', log_loss(y_true, y_pred_global))
print('Log-loss_channelAVG =', log_loss(y_true, y_pred_channel))

