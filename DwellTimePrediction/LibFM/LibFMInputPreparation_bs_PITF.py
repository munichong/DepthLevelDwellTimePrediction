'''
Created on Mar 16, 2016

@author: Wang
'''
from pymongo import MongoClient
from BasicInvestigation.dwell_time_calculation import get_depth_dwell_time
from LibFM import FreqUserPageSearcher as fups
from _collections import defaultdict
from LibFM import Training_test_generator as ttg
from math import sqrt
from numpy import mean, median, array
from sklearn.metrics import mean_squared_error


client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']


user_txt_ouput = open('../data_bs/user.libfm', 'w')
page_txt_ouput = open('../data_bs/page.libfm', 'w')
depth_txt_output = open('../data_bs/depth.libfm', 'w')
screen_txt_output = open('../data_bs/screen.libfm', 'w')
viewport_txt_output = open('../data_bs/viewport.libfm', 'w')
geo_txt_output = open('../data_bs/geo.libfm', 'w')
length_txt_output = open('../data_bs/length.libfm', 'w')
channel_txt_output = open('../data_bs/channel.libfm', 'w')
fresh_txt_output = open('../data_bs/fresh.libfm', 'w')


user_train_output = open('../data_bs/user.train', 'w')
page_train_output = open('../data_bs/page.train', 'w')
depth_train_output = open('../data_bs/depth.train', 'w')
screen_train_output = open('../data_bs/screen.train', 'w')
viewport_train_output = open('../data_bs/viewport.train', 'w')
geo_train_output = open('../data_bs/geo.train', 'w')
length_train_output = open('../data_bs/length.train', 'w')
channel_train_output = open('../data_bs/channel.train', 'w')
fresh_train_output = open('../data_bs/fresh.train', 'w')
y_train_output = open('../data_bs/y.train', 'w')

user_test_output = open('../data_bs/user.test', 'w')
page_test_output = open('../data_bs/page.test', 'w')
depth_test_output = open('../data_bs/depth.test', 'w')
screen_test_output = open('../data_bs/screen.test', 'w')
viewport_test_output = open('../data_bs/viewport.test', 'w')
geo_test_output = open('../data_bs/geo.test', 'w')
length_test_output = open('../data_bs/length.test', 'w')
channel_test_output = open('../data_bs/channel.test', 'w')
fresh_test_output = open('../data_bs/fresh.test', 'w')
y_test_output = open('../data_bs/y.test', 'w')



''' write depths to depth.txt '''
for d in range(101):
    depth_txt_output.write('0 ' + str(d) + ':1\n')
depth_txt_output.close()  


def write_new_featval(featval, lookup, output):
    if featval not in lookup:
        lookup[featval] = len(lookup)
        ''' append this new user in user.txt '''
        output.write('0 ' + str(lookup[featval]) + ':1\n')    


''' get index lookup table for each feature '''
user_index_lookup = defaultdict(int)
page_index_lookup = defaultdict(int)
screen_index_lookup = defaultdict(int)
viewport_index_lookup = defaultdict(int)
geo_index_lookup = defaultdict(int)
length_index_lookup = defaultdict(int)
channel_index_lookup = defaultdict(int)
fresh_index_lookup = defaultdict(int)

dwell_depth_avg = defaultdict(list)

print("PITF: Iterating through training data")
for train_data in ttg.training_set:
    dwell, uid, url, depth, screen, viewport, user_geo, body_length, channel, freshness, _ = train_data
    
    write_new_featval(uid, user_index_lookup, user_txt_ouput)
    write_new_featval(url, page_index_lookup, page_txt_ouput)
    write_new_featval(screen, screen_index_lookup, screen_txt_output)
    write_new_featval(viewport, viewport_index_lookup, viewport_txt_output)
    write_new_featval(user_geo, geo_index_lookup, geo_txt_output)
    write_new_featval(body_length, length_index_lookup, length_txt_output)
    write_new_featval(channel, channel_index_lookup, channel_txt_output)
    write_new_featval(freshness, fresh_index_lookup, fresh_txt_output)
    
    
    user_train_output.write(str(user_index_lookup[uid]) + '\n')
    page_train_output.write(str(page_index_lookup[url]) + '\n')
    depth_train_output.write(str(depth) + '\n')
    screen_train_output.write(str(screen_index_lookup[screen]) + '\n')
    viewport_train_output.write(str(viewport_index_lookup[viewport]) + '\n')
    geo_train_output.write(str(geo_index_lookup[user_geo]) + '\n')
    length_train_output.write(str(length_index_lookup[body_length]) + '\n')
    channel_train_output.write(str(channel_index_lookup[channel]) + '\n')
    fresh_train_output.write(str(fresh_index_lookup[freshness]) + '\n')
    
    y_train_output.write(str(dwell) + '\n')

    dwell_depth_avg[depth].append(dwell) 



for depth in dwell_depth_avg:
    dwell_depth_avg[depth] = mean(dwell_depth_avg[depth]) # can change from "mean" to "median", "mean" has higher RMSD
y_pred_globalAVG = []
y_true_globalAVG = []    


print("PITF: Iterating through test data")
for test_data in ttg.test_set:
    dwell, uid, url, depth, screen, viewport, user_geo, body_length, channel, freshness, _ = test_data
    
    if ( uid not in user_index_lookup or url not in page_index_lookup or 
         viewport not in viewport_index_lookup or 
         user_geo not in geo_index_lookup or 
         body_length not in length_index_lookup
         ):
        continue
    
    user_test_output.write(str(user_index_lookup[uid]) + '\n')
    page_test_output.write(str(page_index_lookup[url]) + '\n')
    depth_test_output.write(str(depth) + '\n')
    screen_test_output.write(str(screen_index_lookup[screen]) + '\n')
    viewport_test_output.write(str(viewport_index_lookup[viewport]) + '\n')
    geo_test_output.write(str(geo_index_lookup[user_geo]) + '\n')
    length_test_output.write(str(length_index_lookup[body_length]) + '\n')
    channel_test_output.write(str(channel_index_lookup[channel]) + '\n')
    fresh_test_output.write(str(fresh_index_lookup[freshness]) + '\n')
    
    y_test_output.write(str(dwell) + '\n')
    
    y_pred_globalAVG.append( dwell_depth_avg[depth] )
    y_true_globalAVG.append( dwell ) 

print("Finish outputting\n")

print('RMSD_gloablAVG =', sqrt(mean_squared_error(y_true_globalAVG, y_pred_globalAVG)))


