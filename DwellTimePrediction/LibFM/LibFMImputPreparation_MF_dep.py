'''
Created on Mar 25, 2016

@author: Wang
'''
from pymongo import MongoClient
from BasicInvestigation.dwell_time_calculation import get_depth_dwell_time
from LibFM import FreqUserPageSearcher as fups
from networkx.algorithms.link_analysis import pagerank_alg

client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']


train_output = open('../data_bs/train.libfm', 'w')
test_output = open('../data_bs/test.libfm', 'w')



train_pv_num = 0
test_pv_num = 0
valid_pv_num = 0
train_dep_num = 0; test_dep_num = 0
user_lookup_table = {}
page_lookup_table = {}
train_examples = []
test_examples = []
for user_doc in fups.freq_uids: # for each unique user
    uid = user_doc['uid']
    user_has_valid_pv = False
    
    """ for each pageview """
    for pv_doc in userlog.find({'uid':uid}):
        url = pv_doc['url']
        unix_start_time = pv_doc['unix_start_time']
        
        """ if this page is not a frequent page """
        if url not in fups.freq_page_list:
            continue
        
        
        depth_dwell_time = get_depth_dwell_time(pv_doc['loglist'])
#         print(depth_dwell_time)
#         print()
        if depth_dwell_time is None:
            continue
        
        ''' this is a valid page view '''
        valid_pv_num += 1
        user_has_valid_pv = True
        
        
        
        if url not in page_lookup_table:
            page_lookup_table[url] = len(page_lookup_table)
        
        
        ''' distinguish training and test data '''
        if unix_start_time >= 1449792000 and unix_start_time < 1450569600:
            ''' training data '''
            train_pv_num += 1
            for d, t in depth_dwell_time.items():
                train_examples.append( (t, uid, url, d) )
                train_dep_num += 1
        elif unix_start_time >= 1450569600 and unix_start_time < 1450656000:
            ''' test data '''
            test_pv_num += 1
            for d, t in depth_dwell_time.items():
                test_examples.append( (t, uid, url, d) )
                test_dep_num += 1
        else:
            continue
        
        
    if user_has_valid_pv is True:
        user_lookup_table[uid] = len(user_lookup_table)
        print(len(user_lookup_table))
            
            

print("valid_pv_num =", valid_pv_num)
print("train_pv_num =", train_pv_num, "; train_dep_num =", train_dep_num)
print("test_pv_num =", test_pv_num, "; test_dep_num =", test_dep_num)
print()



for t, uid, url, d in train_examples:
    train_output.write(str(t) + ' ' + str(user_lookup_table[uid]) + ':1 ' + str(len(user_lookup_table) + page_lookup_table[url]) + ':1 ' + 
                       str(len(user_lookup_table) + len(page_lookup_table)) + ':' + str(d) + '\n')
    
for t, uid, url, d in test_examples:
    test_output.write(str(t) + ' ' + str(user_lookup_table[uid]) + ':1 ' + str(len(user_lookup_table) + page_lookup_table[url]) + ':1 ' + 
                       str(len(user_lookup_table) + len(page_lookup_table)) + ':' + str(d) + '\n')