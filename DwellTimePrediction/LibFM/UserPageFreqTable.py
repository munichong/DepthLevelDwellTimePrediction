'''
Created on Mar 16, 2016

@author: Wang
'''
import sys
from pymongo import MongoClient, HASHED
from _collections import defaultdict

TRAINING_START = 1449792000 # 12/11/2015
''' TRAINING_END = 1450569600 # 12/20/2015 '''
TEST_END = 1450656000 # 12/21/2015 '''

client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']



user_freq_table = client['Forbes_Dec2015']['UserFreq_all'] # NEW; WAS: UserFreq_training
user_freq_table.create_index([('uid', HASHED)])
user_freq_table.create_index([('freq', 1)])
unique_uid = defaultdict(int)
  
n = 0
for pv_doc in userlog.find({'unix_start_time': {"$gte":TRAINING_START, "$lte":TEST_END}}):
#     print(pv_doc['unix_start_time'])
    if pv_doc['unix_start_time'] < TRAINING_START or pv_doc['unix_start_time'] > TEST_END:
        sys.exit()
    uid = pv_doc['uid']
    unique_uid[uid] += 1
    n+=1
    print(n)
      
print(len(unique_uid), "unique users")
      
n = 0
for uid, user_freq in unique_uid.items():
    if user_freq <= 1:
        continue
    user_freq_table.insert({"uid": uid, "freq":user_freq})
    n+=1
    print(n)
   
unique_uid.clear()



# unique_url = defaultdict(int)
# 
# page_freq_table = client['Forbes_Dec2015']['PageFreq_all'] # NEW; WAS: PageFreq_training
# page_freq_table.create_index([('url', HASHED)])
# page_freq_table.create_index([('freq', 1)])
# 
# n = 0
# for pv_doc in userlog.find({'unix_start_time': {"$gte":TRAINING_START, "$lte":TEST_END}}):
#     if pv_doc['unix_start_time'] < TRAINING_START or pv_doc['unix_start_time'] > TEST_END:
#         sys.exit()
#     url = pv_doc['url']
#     unique_url[url] += 1
#     n+=1
#     print(n)
#     
# print(len(unique_url), "unique pages")
#     
# n = 0
# for url, page_freq in unique_url.items():
#     if page_freq <= 1:
#         continue
#     page_freq_table.insert({"url": url, "freq":page_freq})
#     n+=1
#     print(n)
    