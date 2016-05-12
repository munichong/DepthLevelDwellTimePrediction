'''
Created on Mar 16, 2016

@author: Wang
'''
from pymongo import MongoClient, HASHED
from _collections import defaultdict


COLD_START_THRESHOLD = 5


client = MongoClient()
user_freq_table = client['Forbes_Dec2015']['UserFreq_training']
page_freq_table = client['Forbes_Dec2015']['PageFreq_training']

def get_freq_uids(threshold=0):
    return user_freq_table.find({"freq": {"$gte":threshold}}, {'uid':1})

def get_freq_urls(threshold=0):
    return page_freq_table.find({"freq": {"$gte":threshold}}, {'url':1})

freq_uids = get_freq_uids(COLD_START_THRESHOLD)
freq_urls = get_freq_urls(COLD_START_THRESHOLD)

print(freq_uids.count(), "unique users")
print(freq_urls.count(), "unique urls")
print()


freq_page_list = []
for page_doc in freq_urls:
    freq_page_list.append(page_doc['url'])