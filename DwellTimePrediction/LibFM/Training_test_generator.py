'''
Created on Apr 1, 2016

@author: Wang
'''
from pymongo import MongoClient
from BasicInvestigation.dwell_time_calculation import get_depth_dwell_time
from LibFM import FreqUserPageSearcher as fups
from _collections import defaultdict
from dateutil import parser
from bs4 import BeautifulSoup
from datetime import datetime


client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']
articleInfo = client['Forbes_Dec2015']['ArticleInfo']


training_set = []
test_set = []
''' The keys are all users in the training set (before applying threshold) '''
training_users_freq = defaultdict(int) 
''' The keys are all pages in the training set (before applying threshold) '''
training_pages_freq = defaultdict(int) 


def get_user_geo(country, state):
    if country != 'US':
        return country
    if country == 'US' and state:
        return '-'.join((country, state))
    else:
        return 'US'

def discretize_body_length(doc):
    if 'body' in doc:
        return str(int(len(BeautifulSoup(doc['body']).getText().split()) / 100))
    else:
        return 'unknown'

def get_freshness(doc, pv_start_time):
    if 'date' in doc:
        pub_time = int(str(doc['date'])[:-3])
        freshness_hour = str(int((pv_start_time - pub_time) / 3600))
        return freshness_hour
    else:
        return 'unknown'
    
def get_article_info(userlog_url, pv_start_time):
    for doc in articleInfo.find({'URL_IN_USERLOG':userlog_url}):
        
        body_length = discretize_body_length(doc)
        channel = doc['displayChannel'] if 'displayChannel' in doc else 'unknown'
        freshness = get_freshness(doc, pv_start_time)
        
        return body_length, channel.lower(), freshness
    return ['unknown'] * 3

user_num = 0
valid_pv_num = 0
train_pv_num = 0; train_dep_num = 0
test_pv_num = 0; test_dep_num = 0
channel_dist = defaultdict(int)

for user_doc in fups.freq_uids: # for each unique user
    uid = user_doc['uid']

#     if uid == "a805c8a0-22f9-5430-563f-603b9314c511":
#         print()
    
    user_num += 1
    print(user_num)
    
    """ for each page view """
    for pv_doc in userlog.find({'uid':uid}):
        url = pv_doc['url']
        unix_start_time = pv_doc['unix_start_time']
        
        """ if this page is not a frequent page """
        if url not in fups.freq_page_list:
            continue
        

        depth_dwell_time = get_depth_dwell_time(pv_doc['loglist'])
        ''' if this page view does not have a valid dwell time '''
        if depth_dwell_time is None:
            continue
        
        ''' this is a valid page view '''
        valid_pv_num += 1
        
        
        screen_size = pv_doc['loglist'][0]['additionalinfo']['screenSize'] if 'screenSize' in pv_doc['loglist'][0]['additionalinfo'] else 'unknown'
        viewport_size = pv_doc['loglist'][0]['additionalinfo']['viewportSize'] if 'viewportSize' in pv_doc['loglist'][0]['additionalinfo'] else 'unknown'
        user_geo = get_user_geo(pv_doc['country'], pv_doc['state'])
        body_length, channel, freshness = get_article_info(url, pv_doc['unix_start_time'])
#         print(screen_size, viewport_size, user_geo, body_length, channel, freshness)
        channel_dist[channel] += 1
        
        ''' distinguish training and test data '''
        if unix_start_time >= 1449792000 and unix_start_time < 1450569600:
            ''' training data '''
            train_pv_num += 1
            training_users_freq[uid] += 1
            training_pages_freq[url] += 1
            for d, t in depth_dwell_time.items():
                train_dep_num += 1
                training_set.append((t, uid, url, d, screen_size, viewport_size, user_geo, body_length, channel, freshness, pv_doc['ua']))
                
        elif unix_start_time >= 1450569600 and unix_start_time < 1450656000:
            ''' test data '''
            test_pv_num += 1
            for d, t in depth_dwell_time.items():
                test_dep_num += 1
                test_set.append((t, uid, url, d, screen_size, viewport_size, user_geo, body_length, channel, freshness, pv_doc['ua']))
                
        else:
            continue
          
        
print()
# print("valid_pv_num =", valid_pv_num)
print("train_pv_num =", train_pv_num, "; train_dep_num =", train_dep_num)
print("test_pv_num =", test_pv_num, "; test_dep_num =", test_dep_num)
print("initial training user num =", len(training_users_freq), "; initial training page num =", len(training_pages_freq))
print()


# for cha in channel_dist:
#     print(cha, ',', channel_dist[cha])


def filter_pageviews_by_minPVnum(dataset):
    filtered_dataset = []
    for data in dataset:
        uid = data[1]
        url = data[2]
        if ( training_users_freq[uid] < fups.COLD_START_THRESHOLD or 
             training_pages_freq[url] < fups.COLD_START_THRESHOLD ):
            continue
        filtered_dataset.append(data)
    return filtered_dataset


training_set = filter_pageviews_by_minPVnum(training_set)
test_set = filter_pageviews_by_minPVnum(test_set)


def _test_overlap(training_set, test_set):
    unique_training_users = set(); unique_training_pages = set()
    unique_test_users = set(); unique_test_pages = set()
    for data in training_set:
        unique_training_users.add(data[1])
        unique_training_pages.add(data[2])
    for data in test_set:
        unique_test_users.add(data[1])
        unique_test_pages.add(data[2])
        
    print()
    print("final pageview num:", len(training_set)/100)
    print("final training users:", len(unique_training_users), "final training pages:", len(unique_training_pages))
    print("final test users:", len(unique_test_users), "final test pages:", len(unique_test_pages))
    print("user overlap:", len(unique_training_users & unique_test_users), "page overlap:", len(unique_training_pages & unique_test_pages))
    
#     for abnormal_user in unique_test_users - unique_training_users:
#         print(abnormal_user)
    final_user_freq = defaultdict(int)
    final_page_freq = defaultdict(int)
    for data in training_set:
        final_user_freq[data[1]] += 1
        final_page_freq[data[2]] += 1
#     print(len(final_user_freq), len(final_page_freq))
    infreq_user_num = 0
    infreq_page_num = 0
    for user in final_user_freq:
        if final_user_freq[user] < fups.COLD_START_THRESHOLD:
            infreq_user_num += 1
    for page in final_page_freq:
        if final_page_freq[page] < fups.COLD_START_THRESHOLD:
            infreq_page_num += 1
    print(infreq_user_num, infreq_page_num)
    
    
_test_overlap(training_set, test_set)    

print()

