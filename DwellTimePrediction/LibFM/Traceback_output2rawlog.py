'''
Created on Apr 1, 2016

@author: Wang
'''
import re
from math import sqrt
from pprint import pprint
from random import random, sample
from collections import Counter
from pymongo import MongoClient
from BasicInvestigation.dwell_time_calculation import get_depth_dwell_time
from LibFM import FreqUserPageSearcher as fups
from _collections import defaultdict
from bs4 import BeautifulSoup
# from itertools import ifilter

  

client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']
articleInfo = client['Forbes_Dec2015']['ArticleInfo']

# 
# class outfile_pageview:
#     def __init__(self):
#         self.rows = []
# 
# 
# with open("I:/Desktop/FFMInput_dwells_train.csv") as training_file:
#     n = 0
#     for line in training_file:
#         n+= 1
# print("The training file has", n/101.0, "pageviews")
# 
# print("Reading training_pv_fromfile\n")
# training_pv_fromfile = {} # targets : outfile_pageview
# with open("I:/Desktop/FFMInput_dwells_train.csv") as training_file:
#     m = 0
#     dup = 0
#     while m < n/101.0:
#         m += 1
#         pv = outfile_pageview()
#         targets = []
#         for line in training_file:
#             line = line.strip()
#             pv.rows.append(line)
#             targets.append(int(line.split(" ")[0]))
#             if len(targets) == 101:
#                 break
#         targets = tuple(targets)
#         if targets in training_pv_fromfile:
#             dup += 1
# #             print(targets)
# #             
# #             for i in range(101):
# #                 print(pv.rows[i][:100])
# #                 print(training_pv_fromfile[targets].rows[i][:100])
# #                 print()
# #             print()
#         else:
#             training_pv_fromfile[targets] = pv
# 
# print(dup, "duplicate targets in", m, 'pageviews')
#        
# with open("I:/Desktop/FFMInput_dwells_test.csv") as test_file:
#     n = 0
#     for line in test_file:
#         n+= 1
# print("The test file has", n/101.0, "training pageviews\n")
# 
# print("Reading test_pv_fromfile")
# test_pv_fromfile = {} # targets : outfile_pageview
# with open("I:/Desktop/FFMInput_dwells_test.csv") as test_file:
#     m = 0
#     dup = 0
#     while m < n/101.0:
#         m += 1
#         pv = outfile_pageview()
#         targets = []
#         for line in test_file:
#             line = line.strip()
#             pv.rows.append(line)
#             targets.append(int(line.split(" ")[0]))
#             if len(targets) == 101:
#                 break
#         targets = tuple(targets)
#         if targets in test_pv_fromfile:
#             dup += 1
# #             print(targets)
# #             
# #             for i in range(101):
# #                 print(pv.rows[i][:100])
# #                 print(test_pv_fromfile[targets].rows[i][:100])
# #                 print()
# #             print()
#         else:
#             test_pv_fromfile[targets] = pv
# 
# print(dup, "duplicate targets in", m, 'test pageviews\n')


def get_user_geo(country, state):
    if country != 'US':
        return country
    if country == 'US' and state:
        return '-'.join((country, state))
    else:
        return 'US'

# def discretize_body_length(doc):
#     if 'body' in doc:
#         return str(int(len(BeautifulSoup(doc['body']).getText().split()) / 100))
#     else:
#         return 'unknown'

def get_freshness(doc, pv_start_time):
    if 'date' in doc:
        pub_time = int(str(doc['date'])[:-3])
        freshness_hour = int((pv_start_time - pub_time) / 86400) # unit: day
        if freshness_hour > 10:
            freshness_hour = '>10d'
        return str(freshness_hour)
    else:
        return 'unknown'

def get_channel_group(doc):
    if 'channelSection' not in doc:
        return ['unknown']
    channel_group = set()
    for chanSec_dict in doc['channelSection']:
        channel_group.add(chanSec_dict['channelId'].lower())
    return list(channel_group)

def get_article_info(userlog_url, pv_start_time):
    for doc in articleInfo.find({'URL_IN_USERLOG':userlog_url}):
        
        if 'body' in doc:
            body_text = BeautifulSoup(doc['body']).getText()
            body_text = re.sub(r'\[.*?\]', ' ', body_text)
#             print(body_text)
            body_length = str(int(len(body_text.split()) / 100))
        else:
            body_text = 'unknown'
            body_length = 'unknown'
        channel = doc['displayChannel'] if 'displayChannel' in doc else 'unknown'
        channel_group = get_channel_group(doc) # list
        freshness = get_freshness(doc, pv_start_time)
        
        return body_length, channel.lower(), channel_group, freshness
    return ['unknown', 'unknown', ['unknown'], 'unknown']

def get_body_text(userlog_url):
    for doc in articleInfo.find({'URL_IN_USERLOG':userlog_url}):
        
        if 'body' in doc:
            body_text = BeautifulSoup(doc['body']).getText()
            body_text = re.sub(r'\[.*?\]', ' ', body_text)
        else:
            body_text = 'unknown'
        
        return body_text.lower()
    return 'unknown'


class Pageview:
    def __init__(self, uid, url, depth_dwell_time, **auxiliary):
        self.uid = uid
        self.url = url
        self.screen, self.viewport = auxiliary['screen'], auxiliary['viewport']
        self.create_depth_level(uid, url, depth_dwell_time, auxiliary)
    
    def create_depth_level(self, uid, url, depth_dwell_time, auxiliary):
        """ convert pageview level data to depth level data
        depth level data are used to train the predictive model """
        self.depth_level_rows = []
        for depth, (dwell, top, bottom) in depth_dwell_time.items():
#             area_text = self.get_area_text(top, bottom, auxiliary['fulltext'])
            self.depth_level_rows.append((dwell, uid, url, depth, top, bottom, 
                                          auxiliary['screen'], auxiliary['viewport'], 
                                          auxiliary['geo'], auxiliary['agent'],
                                          auxiliary['weekday'], auxiliary['hour'], 
                                          auxiliary['length'], auxiliary['channel'],
                                          auxiliary['channel_group'], auxiliary['fresh']
                                          ))

    
        

user_num = 0
valid_pv_num = 0
user_freq = defaultdict(int); page_freq = defaultdict(int);


all_pageviews = []
for user_doc in fups.freq_uids: # for each unique user
    uid = user_doc['uid']
    
    user_num += 1
    print(user_num)
    
    """ for each page view """
    for pv_doc in userlog.find({'uid':uid}):
        url = pv_doc['url']
        unix_start_time = pv_doc['unix_start_time']
              
         
        if unix_start_time < 1449792000 or unix_start_time > 1450656000:
            continue
        
        """ if this page is not a frequent page """
        if url not in fups.freq_page_set:
            continue
        

        depth_dwell_time, pv_summary = get_depth_dwell_time(pv_doc['loglist']) # depth_dwell_time: dict: depth -> (dwell, screen_top, screen_bottom)
        ''' if this page view does not have a valid dwell time '''
        if depth_dwell_time is None:
            continue
        
        targets = [0] * 101
        for d in depth_dwell_time:
            targets[d] = depth_dwell_time[d][0]
        print(pv_summary)
        print(targets)
        
        
        
        ''' AUXILIARY FEATURES '''
        screen_size = pv_doc['loglist'][0]['additionalinfo']['screenSize'] if 'screenSize' in pv_doc['loglist'][0]['additionalinfo'] else 'unknown'
        
        viewport_size = pv_doc['loglist'][0]['additionalinfo']['viewportSize'] if 'viewportSize' in pv_doc['loglist'][0]['additionalinfo'] else 'unknown'
        
        user_geo = get_user_geo(pv_doc['country'], pv_doc['state'])
        
        isodate = pv_doc['local_start_time']
        if isodate:
            local_weekday, local_hour = str(isodate.weekday()), str(isodate.hour) # The range of weekday is [0, 6]
        else:
            local_weekday, loca_hour = 'unknown', 'unknown'
        
        
        body_length, channel, channel_group, freshness = get_article_info(url, pv_doc['unix_start_time'])

#         length_dist[body_length] += 1
#         channel_dist[channel] += 1
#         fresh_dist[freshness] += 1
    
        
        ''' this is a valid page view '''
        valid_pv_num += 1
        user_freq[uid] += 1
        page_freq[url] += 1
        
        pageview = Pageview(uid, url, depth_dwell_time, screen=screen_size, 
                            viewport=viewport_size, geo=user_geo, agent=pv_doc['ua'], 
                            weekday=local_weekday, hour=local_hour, 
                            length=body_length, channel=channel, channel_group=channel_group,
                            fresh=freshness
                            )
        all_pageviews.append(pageview)
        
                              
        
print()
print("=============== Statistics of Initial Data ================")
print("valid_pv_num =", valid_pv_num)
print(len(user_freq), " unique users and ", len(page_freq), " unique pages")
print("density =", valid_pv_num/float(len(user_freq) * len(page_freq)))
print()



user_freq2 = defaultdict(int)
page_freq2 = defaultdict(int)
valid_pv_num2 = 0
def filter_pageviews_by_minPVnum(pvs):
    filtered_dataset = []
    global valid_pv_num2, user_freq2, page_freq2
    for pv in pvs:
        uid = pv.uid
        url = pv.url
        if ( user_freq[uid] < fups.COLD_START_THRESHOLD or 
             page_freq[url] < fups.COLD_START_THRESHOLD ):
            continue
        user_freq2[uid] += 1
        page_freq2[url] += 1 
        valid_pv_num2 += 1
        filtered_dataset.append(pv)
    return filtered_dataset


filtered_pageviews = filter_pageviews_by_minPVnum(all_pageviews)

print()
print("=============== Statistics of Further Data ================")
print("valid_pv_num2 =", valid_pv_num2)
print(len(user_freq2), "unique users and", len(page_freq2), "unique pages")
print("density =", valid_pv_num2/float(len(user_freq2) * len(page_freq2)))
print()

print("(count, freqOfCount)")
print( Counter(user_freq2.values()).most_common() )
print( Counter(page_freq2.values()).most_common() )

print(sorted(Counter(user_freq2.values()).most_common(), key=lambda x:x[0], reverse=False))
print(sorted(Counter(page_freq2.values()).most_common(), key=lambda x:x[0], reverse=False))




print("\n=============== Separating Training and Test Data ================")
""" Randomly pick training and test instances """
training_set = []
test_set = []
all_training_text = defaultdict(str) # url -> body_text
all_test_text = defaultdict(str)
# users_have_been_in_test = set()
# pages_have_been_in_test = set()
for pv in filtered_pageviews:
#     if ( user_freq2[uid] < fups.COLD_START_THRESHOLD or 
#         page_freq2[url] < fups.COLD_START_THRESHOLD ):
#         continue
    body_text = get_body_text(pv.url)
    # if len(test_set) / float(len(training_set) + len(test_set)) < 0.1 and 
    if ( user_freq2[pv.uid] > fups.COLD_START_THRESHOLD and 
         page_freq2[pv.url] > fups.COLD_START_THRESHOLD ):
        test_set.append(pv)
        if body_text != 'unknown':
            all_test_text[pv.url] = body_text
        user_freq2[pv.uid] -= 1
        page_freq2[pv.url] -= 1
    else:
        training_set.append(pv)
        if body_text != 'unknown':
            all_training_text[pv.url] = body_text
            
        
print()
print(len(training_set), "pageviews in the training set")
print(len(test_set), "pageviews in the test set")
print("The fraction is", len(test_set) / float(valid_pv_num2))
print()

