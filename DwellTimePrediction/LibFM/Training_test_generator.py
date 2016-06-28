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
from dateutil import parser
from bs4 import BeautifulSoup
from datetime import datetime
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim import corpora, similarities, models
import matplotlib.pyplot as plt
from bottleneck.nonreduce import replace
from itertools import ifilter

  

client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']
articleInfo = client['Forbes_Dec2015']['ArticleInfo']


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
    return ['unknown'] * 4

def get_body_text(userlog_url):
    for doc in articleInfo.find({'URL_IN_USERLOG':userlog_url}):
        
        if 'body' in doc:
            body_text = BeautifulSoup(doc['body']).getText()
            body_text = re.sub(r'\[.*?\]', ' ', body_text)
        else:
            body_text = 'unknown'
        
        return body_text
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

# length_dist = defaultdict(int)
# channel_dist = defaultdict(int)
# fresh_dist = defaultdict(int)

porter_stemmer = PorterStemmer()

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
        

        depth_dwell_time = get_depth_dwell_time(pv_doc['loglist']) # depth -> (dwell, screen_top, screen_bottom)
        ''' if this page view does not have a valid dwell time '''
        if depth_dwell_time is None:
            continue
        
        
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
        
#         ''' distinguish training and test data '''
#         if unix_start_time >= 1449792000 and unix_start_time < 1450569600:
#             ''' training data '''
#             train_pv_num += 1
#             training_users_freq[uid] += 1
#             training_pages_freq[url] += 1
#             for d, t in depth_dwell_time.items():
#                 train_dep_num += 1
#                 training_set.append((t, uid, url, d, screen_size, viewport_size, user_geo, pv_doc['ua']))
#               
#         elif unix_start_time >= 1450569600 and unix_start_time < 1450656000:
#             ''' test data '''
#             test_pv_num += 1
#             for d, t in depth_dwell_time.items():
#                 test_dep_num += 1
#                 test_set.append((t, uid, url, d, screen_size, viewport_size, user_geo, pv_doc['ua']))
# #                 test_set.append((t, uid, url, d, screen_size, viewport_size, user_geo, body_length, channel, freshness, pv_doc['ua']))
                
                              
        
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





# def _test_overlap(training_set, test_set):
#     unique_training_users = set(); unique_training_pages = set()
#     unique_test_users = set(); unique_test_pages = set()
#     for data in training_set:
#         unique_training_users.add(data[1])
#         unique_training_pages.add(data[2])
#     for data in test_set:
#         unique_test_users.add(data[1])
#         unique_test_pages.add(data[2])
#         
#     print()
#     print("final pageview num:", len(training_set)/100)
#     print("final training users:", len(unique_training_users), "final training pages:", len(unique_training_pages))
#     print("final test users:", len(unique_test_users), "final test pages:", len(unique_test_pages))
#     print("user overlap:", len(unique_training_users & unique_test_users), "page overlap:", len(unique_training_pages & unique_test_pages))
#     print("density =", len(training_set)/float(100 * len(unique_training_users) * len(unique_training_pages)))
#     
# #     for abnormal_user in unique_test_users - unique_training_users:
# #         print(abnormal_user)
#     final_user_freq = defaultdict(int)
#     final_page_freq = defaultdict(int)
#     for data in training_set:
#         final_user_freq[data[1]] += 1
#         final_page_freq[data[2]] += 1
# #     print(len(final_user_freq), len(final_page_freq))
#     infreq_user_num = 0
#     infreq_page_num = 0
#     for user in final_user_freq:
#         if final_user_freq[user] < fups.COLD_START_THRESHOLD:
#             infreq_user_num += 1
#     for page in final_page_freq:
#         if final_page_freq[page] < fups.COLD_START_THRESHOLD:
#             infreq_page_num += 1
#     print(infreq_user_num, infreq_page_num)
#     
#     
# _test_overlap(training_set, test_set)    
# 
# print()


print("\n=============== Separating Training and Test Data ================")
""" Randomly pick 10% training and test instances """
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
     
    if ( user_freq2[pv.uid] > fups.COLD_START_THRESHOLD and 
         page_freq2[pv.url] > fups.COLD_START_THRESHOLD ):
        test_set.append(pv)
        all_test_text[pv.url] = get_body_text(pv.url)
        user_freq2[pv.uid] -= 1
        page_freq2[pv.url] -= 1
    else:
        training_set.append(pv)
        all_training_text[pv.url] = get_body_text(pv.url)
        
print()
print(len(training_set), "pageviews in the training set")
print(len(test_set), "pageviews in the test set")
print("The fraction is", len(test_set) / float(valid_pv_num2))
print()





""" Build LDA Model """
clean_tokens = []
for text in all_training_text.values():
    clean_tokens.append( [porter_stemmer.stem(t) for t in 
                          ifilter(lambda w : w not in stopwords.words('english') 
                                  and len(w) > 1 and not w.isdigit(), 
                                  word_tokenize(text))] )


print("\nBuilding LDA dictionary...")
dictionary = corpora.Dictionary(clean_tokens)
print("Converting doc to bow...")
corpus = [dictionary.doc2bow(d) for d in clean_tokens]
print("Creating the MM file...")
corpora.MmCorpus.serialize('lda.mm', corpus)
mm = corpora.MmCorpus('lda.mm')
print("Building a LDA model for", int(sqrt(len(all_training_text))), "topics ...")
lda = models.ldamodel.LdaModel(corpus=mm, num_topics=int(sqrt(len(all_training_text))))
print("A LDA model is built.\n")

doc_lda = lda[dictionary.doc2bow(clean_tokens[0])]
pprint(doc_lda)

"""
doc_lda = 
[(0, 0.81995379945948721),
 (1, 0.020003070630106111),
 (2, 0.020007821929791454),
 (3, 0.020005859522950798),
 (4, 0.020007262959776777),
 (5, 0.020002932567543413),
 (6, 0.020003773871628584),
 (7, 0.020005253893555897),
 (8, 0.02000451943754944),
 (9, 0.020005705727610398)]
"""


""" Calculate TF-IDF over all pages """
print("Calculating TF-IDF for", len(all_training_text), "pages")
tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=0, max_features=500,
                                   token_pattern=r'(?u)\b[a-zA-Z]+\b')
tfidf_vectorizer.fit(all_training_text.values())





""" Plot the distribution of screen and viewport """    
# screen_width_list = []
# screen_height_list = []
# viewport_width_list = []
# viewport_height_list = []
# # Check the distribution of screen and viewport
# for pv in filtered_pageviews:
#     if pv.screen == 'unknown' or pv.viewport == 'unknown':
#         continue
#     screen_width, screen_height = [int(pixel) for pixel in pv.screen.split('x')]
#     screen_width_list.append(screen_width)
#     screen_height_list.append(screen_height)
#     viewport_width, viewport_height = [int(pixel) for pixel in pv.viewport.split('x')]
#     viewport_width_list.append(viewport_width)
#     viewport_height_list.append(viewport_height)
#     
# 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.boxplot([screen_width_list, screen_height_list, viewport_width_list, viewport_height_list])
# plt.show()


""" Print the distribution of length, channel, and freshness """
# def print_dict(d):
#     for k, v in d.items():
#         print(str(k)+','+str(v))
#     print()
# 
# print("The distribution of LENGTH")
# print_dict(length_dist)
# print("The distribution of CHANNEL")
# print_dict(channel_dist)
# print("The distribution of FRESHNESS")
# print_dict(fresh_dist)


