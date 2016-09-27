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
from pymongo.errors import CursorNotFound
from dwell_time_calculation import viewport_behaviors
from _collections import defaultdict
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from gensim import corpora, models
import matplotlib.pyplot as plt
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from user_agents import parse
  

client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']
articleInfo = client['Forbes_Dec2015']['ArticleInfo']


COLD_START_THRESHOLD = 23

MAX_SEQ_LEN = 20


client = MongoClient()
user_freq_table = client['Forbes_Dec2015']['UserFreq_all']
page_freq_table = client['Forbes_Dec2015']['PageFreq_all']

def get_freq_uids(threshold=0):
    return user_freq_table.find({"freq": {"$gte":threshold}}, {'uid':1})

def get_freq_urls(threshold=0):
    return page_freq_table.find({"freq": {"$gte":threshold}}, {'url':1})

# freq_uids = get_freq_uids(COLD_START_THRESHOLD) # The uids of freq users
# freq_urls = get_freq_urls(COLD_START_THRESHOLD) # The urls of freq pages
# 
# print(freq_uids.count(), "unique users")
# print(freq_urls.count(), "unique urls")
# print()


freq_page_set = set()
for page_doc in get_freq_urls(COLD_START_THRESHOLD):
    freq_page_set.add(page_doc['url'])

def simplify_version(raw_version):
    if '.' in raw_version:
        return raw_version[:raw_version.index('.')]
    return raw_version
    
def get_info_from_agent(ua_string):
    user_agent = parse(ua_string)
#     print(str(user_agent))
    return [simplify_version(string) for string in str(user_agent).split(' / ')]
#     print(user_agent.device.family, user_agent.device.brand, user_agent.device.model)
#     return (' '.join([user_agent.device.family, user_agent.device.brand, user_agent.device.model]),
#             ' '.join([user_agent.os.family, simplify_version(user_agent.os.version_string)]),
#             ' '.join([user_agent.browser.family, simplify_version(user_agent.browser.version_string)])
#             )

def get_user_geo(country, state):
    if country and country != 'US':
        return country
    if country == 'US' and state:
        return '-'.join((country, state))
    else:
        return 'US'


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

def get_section_group(doc):
    if 'channelSection' not in doc:
        return ['unknown']
    section_group = set()
    for chanSec_dict in doc['channelSection']:
        if 'sectionId' in chanSec_dict:
            section_group.add(chanSec_dict['sectionId'].lower())
    return list(section_group)

def get_article_info(userlog_url, pv_start_time):
    for doc in articleInfo.find({'URL_IN_USERLOG':userlog_url}):
        
        if 'body' in doc:
            try:
                body_text = BeautifulSoup(doc['body']).getText()
            except TypeError:
                break
            
            body_text = re.sub(r'\[.*?\]', ' ', body_text)
#             print(body_text)
            body_length = str(int(len(body_text.split()) / 100))
        else:
            body_text = 'unknown'
            body_length = 'unknown'
        channel = doc['displayChannel'] if 'displayChannel' in doc else 'unknown'
        section = doc['displaySection'] if 'displaySection' in doc else 'unknown'
        channel_group = get_channel_group(doc) # list
        section_group = get_section_group(doc) # list
        freshness = get_freshness(doc, pv_start_time)
        
        return body_length, channel.lower(), section.lower(), channel_group, section_group, freshness
#     return ['unknown', 'unknown', 'unknown', ['unknown'], ['unknown'], 'unknown']
    return None
    
    
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
    def __init__(self, uid, url, viewport_dwell, **auxiliary):
        self.uid = uid
        self.url = url
        self.screen, self.viewport = auxiliary['screen'], auxiliary['viewport']
        self.create_depth_level(uid, url, viewport_dwell, auxiliary)
    
    def create_depth_level(self, uid, url, viewport_dwell, auxiliary):
        """ convert pageview level data to depth level data
        depth level data are used to train the predictive model """
        self.depth_level_rows = []
        for top, bottom, dwell in viewport_dwell:
            self.depth_level_rows.append((dwell, uid, url, top, bottom, 
                                          auxiliary['screen'], auxiliary['viewport'], 
                                          auxiliary['geo'], auxiliary['agent'],
                                          auxiliary['weekday'], auxiliary['hour'], 
                                          auxiliary['length'], auxiliary['channel'], auxiliary['section'],
                                          auxiliary['channel_group'], auxiliary['section_group'],
                                          auxiliary['fresh'], auxiliary['device'], auxiliary['os'],
                                          auxiliary['browser']
                                          ))

    
        


valid_pv_num = 0
user_freq = defaultdict(int); page_freq = defaultdict(int);

# length_dist = defaultdict(int)
# channel_dist = defaultdict(int)
# fresh_dist = defaultdict(int)

user_num = 0
all_pageviews = []
done = False
while not done:
    try:
        freq_uids = get_freq_uids(COLD_START_THRESHOLD)
        freq_uids.skip(user_num)
        for user_doc in freq_uids: # for each unique user
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
#                 if url not in freq_page_set:
#                     continue
                
        
                viewport_dwell = viewport_behaviors(pv_doc['loglist']) # [[screen_top, screen_bottom, dwell_time], [ ... ]]
                
                ''' if this page view has too many time steps '''
                if viewport_dwell is None or len(viewport_dwell) > MAX_SEQ_LEN:
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
                
                article_info = get_article_info(url, pv_doc['unix_start_time'])
                if not article_info:
                    continue
                body_length, channel, section, channel_group, section_group, freshness = article_info
                
                
                device, os, browser = get_info_from_agent(pv_doc['ua'])
                
        
        #         length_dist[body_length] += 1
        #         channel_dist[channel] += 1
        #         fresh_dist[freshness] += 1
                
            
                
                ''' this is a valid page view '''
                valid_pv_num += 1
                user_freq[uid] += 1
                page_freq[url] += 1
                
                pageview = Pageview(uid, url, viewport_dwell, screen=screen_size, 
                                    viewport=viewport_size, geo=user_geo, agent=pv_doc['ua'], 
                                    weekday=local_weekday, hour=local_hour, 
                                    length=body_length, channel=channel, section=section,
                                    channel_group=channel_group, section_group=section_group, 
                                    fresh=freshness, device=device, os=os, browser=browser
                                    )
                all_pageviews.append(pageview)
        done = True
    except CursorNotFound:
        print("pymongo.errors.CursorNotFound")
        print("Will start from", user_num)
        
print()
print("=============== Statistics of Initial Data ================")
print("valid_pv_num =", valid_pv_num)
print(len(user_freq), " unique users and ", len(page_freq), " unique pages")
print("density =", valid_pv_num/float(len(user_freq) * len(page_freq)))
print()



pvbah_len_dist = defaultdict(int)
depth_dwell_dist = defaultdict(int)
for pv in all_pageviews:
    pvbah_length = len(pv.depth_level_rows)
    pvbah_len_dist[pvbah_length] += 1
    
    for action_tuple in pv.depth_level_rows:
        depth_dwell = action_tuple[0]
        depth_dwell_dist[depth_dwell] += 1

print("\n*************** The Distribution of The Action Number of A Pageview ***************")
for length, count in sorted(pvbah_len_dist.items(), key=lambda x: x[0]):
    print(length, ",", count)
print("\n******************************")
del pvbah_len_dist

print("\n*************** The Distribution of The Depth-level Dwell Time ***************")
for depth_dwell, count in sorted(depth_dwell_dist.items(), key=lambda x: x[0]):
    print(depth_dwell, ",", count)
print("\n******************************")
del depth_dwell_dist



user_freq2 = defaultdict(int)
page_freq2 = defaultdict(int)
valid_pv_num2 = 0
def filter_pageviews_by_minPVnum(pvs):
    FURTHER_COLD_START_THRESHOLD = 0 # fups.COLD_START_THRESHOLD
    filtered_dataset = []
    global valid_pv_num2, user_freq2, page_freq2
    for pv in pvs:
        uid = pv.uid
        url = pv.url
        if ( user_freq[uid] < FURTHER_COLD_START_THRESHOLD or 
             page_freq[url] < FURTHER_COLD_START_THRESHOLD ):
            continue
        user_freq2[uid] += 1
        page_freq2[url] += 1 
        valid_pv_num2 += 1
        filtered_dataset.append(pv)
    return filtered_dataset


all_pageviews = filter_pageviews_by_minPVnum(all_pageviews)

# print()
# print("=============== Statistics of Further Data ================")
# print("valid_pv_num2 =", valid_pv_num2)
# print(len(user_freq2), "unique users and", len(page_freq2), "unique pages")
# print("density =", valid_pv_num2/float(len(user_freq2) * len(page_freq2)))
# print()

# print("(count, freqOfCount)")
# print( Counter(user_freq2.values()).most_common() )
# print( Counter(page_freq2.values()).most_common() )
# 
# print(sorted(Counter(user_freq2.values()).most_common(), key=lambda x:x[0], reverse=False))
# print(sorted(Counter(page_freq2.values()).most_common(), key=lambda x:x[0], reverse=False))



print("\n=============== Separating Training and Test Data ================")
""" Randomly pick training and test instances """
training_set = []
test_set = []
all_training_text = defaultdict(str) # url -> body_text
all_test_text = defaultdict(str)
# users_have_been_in_test = set()
# pages_have_been_in_test = set()
for pv in all_pageviews:
#     if ( user_freq2[uid] < fups.COLD_START_THRESHOLD or 
#         page_freq2[url] < fups.COLD_START_THRESHOLD ):
#         continue
#     body_text = get_body_text(pv.url)
    # if len(test_set) / float(len(training_set) + len(test_set)) < 0.1 and 
    if ( user_freq2[pv.uid] > COLD_START_THRESHOLD and 
         page_freq2[pv.url] > COLD_START_THRESHOLD and
         len(test_set) / len(all_pageviews) <= 0.1 ):
        test_set.append(pv)
#         if body_text != 'unknown':
#             all_test_text[pv.url] = body_text
        user_freq2[pv.uid] -= 1
        page_freq2[pv.url] -= 1
    else:
        training_set.append(pv)
#         if body_text != 'unknown':
#             all_training_text[pv.url] = body_text
            
        
print()
print(len(training_set), "pageviews in the training set")
print(len(test_set), "pageviews in the test set")
print("The fraction is", len(test_set) / float(valid_pv_num2))
print()

del all_pageviews

# porter_stemmer = PorterStemmer()
# 
# """ Build LDA Model (over both training and test pages) """
# clean_tokens = []
# for text in all_training_text.values() + all_test_text.values():
#     tmp_tokens = [porter_stemmer.stem(t) for t in 
#                           ifilter(lambda w : w.isalpha() and len(w) > 1 and
#                                   w not in stopwords.words('english'), 
#                                   word_tokenize(text.lower()))]
#     if not tmp_tokens:
#         ''' Pages whose body text is just "\u4154132" will have [] tmp_tokens '''
#         tmp_tokens = ['unknown']
#     
#     clean_tokens.append(tmp_tokens)
# 
# 
# print("\nBuilding LDA dictionary...")
# dictionary = corpora.Dictionary(clean_tokens)
# print("Converting doc to bow...")
# corpus = [dictionary.doc2bow(d) for d in clean_tokens]
# print("Creating the MM file...")
# corpora.MmCorpus.serialize('lda.mm', corpus)
# mm = corpora.MmCorpus('lda.mm')
# # print("Building a LDA model for", int(sqrt(len(all_training_text))), "topics ...")
# # lda = models.ldamodel.LdaModel(corpus=mm, num_topics=int(sqrt(len(all_training_text))))
# 
# print("Building a LDA model for", 20, "topics ...") # *** If change the number of topics, change the output filename in LibFMInputPreparation as well ***
# lda = models.ldamodel.LdaModel(corpus=mm, num_topics=20)
# print("A LDA model is built.\n")
# 
# doc_lda = lda[dictionary.doc2bow(clean_tokens[0])]
# pprint(doc_lda)
# 
# """
# doc_lda = 
# [(0, 0.81995379945948721),
#  (1, 0.020003070630106111),
#  (2, 0.020007821929791454),
#  (3, 0.020005859522950798),
#  (4, 0.020007262959776777),
#  (5, 0.020002932567543413),
#  (6, 0.020003773871628584),
#  (7, 0.020005253893555897),
#  (8, 0.02000451943754944),
#  (9, 0.020005705727610398)]
# """



""" Building Doc2Vec Model (over both training and test pages) """
# class LabeledLineSentence(object):
#     def __init__(self, tokens):
#         self.tokens = tokens
#     def __iter__(self):
#         for pageid, doc_tokens in enumerate(self.tokens):
#             yield LabeledSentence(doc_tokens, ['PAGE_%s' % pageid])
# 
# model = Doc2Vec(size=100, window=8, min_count=0, workers=4)
# labeled_documents = LabeledLineSentence(clean_tokens)
# model.build_vocab(labeled_documents)
# model.train(labeled_documents)

# print("Doc2Vec: Creating LabelSentences")
# clean_tokens = [LabeledSentence(clean_tokens[i], ['PAGE_%s' % i]) 
#                 for i in range(len(clean_tokens))]  
# print(len(clean_tokens), "articles")  
# doc2vec_model = Doc2Vec(size=20, window=10, min_alpha=0.025, min_count=1, workers=4)
# print("Doc2Vec: Building Vocabulary")
# doc2vec_model.build_vocab(clean_tokens)
# print("Doc2Vec: Training")
# doc2vec_model.train(clean_tokens)
# print("Doc2Vec: Store the model to mmap-able files")
# doc2vec_model.save('../my_model.doc2vec')
# load the model back
# doc2vec_model = Doc2Vec.load('/my_model.doc2vec')



""" Calculate TF-IDF over all pages (over both training and test pages) """
# print("Calculating TF-IDF for", len(all_training_text) + len(all_test_text), "pages")
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_features=500,
#                                    token_pattern=r'(?u)\b[a-zA-Z]+\b')
# tfidf_vectorizer.fit(list(all_training_text.values()) + list(all_test_text.values()))
# 
# print("Finish Training_test_generator module")
# print()


""" Plot the distribution of screen and viewport """    
# screen_width_list = []
# screen_height_list = []
# viewport_width_list = []
# viewport_height_list = []
# # Check the distribution of screen and viewport
# for pv in filtered_pageviews:
#     if pv.screen == 'unknown' or pv.viewport == 'unknown':
#         continue
# #     screen_width, screen_height = [int(pixel) for pixel in pv.screen.split('x')]
# #     screen_width_list.append(screen_width)
# #     screen_height_list.append(screen_height)
#     viewport_width, viewport_height = [int(pixel) for pixel in pv.viewport.split('x')]
#     viewport_width_list.append(viewport_width)
#     viewport_height_list.append(viewport_height)
#      
#  
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.boxplot([viewport_width_list, viewport_height_list])
# # ax.boxplot([screen_width_list, screen_height_list, viewport_width_list, viewport_height_list])
# plt.show()


""" Print the distribution of length, channel, and freshness """
# def print_dict(d):
#     for k, v in d.items():
#         print(str(k)+','+str(v))
#     print()
 
# print("The distribution of LENGTH")
# print_dict(length_dist)
# print("The distribution of CHANNEL")
# print_dict(channel_dist)
# print("The distribution of FRESHNESS")
# print_dict(fresh_dist)


