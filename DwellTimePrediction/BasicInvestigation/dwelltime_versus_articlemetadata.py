'''
Created on Jan 4, 2016

@author: Wang
'''
import numpy
from pymongo import MongoClient 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import f_regression
from nltk.tokenize import word_tokenize

client = MongoClient()
featVec_DB = client['Forbes_Dec2015']['FreqUserLogPV']
article_DB = client['Forbes_Dec2015']['ArticleInfo']

def get_articleInfo(userlog_url):
    for doc in article_DB.find({'URL_IN_USERLOG': userlog_url}):
        return doc

def dwelltime_dist_channel():
    n = 0
    channel_stats = {}
    for pv_doc in featVec_DB.find():
        print(n)
        n += 1
        
#         if n ==10000:
#             break
        
        pv_summary = [] # [[screen_top, screen_bottom, dwell_time], [ ... ]]
        skip_pageview = False
        for index, log_dict in enumerate(pv_doc['loglist']):
            additionalinfo = log_dict['additionalinfo']
            
            if log_dict["eventname"] == "Post reading":
                if 'Percentage reading from' not in additionalinfo:
                    skip_pageview = True
                    break
                screen_top = additionalinfo['Percentage reading from']
                screen_bottom = additionalinfo['Percentage of reading']
                dwell_time = additionalinfo['Time on article']
                if dwell_time > 120:
                    skip_pageview = True
                    break
                
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = dwell_time - pv_summary[-1][2]
                if index < len(pv_doc['loglist']) - 1:
                    pv_summary.append([screen_top, screen_bottom, dwell_time])
                    
            elif log_dict["eventname"] == "Text selection":
                continue    
            elif log_dict["eventname"] == "Probably left":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Load new article":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Page leave":
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = additionalinfo['Total time'] - pv_summary[-1][2]
                else:
                    skip_pageview = True
                break
            else:
                continue
        
        if skip_pageview:
            continue
        
        total_dwell_time = 0
        for _, _, dt in pv_summary:
            total_dwell_time += dt
        
        if total_dwell_time > 300 or total_dwell_time < 0:
            continue
        
        userlog_url = pv_doc['url']
        article_metadata = get_articleInfo(userlog_url)
        if not article_metadata or 'displayChannel' not in article_metadata:
            continue
        channel = article_metadata['displayChannel'].lower()
        if channel not in channel_stats:
            channel_stats[channel] = [total_dwell_time]
        else:
            channel_stats[channel].append(total_dwell_time)
    
    print()
    # filter up the channels with insufficient number of page views
    channel_stats = {channel: channel_stats[channel] for channel in channel_stats if len(channel_stats[channel]) > 1000}
    
    for channel, dwell_time_list in channel_stats.items():
        mean = sum(dwell_time_list) / float(len(dwell_time_list))
        print(channel, ',', mean, ',', len(dwell_time_list))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(list(channel_stats.values()))
    ax.set_xticklabels(list(channel_stats.keys()))

    plt.show()

def metadata_vs_dwelltime_ttest():
    n = 0
    features = []
    targets = []
    channel_counts = {}
    for pv_doc in featVec_DB.find():
        print(n)
        n += 1
        
        if n == 20000:
            break
        
        pv_summary = [] # [[screen_top, screen_bottom, dwell_time], [ ... ]]
        skip_pageview = False
        for index, log_dict in enumerate(pv_doc['loglist']):
            additionalinfo = log_dict['additionalinfo']
            
            if log_dict["eventname"] == "Post reading":
                if 'Percentage reading from' not in additionalinfo:
                    skip_pageview = True
                    break
                screen_top = additionalinfo['Percentage reading from']
                screen_bottom = additionalinfo['Percentage of reading']
                dwell_time = additionalinfo['Time on article']
                if dwell_time > 120:
                    skip_pageview = True
                    break
                
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = dwell_time - pv_summary[-1][2]
                if index < len(pv_doc['loglist']) - 1:
                    pv_summary.append([screen_top, screen_bottom, dwell_time])
                    
            elif log_dict["eventname"] == "Text selection":
                continue    
            elif log_dict["eventname"] == "Probably left":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Load new article":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Page leave":
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = additionalinfo['Total time'] - pv_summary[-1][2]
                else:
                    skip_pageview = True
                break
            else:
                continue
        
        if skip_pageview:
            continue
        
        total_dwell_time = 0
        for _, _, dt in pv_summary:
            total_dwell_time += dt
        
        if total_dwell_time > 300 or total_dwell_time < 0:
            continue
        
        userlog_url = pv_doc['url']
        article_metadata = get_articleInfo(userlog_url)
        if not article_metadata or 'displayChannel' not in article_metadata or 'body' not in article_metadata:
            continue
        channel = article_metadata['displayChannel'].lower()
        body_length = len(word_tokenize(article_metadata['body']))
        if body_length < 500:
            continue
        
        if channel in channel_counts and channel_counts[channel] == 300:
            continue
        
        
        features.append({'channel': channel, 'body_length': body_length})
        targets.append(total_dwell_time)   
        if channel in channel_counts:
            channel_counts[channel] += 1
        else:
            channel_counts[channel] = 1
        
    print()
    # filter up the channels with insufficient number of page views
    discarded_channels = [channel for channel in channel_counts if channel_counts[channel] < 200]
    print(len(discarded_channels), 'channels are discarded.')
    
    clean_features = []
    clean_targets = []
    index = 0
    for feats in features:
        if feats['channel'] not in discarded_channels:
            clean_features.append(feats)
            clean_targets.append(targets[index])
        index += 1
    
    vec = DictVectorizer()
    sparse_X = vec.fit_transform(clean_features)
#     print(sparse_X)
#     print(clean_targets)
    print(vec.vocabulary_)
    
#     with open('I:\Desktop\channel_dwell.csv', 'w') as output:
#         index = 0
#         for row in sparse_X.toarray():
#             output.write(str(targets[index]))
#             index += 1
#             for num in row:
#                 output.write('\t')
#                 output.write(str(num.astype(numpy.int64)))
#             output.write('\r\n')
    print(channel_counts)
    F, pval = f_regression(sparse_X.toarray(), numpy.array(clean_targets))
    print(F)
    print(pval)
    print()
    print(list(zip(vec.feature_names_, pval)))
        
if __name__ == "__main__":
#     dwelltime_dist_channel()
    metadata_vs_dwelltime_ttest()