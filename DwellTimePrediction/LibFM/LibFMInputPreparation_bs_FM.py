'''
Created on Mar 16, 2016

@author: Wang
'''
import operator
from pymongo import MongoClient
from _collections import defaultdict
from LibFM import Training_test_generator as ttg
from math import sqrt
from numpy import mean, median, array
from sklearn.metrics import mean_squared_error, log_loss
from gensim.models import Doc2Vec, ldamodel
from gensim import corpora
from sklearn.preprocessing import normalize
import scipy as sp




client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']


user_txt_ouput = open('../data_bs/user.libfm', 'w')
page_txt_ouput = open('../data_bs/page.libfm', 'w')
depth_libfm_output = open('../data_bs/depth.libfm', 'w')
screen_libfm_output = open('../data_bs/screen.libfm', 'w')
viewport_libfm_output = open('../data_bs/viewport.libfm', 'w')
geo_libfm_output = open('../data_bs/geo.libfm', 'w')
screen_height_libfm_output = open('../data_bs/screen_height.libfm', 'w')
screen_width_libfm_output = open('../data_bs/screen_width.libfm', 'w')
viewport_height_libfm_output = open('../data_bs/viewport_height.libfm', 'w')
viewport_width_libfm_output = open('../data_bs/viewport_width.libfm', 'w')
weekday_libfm_output = open('../data_bs/weekday.libfm', 'w')
hour_libfm_output = open('../data_bs/hour.libfm', 'w')
length_libfm_output = open('../data_bs/length.libfm', 'w')
channel_libfm_output = open('../data_bs/channel.libfm', 'w')
channelgroup_libfm_output = open('../data_bs/channel_group.libfm', 'w')
fresh_libfm_output = open('../data_bs/fresh.libfm', 'w')
# keyword_libfm_output = open('../data_bs/keyword.libfm', 'w')
# topic10_libfm_output = open('../data_bs/topic_10.libfm', 'w')
# topicgroup10_libfm_output = open('../data_bs/topic_group_10.libfm', 'w')
# topic20_libfm_output = open('../data_bs/topic_20.libfm', 'w')
# topicgroup20_libfm_output = open('../data_bs/topic_group_20.libfm', 'w')
# topic30_libfm_output = open('../data_bs/topic_30.libfm', 'w')
# topicgroup30_libfm_output = open('../data_bs/topic_group_30.libfm', 'w')
# topic40_libfm_output = open('../data_bs/topic_40.libfm', 'w')
# topicgroup40_libfm_output = open('../data_bs/topic_group_40.libfm', 'w')
# d2v50_libfm_output = open('../data_bs/doc2vec_50.libfm', 'w')
# d2v100_libfm_output = open('../data_bs/doc2vec_100.libfm', 'w')
# d2v150_libfm_output = open('../data_bs/doc2vec_150.libfm', 'w')
# d2v200_libfm_output = open('../data_bs/doc2vec_200.libfm', 'w')


user_train_output = open('../data_bs/user.train', 'w')
page_train_output = open('../data_bs/page.train', 'w')
depth_train_output = open('../data_bs/depth.train', 'w')
screen_train_output = open('../data_bs/screen.train', 'w')
viewport_train_output = open('../data_bs/viewport.train', 'w')
geo_train_output = open('../data_bs/geo.train', 'w')
screen_height_train_output = open('../data_bs/screen_height.train', 'w')
screen_width_train_output = open('../data_bs/screen_width.train', 'w')
viewport_height_train_output = open('../data_bs/viewport_height.train', 'w')
viewport_width_train_output = open('../data_bs/viewport_width.train', 'w')
weekday_train_output = open('../data_bs/weekday.train', 'w')
hour_train_output = open('../data_bs/hour.train', 'w')
length_train_output = open('../data_bs/length.train', 'w')
channel_train_output = open('../data_bs/channel.train', 'w')
channelgroup_train_output = open('../data_bs/channel_group.train', 'w')
fresh_train_output = open('../data_bs/fresh.train', 'w')
# keyword_train_output = open('../data_bs/keyword.train', 'w')
# topic10_train_output = open('../data_bs/topic_10.train', 'w')
# topicgroup10_train_output = open('../data_bs/topic_group_10.train', 'w')
# topic20_train_output = open('../data_bs/topic_20.train', 'w')
# topicgroup20_train_output = open('../data_bs/topic_group_20.train', 'w')
# topic30_train_output = open('../data_bs/topic_30.train', 'w')
# topicgroup30_train_output = open('../data_bs/topic_group_30.train', 'w')
# topic40_train_output = open('../data_bs/topic_40.train', 'w')
# topicgroup40_train_output = open('../data_bs/topic_group_40.train', 'w')
# d2v50_train_output = open('../data_bs/doc2vec_50.train', 'w')
# d2v100_train_output = open('../data_bs/doc2vec_100.train', 'w')
# d2v150_train_output = open('../data_bs/doc2vec_150.train', 'w')
# d2v200_train_output = open('../data_bs/doc2vec_200.train', 'w')
y_train_output = open('../data_bs/y.train', 'w')

user_test_output = open('../data_bs/user.test', 'w')
page_test_output = open('../data_bs/page.test', 'w')
depth_test_output = open('../data_bs/depth.test', 'w')
screen_test_output = open('../data_bs/screen.test', 'w')
viewport_test_output = open('../data_bs/viewport.test', 'w')
geo_test_output = open('../data_bs/geo.test', 'w')
screen_height_test_output = open('../data_bs/screen_height.test', 'w')
screen_width_test_output = open('../data_bs/screen_width.test', 'w')
viewport_height_test_output = open('../data_bs/viewport_height.test', 'w')
viewport_width_test_output = open('../data_bs/viewport_width.test', 'w')
weekday_test_output = open('../data_bs/weekday.test', 'w')
hour_test_output = open('../data_bs/hour.test', 'w')
length_test_output = open('../data_bs/length.test', 'w')
channel_test_output = open('../data_bs/channel.test', 'w')
channelgroup_test_output = open('../data_bs/channel_group.test', 'w')
fresh_test_output = open('../data_bs/fresh.test', 'w')
# keyword_test_output = open('../data_bs/keyword.test', 'w')
# topic10_test_output = open('../data_bs/topic_10.test', 'w')
# topicgroup10_test_output = open('../data_bs/topic_group_10.test', 'w')
# topic20_test_output = open('../data_bs/topic_20.test', 'w')
# topicgroup20_test_output = open('../data_bs/topic_group_20.test', 'w')
# topic30_test_output = open('../data_bs/topic_30.test', 'w')
# topicgroup30_test_output = open('../data_bs/topic_group_30.test', 'w')
# topic40_test_output = open('../data_bs/topic_40.test', 'w')
# topicgroup40_test_output = open('../data_bs/topic_group_40.test', 'w')
# d2v50_test_output = open('../data_bs/doc2vec_50.test', 'w')
# d2v100_test_output = open('../data_bs/doc2vec_100.test', 'w')
# d2v150_test_output = open('../data_bs/doc2vec_150.test', 'w')
# d2v200_test_output = open('../data_bs/doc2vec_200.test', 'w')
y_test_output = open('../data_bs/y.test', 'w')



''' write depths to depth.txt '''
for d in range(101):
    depth_libfm_output.write('0 ' + str(d) + ':1\n')
depth_libfm_output.close()  


def write_new_featval(featval, lookup, output):
    if featval not in lookup:
        """ Only if it is a new value, write it to the corresponding file. Otherwise, skip this function """
        lookup[featval] = len(lookup)
        ''' append this new user in user.txt '''
        output.write('0 ' + str(lookup[featval]) + ':1\n')    

def write_new_featval_multi(indice, vals, key, lookup, output): 
    """ if this is a new variable """
    if key not in lookup:
        lookup[key] = len(lookup)
        output.write('0')
        for i, v in zip(indice, vals):
            output.write(' ' + str(i) + ':' + str(v))
        output.write('\n')

def discretize_pixel_area(pixels):
    if pixels == 'unknown':
        return pixels
    return 'x'.join([str(int(p)//100) for p in pixels.split('x')])

def get_area_text(top, bottom, fulltext):
    length = len(fulltext.split())
    from_top = length * top / 100
    to_bottom = length * bottom / 100 + 1
    return fulltext.split()[from_top : to_bottom]
#     return [ ' '.join(fulltext.split()[from_top : to_bottom]) ]

# def get_lda_topic(lda, text):
#     topic_dist = lda[lda_dict.doc2bow(text)]
#     
#     return [str(t[0]) for t in topic_dist], [str(t[1]) for t in topic_dist], max(topic_dist, key=operator.itemgetter(1))

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


''' get index lookup table for each feature '''
user_index_lookup = defaultdict(int)
page_index_lookup = defaultdict(int)
screen_index_lookup = defaultdict(int)
viewport_index_lookup = defaultdict(int)
geo_index_lookup = defaultdict(int)
screen_height_index_lookup = defaultdict(int)
screen_width_index_lookup = defaultdict(int)
viewport_height_index_lookup = defaultdict(int)
viewport_width_index_lookup = defaultdict(int)
weekday_index_lookup = defaultdict(int)
hour_index_lookup = defaultdict(int)
length_index_lookup = defaultdict(int)
channel_index_lookup = defaultdict(int)
channelgroup_index_lookup = defaultdict(int)
fresh_index_lookup = defaultdict(int)
keyword_index_lookup = defaultdict(int)
topic10_index_lookup = defaultdict(int)
topicgroup10_index_lookup = defaultdict(int)
topic20_index_lookup = defaultdict(int)
topicgroup20_index_lookup = defaultdict(int)
topic30_index_lookup = defaultdict(int)
topicgroup30_index_lookup = defaultdict(int)
topic40_index_lookup = defaultdict(int)
topicgroup40_index_lookup = defaultdict(int)
d2v50_index_lookup = defaultdict(int)
d2v100_index_lookup = defaultdict(int)
d2v150_index_lookup = defaultdict(int)
d2v200_index_lookup = defaultdict(int)


''' For BASELINE '''
dwell_depth_dict = defaultdict(list)
dwell_user_depth_dict = defaultdict(lambda:defaultdict(list))
dwell_page_depth_dict = defaultdict(lambda:defaultdict(list))


# screen_val_dist = defaultdict(int)
# viewport_val_dist = defaultdict(int)


# load the model back
# doc2vec_50 = Doc2Vec.load('../doc2vec_models/d2v_model_50.doc2vec')
# doc2vec_100 = Doc2Vec.load('../doc2vec_models/d2v_model_100.doc2vec')
# doc2vec_150 = Doc2Vec.load('../doc2vec_models/d2v_model_150.doc2vec')
# doc2vec_200 = Doc2Vec.load('../doc2vec_models/d2v_model_200.doc2vec')
# lda_10 = ldamodel.LdaModel.load('../lda_models/lda_model_10.lda', mmap='r')
# lda_20 = ldamodel.LdaModel.load('../lda_models/lda_model_20.lda', mmap='r')
# lda_30 = ldamodel.LdaModel.load('../lda_models/lda_model_30.lda', mmap='r')
# lda_40 = ldamodel.LdaModel.load('../lda_models/lda_model_40.lda', mmap='r')
# lda_dict = corpora.dictionary.Dictionary.load('../lda_models/dictionary.dict', mmap='r')


print("FM: Iterating through training data")
for training_pv in ttg.training_set:
    for training_depth in training_pv.depth_level_rows:
        (dwell, uid, url, depth, top, bottom, screen, viewport, user_geo, agent,
                                weekday, hour, length, channel, channel_group, fresh) = training_depth
        
        
        ''' transform features if necessary '''
        screen = discretize_pixel_area(screen)
#         print("BEFORE:", viewport)
        viewport = discretize_pixel_area(viewport) 
#         print("AFTER:", viewport)
        screen_height = screen.split('x')[0] if screen != 'unknown' else 'unknown'
        screen_width = screen.split('x')[1] if screen != 'unknown' else 'unknown'
        viewport_height = viewport.split('x')[0] if viewport != 'unknown' else 'unknown'
        viewport_width = viewport.split('x')[1] if viewport != 'unknown' else 'unknown'
        
#         if url in ttg.all_training_text:
#             area_text = get_area_text(top, bottom, ttg.all_training_text[url]) # e.g. ['abc', 'adsf', 'asde']
#             topic_indice_10, topic_values_10, (topic_index_10, _) = get_lda_topic(lda_10, area_text)
#             topic_10 = str(topic_index_10)
#             topicgroup_key_10 = ' '.join(topic_indice_10 + topic_values_10)
#             topic_indice_20, topic_values_20, (topic_index_20, _) = get_lda_topic(lda_20, area_text)
#             topic_20 = str(topic_index_20)
#             topicgroup_key_20 = ' '.join(topic_indice_20 + topic_values_20)
#             topic_indice_30, topic_values_30, (topic_index_30, _) = get_lda_topic(lda_30, area_text)
#             topic_30 = str(topic_index_30)
#             topicgroup_key_30 = ' '.join(topic_indice_30 + topic_values_30)
#             topic_indice_40, topic_values_40, (topic_index_40, _) = get_lda_topic(lda_40, area_text)
#             topic_40 = str(topic_index_40)
#             topicgroup_key_40 = ' '.join(topic_indice_40 + topic_values_40)
        
        
#             d2v_vec_50 = doc2vec_50.infer_vector(area_text)
#             d2v_key_50 = ' '.join(str(v) for v in d2v_vec_50)
#             d2v_vec_100 = doc2vec_100.infer_vector(area_text)
#             d2v_key_100 = ' '.join(str(v) for v in d2v_vec_100)
#             d2v_vec_150 = doc2vec_150.infer_vector(area_text)
#             d2v_key_150 = ' '.join(str(v) for v in d2v_vec_150)
#             d2v_vec_200 = doc2vec_200.infer_vector(area_text)
#             d2v_key_200 = ' '.join(str(v) for v in d2v_vec_200)
        
#             area_text = [' '.join(area_text)] # e.g. ['abc adsf asde']
#             keyword_indice = sorted(list( ttg.tfidf_vectorizer.transform(area_text).nonzero()[1] ))
#             if not keyword_indice:
# #                 print("NO TF-IDF KEYWORDS ARE FOUND.")
#                 area_text = get_area_text(top, bottom, ttg.all_training_text[url])
#             keyword_values = [1] * len(keyword_indice)
#             keyword_key = ' '.join([str(i) for i in keyword_indice])
        
        
        channelgroup_values = [1.0/len(channel_group)] * len(channel_group)
        channel_group_indice = [c.split('_')[-1] for c in channel_group]
#         channelgroup_values = [1] * len(channel_group)
        channelgroup_key = ' '.join(channel_group +
                                    [str(v) for v in channelgroup_values])
        
#         screen_val_dist[screen] += 1
#         viewport_val_dist[viewport] += 1
    
        ''' Output features to .libfm files '''
        write_new_featval(uid, user_index_lookup, user_txt_ouput)
        write_new_featval(url, page_index_lookup, page_txt_ouput)
        write_new_featval(screen, screen_index_lookup, screen_libfm_output)
        write_new_featval(viewport, viewport_index_lookup, viewport_libfm_output)
        write_new_featval(user_geo, geo_index_lookup, geo_libfm_output)
        write_new_featval(screen_height, screen_height_index_lookup, screen_height_libfm_output)
        write_new_featval(screen_width, screen_width_index_lookup, screen_width_libfm_output)
        write_new_featval(viewport_height, viewport_height_index_lookup, viewport_height_libfm_output)
        write_new_featval(viewport_width, viewport_width_index_lookup, viewport_width_libfm_output)
        write_new_featval(weekday, weekday_index_lookup, weekday_libfm_output)
        write_new_featval(hour, hour_index_lookup, hour_libfm_output)
        write_new_featval(length, length_index_lookup, length_libfm_output)
        write_new_featval(channel, channel_index_lookup, channel_libfm_output)
        write_new_featval_multi(channel_group_indice, channelgroup_values, channelgroup_key,
                                channelgroup_index_lookup, channelgroup_libfm_output)
        write_new_featval(fresh, fresh_index_lookup, fresh_libfm_output)
        
#         if url in ttg.all_training_text:
#             write_new_featval_multi(keyword_indice, keyword_values, keyword_key,
#                                 keyword_index_lookup, keyword_libfm_output)
#             write_new_featval(topic_10, topic10_index_lookup, topic10_libfm_output)
#             write_new_featval_multi(topic_indice_10, topic_values_10, topicgroup_key_10,
#                                 topicgroup10_index_lookup, topicgroup10_libfm_output)
#             write_new_featval(topic_20, topic20_index_lookup, topic20_libfm_output)
#             write_new_featval_multi(topic_indice_20, topic_values_20, topicgroup_key_20,
#                                 topicgroup20_index_lookup, topicgroup20_libfm_output)
#             write_new_featval(topic_30, topic30_index_lookup, topic30_libfm_output)
#             write_new_featval_multi(topic_indice_30, topic_values_30, topicgroup_key_30,
#                                 topicgroup30_index_lookup, topicgroup30_libfm_output)
#             write_new_featval(topic_40, topic40_index_lookup, topic40_libfm_output)
#             write_new_featval_multi(topic_indice_40, topic_values_40, topicgroup_key_40,
#                                 topicgroup40_index_lookup, topicgroup40_libfm_output)
#             write_new_featval_multi(range(1, len(d2v_vec_50)+1), d2v_vec_50, d2v_key_50, 
#                                 d2v50_index_lookup, d2v50_libfm_output)
#             write_new_featval_multi(range(1, len(d2v_vec_100)+1), d2v_vec_100, d2v_key_100, 
#                                 d2v100_index_lookup, d2v100_libfm_output)
#             write_new_featval_multi(range(1, len(d2v_vec_150)+1), d2v_vec_150, d2v_key_150, 
#                                 d2v150_index_lookup, d2v150_libfm_output)
#             write_new_featval_multi(range(1, len(d2v_vec_200)+1), d2v_vec_200, d2v_key_200, 
#                                 d2v200_index_lookup, d2v200_libfm_output)
        
        
        ''' Output to training files '''
        user_train_output.write(str(user_index_lookup[uid]) + '\n')
        page_train_output.write(str(page_index_lookup[url]) + '\n')
        depth_train_output.write(str(depth) + '\n')
        screen_train_output.write(str(screen_index_lookup[screen]) + '\n')
        viewport_train_output.write(str(viewport_index_lookup[viewport]) + '\n')
        geo_train_output.write(str(geo_index_lookup[user_geo]) + '\n')
        screen_height_train_output.write(str(screen_height_index_lookup[screen_height]) + '\n')
        screen_width_train_output.write(str(screen_width_index_lookup[screen_width]) + '\n')
        viewport_height_train_output.write(str(viewport_height_index_lookup[viewport_height]) + '\n')
        viewport_width_train_output.write(str(viewport_width_index_lookup[viewport_width]) + '\n')
        weekday_train_output.write(str(weekday_index_lookup[weekday]) + '\n')
        hour_train_output.write(str(hour_index_lookup[hour]) + '\n')
        length_train_output.write(str(length_index_lookup[length]) + '\n')
        channel_train_output.write(str(channel_index_lookup[channel]) + '\n')
        channelgroup_train_output.write(str(channelgroup_index_lookup[channelgroup_key]) + '\n')
        fresh_train_output.write(str(fresh_index_lookup[fresh]) + '\n')
        
#         if url in ttg.all_training_text:
#             keyword_train_output.write(str(keyword_index_lookup[keyword_key]) + '\n')
#             topic10_train_output.write(str(topic10_index_lookup[topic_10]) + '\n')
#             topicgroup10_train_output.write(str(topicgroup10_index_lookup[topicgroup_key_10]) + '\n')
#             topic20_train_output.write(str(topic20_index_lookup[topic_20]) + '\n')
#             topicgroup20_train_output.write(str(topicgroup20_index_lookup[topicgroup_key_20]) + '\n')
#             topic30_train_output.write(str(topic30_index_lookup[topic_30]) + '\n')
#             topicgroup30_train_output.write(str(topicgroup30_index_lookup[topicgroup_key_30]) + '\n')
#             topic40_train_output.write(str(topic40_index_lookup[topic_40]) + '\n')
#             topicgroup40_train_output.write(str(topicgroup40_index_lookup[topicgroup_key_40]) + '\n')
#             d2v50_train_output.write(str(d2v50_index_lookup[d2v_key_50]) + '\n')
#             d2v100_train_output.write(str(d2v100_index_lookup[d2v_key_100]) + '\n')
#             d2v150_train_output.write(str(d2v150_index_lookup[d2v_key_150]) + '\n')
#             d2v200_train_output.write(str(d2v200_index_lookup[d2v_key_200]) + '\n')


        ''' dwell time prediction '''
#         y_train_output.write(str(dwell) + '\n')
#         ''' For baselines '''
#         dwell_depth_list[depth].append(dwell) 
        
#         if depth in dwell_user_depth_list[uid]:
#             dwell_user_depth_list[uid][depth].append(dwell)
#         else:
#             dwell_user_depth_list[uid][depth] = [dwell]
#             
#         if depth in dwell_page_depth_list[url]:
#             dwell_page_depth_list[url][depth].append(dwell)
#         else:
#             dwell_page_depth_list[url][depth] = [dwell]
    
        ''' viewability prediction '''  
        if dwell >= 10:
            y_train_output.write('1\n')
            
            ''' For baselines '''
            dwell_depth_dict[depth].append(1) 
            dwell_user_depth_dict[uid][depth].append(1)
            dwell_page_depth_dict[url][depth].append(1)
            
        else:
            y_train_output.write('0\n')
            
            ''' For baselines '''
            dwell_depth_dict[depth].append(0) 
            dwell_user_depth_dict[uid][depth].append(0)
            dwell_page_depth_dict[url][depth].append(0)
            


# print(dwell_depth_dict[30])
# print(list(dwell_user_depth_dict.items())[0])
# print(list(dwell_page_depth_dict.items())[0])

'''
5s
('LogLoss_gloablMEAN =', 0.65038167934570745)
('LogLoss_userMEAN =', 0.98877404208284758)
('LogLoss_pageMEAN =', 0.93869638785483234)

7s:
('LogLoss_gloablMEAN =', 0.63023544987234292)
('LogLoss_userMEAN =', 0.97725477062906463)
('LogLoss_pageMEAN =', 0.76804345977914845)

10s:
('LogLoss_gloablMEAN =', 0.56063854990284279)
('LogLoss_userMEAN =', 1.4259755316961391)
('LogLoss_pageMEAN =', 0.68864661902342794)
'''

''' For baselines '''
dwell_globaldepth_mean = defaultdict(float)
dwell_globaldepth_median = defaultdict(float)
for depth in dwell_depth_dict:
    dwell_globaldepth_mean[depth] = mean(dwell_depth_dict[depth])
    dwell_globaldepth_median[depth] = median(dwell_depth_dict[depth])


dwell_userdepth_mean = defaultdict(dict)
dwell_userdepth_median = defaultdict(dict)
for uid in dwell_user_depth_dict:
    for depth in dwell_user_depth_dict[uid]:
        dwell_userdepth_mean[uid][depth] = mean(dwell_user_depth_dict[uid][depth])
        dwell_userdepth_median[uid][depth] = median(dwell_user_depth_dict[uid][depth])

dwell_pagedepth_mean = defaultdict(dict)
dwell_pagedepth_median = defaultdict(dict)
for url in dwell_page_depth_dict:
    for depth in dwell_page_depth_dict[url]:
        dwell_pagedepth_mean[url][depth] = mean(dwell_page_depth_dict[url][depth])
        dwell_pagedepth_median[url][depth] = median(dwell_page_depth_dict[url][depth])


# print(dwell_globaldepth_mean)
# print(dwell_userdepth_mean)
# print(dwell_pagedepth_mean)



y_pred_globalMEAN = []
y_pred_globalMEDIAN = []
y_pred_userMEAN = []
y_pred_userMEDIAN = []
y_pred_pageMEAN = []
y_pred_pageMEDIAN = []
y_true = []    

print("FM: Iterating through test data")
for test_pv in ttg.test_set:
    for test_depth in test_pv.depth_level_rows:
        (dwell, uid, url, depth, top, bottom, screen, viewport, user_geo, agent, 
                                     weekday, hour, length, channel, channel_group, fresh) = test_depth
        
        ''' transform features if necessary '''
        screen = discretize_pixel_area(screen)
        viewport = discretize_pixel_area(viewport) 
        screen_height = screen.split('x')[0] if screen != 'unknown' else 'unknown'
        screen_width = screen.split('x')[1] if screen != 'unknown' else 'unknown'
        viewport_height = viewport.split('x')[0] if viewport != 'unknown' else 'unknown'
        viewport_width = viewport.split('x')[1] if viewport != 'unknown' else 'unknown'
        
        
#         if url in ttg.all_test_text:
#             area_text = get_area_text(top, bottom, ttg.all_test_text[url])
#             topic_indice_10, topic_values_10, (topic_index_10, _) = get_lda_topic(lda_10, area_text)
#             topic_10 = str(topic_index_10)
#             topicgroup_key_10 = ' '.join(topic_indice_10 + topic_values_10)
#             topic_indice_20, topic_values_20, (topic_index_20, _) = get_lda_topic(lda_20, area_text)
#             topic_20 = str(topic_index_20)
#             topicgroup_key_20 = ' '.join(topic_indice_20 + topic_values_20)
#             topic_indice_30, topic_values_30, (topic_index_30, _) = get_lda_topic(lda_30, area_text)
#             topic_30 = str(topic_index_30)
#             topicgroup_key_30 = ' '.join(topic_indice_30 + topic_values_30)
#             topic_indice_40, topic_values_40, (topic_index_40, _) = get_lda_topic(lda_40, area_text)
#             topic_40 = str(topic_index_40)
#             topicgroup_key_40 = ' '.join(topic_indice_40 + topic_values_40)
        
        
#             d2v_vec_50 = doc2vec_50.infer_vector(area_text)
#             d2v_key_50 = ' '.join(str(v) for v in d2v_vec_50)
#             d2v_vec_100 = doc2vec_100.infer_vector(area_text)
#             d2v_key_100 = ' '.join(str(v) for v in d2v_vec_100)
#             d2v_vec_150 = doc2vec_150.infer_vector(area_text)
#             d2v_key_150 = ' '.join(str(v) for v in d2v_vec_150)
#             d2v_vec_200 = doc2vec_200.infer_vector(area_text)
#             d2v_key_200 = ' '.join(str(v) for v in d2v_vec_200)
        
        
#             area_text = [' '.join(area_text)] # e.g. ['abc adsf asde']
#             keyword_indice = sorted(list( ttg.tfidf_vectorizer.transform(area_text).nonzero()[1] ))
#             keyword_values = [1] * len(keyword_indice)
#             keyword_key = ' '.join([str(i) for i in keyword_indice])
        
        channelgroup_values = [1.0/len(channel_group)] * len(channel_group)
#         channel_group_indice = [c.split('_')[-1] for c in channel_group]
        channelgroup_key = ' '.join(channel_group +
                                    [str(v) for v in channelgroup_values])
        
    
        if ( uid not in user_index_lookup or 
             url not in page_index_lookup or 
             screen not in screen_index_lookup or 
             viewport not in viewport_index_lookup or 
             user_geo not in geo_index_lookup or 
             weekday not in weekday_index_lookup or 
             hour not in hour_index_lookup or 
             screen_height not in screen_height_index_lookup or
             screen_width not in screen_width_index_lookup or
             viewport_height not in viewport_height_index_lookup or
             viewport_width not in viewport_width_index_lookup or 
             channel not in channel_index_lookup or 
             length not in length_index_lookup or 
             fresh not in fresh_index_lookup 
#              topic_10 not in topic_index_lookup
            ):
            continue
        
        ''' Output features to .libfm files '''
#         write_new_featval_multi(keyword_indice, keyword_values, keyword_key,
#                                 keyword_index_lookup, keyword_libfm_output) # write new keywords combination into libfm file
#         write_new_featval_multi(channel_group, channelgroup_values, channelgroup_key,
#                                 channelgroup_index_lookup, channelgroup_libfm_output)
#         write_new_featval_multi(topic_indice_10, topic_values_10, topicgroup_key_10,
#                                 topicgroup10_index_lookup, topicgroup10_libfm_output)
#         write_new_featval_multi(topic_indice_20, topic_values_20, topicgroup_key_20,
#                                 topicgroup20_index_lookup, topicgroup20_libfm_output)
#         write_new_featval_multi(topic_indice_30, topic_values_30, topicgroup_key_30,
#                                 topicgroup30_index_lookup, topicgroup30_libfm_output)
#         write_new_featval_multi(topic_indice_40, topic_values_40, topicgroup_key_40,
#                                 topicgroup40_index_lookup, topicgroup40_libfm_output)
#         write_new_featval_multi(range(1, len(d2v_vec_50)+1), d2v_vec_50, d2v_key_50, 
#                                 d2v50_index_lookup, d2v50_libfm_output)
#         write_new_featval_multi(range(1, len(d2v_vec_100)+1), d2v_vec_100, d2v_key_100, 
#                                 d2v100_index_lookup, d2v100_libfm_output)
#         write_new_featval_multi(range(1, len(d2v_vec_150)+1), d2v_vec_150, d2v_key_150, 
#                                 d2v150_index_lookup, d2v150_libfm_output)
#         write_new_featval_multi(range(1, len(d2v_vec_200)+1), d2v_vec_200, d2v_key_200, 
#                                 d2v200_index_lookup, d2v200_libfm_output)
        
        
        user_test_output.write(str(user_index_lookup[uid]) + '\n')
        page_test_output.write(str(page_index_lookup[url]) + '\n')
        depth_test_output.write(str(depth) + '\n')
        screen_test_output.write(str(screen_index_lookup[screen]) + '\n')
        viewport_test_output.write(str(viewport_index_lookup[viewport]) + '\n')
        geo_test_output.write(str(geo_index_lookup[user_geo]) + '\n')
        screen_height_test_output.write(str(screen_height_index_lookup[screen_height]) + '\n')
        screen_width_test_output.write(str(screen_width_index_lookup[screen_width]) + '\n')
        viewport_height_test_output.write(str(viewport_height_index_lookup[viewport_height]) + '\n')
        viewport_width_test_output.write(str(viewport_width_index_lookup[viewport_width]) + '\n')
        weekday_test_output.write(str(weekday_index_lookup[weekday]) + '\n')
        hour_test_output.write(str(hour_index_lookup[hour]) + '\n')
        length_test_output.write(str(length_index_lookup[length]) + '\n')
        channel_test_output.write(str(channel_index_lookup[channel]) + '\n')
        channelgroup_test_output.write(str(channelgroup_index_lookup[channelgroup_key]) + '\n')
        fresh_test_output.write(str(fresh_index_lookup[fresh]) + '\n')
        
#         if url in ttg.all_test_text:
#             keyword_test_output.write(str(keyword_index_lookup[keyword_key]) + '\n')
#             topic10_test_output.write(str(topic10_index_lookup[topic_10]) + '\n')
#             topicgroup10_test_output.write(str(topicgroup10_index_lookup[topicgroup_key_10]) + '\n')
#             topic20_test_output.write(str(topic20_index_lookup[topic_20]) + '\n')
#             topicgroup20_test_output.write(str(topicgroup20_index_lookup[topicgroup_key_20]) + '\n')
#             topic30_test_output.write(str(topic30_index_lookup[topic_30]) + '\n')
#             topicgroup30_test_output.write(str(topicgroup30_index_lookup[topicgroup_key_30]) + '\n')
#             topic40_test_output.write(str(topic40_index_lookup[topic_40]) + '\n')
#             topicgroup40_test_output.write(str(topicgroup40_index_lookup[topicgroup_key_40]) + '\n')
#             d2v50_test_output.write(str(d2v50_index_lookup[d2v_key_50]) + '\n')
#             d2v100_test_output.write(str(d2v100_index_lookup[d2v_key_100]) + '\n')
#             d2v150_test_output.write(str(d2v150_index_lookup[d2v_key_150]) + '\n')
#             d2v200_test_output.write(str(d2v200_index_lookup[d2v_key_200]) + '\n')
    
        ''' dwell time prediction  '''
#         y_test_output.write(str(dwell) + '\n')
#         y_true.append( dwell ) 
        
        ''' viewability prediction '''  
        if dwell >= 10:
            y_test_output.write('1\n')
            y_true.append( 1 ) 
        else:
            y_test_output.write('0\n')
            y_true.append( 0 ) 
        
    
        y_pred_globalMEAN.append( dwell_globaldepth_mean[depth] )
        y_pred_globalMEDIAN.append( dwell_globaldepth_median[depth] )
        y_pred_userMEAN.append( dwell_userdepth_mean[uid][depth] )
        y_pred_userMEDIAN.append( dwell_userdepth_median[uid][depth] )
        y_pred_pageMEAN.append( dwell_pagedepth_mean[url][depth] )
        y_pred_pageMEDIAN.append( dwell_pagedepth_median[url][depth] )

        


print("Finish outputting\n")

# print('RMSD_gloablMEAN =', sqrt(mean_squared_error(y_true, y_pred_globalMEAN)))
# print('RMSD_gloablMEDIAN =', sqrt(mean_squared_error(y_true, y_pred_globalMEDIAN)))
# print('RMSD_userMEAN =', sqrt(mean_squared_error(y_true, y_pred_userMEAN)))
# print('RMSD_userMEDIAN =', sqrt(mean_squared_error(y_true, y_pred_userMEDIAN)))
# print('RMSD_pageMEAN =', sqrt(mean_squared_error(y_true, y_pred_pageMEAN)))
# print('RMSD_pageMEDIAN =', sqrt(mean_squared_error(y_true, y_pred_pageMEDIAN)))

# print(y_pred_globalMEAN)
# print(y_pred_userMEAN)
# print(y_pred_pageMEAN)
# 
# print('LogLoss_gloablMEAN =', logloss(y_true, y_pred_globalMEAN))
# print('LogLoss_userMEAN =', logloss(y_true, y_pred_userMEAN))
# print('LogLoss_pageMEAN =', logloss(y_true, y_pred_pageMEAN))

# print("SCREEN")
# for s, c in screen_val_dist.items():
#     print(s + ',' + str(c))
#     
# print("VIEWPORT")
# for v, c in viewport_val_dist.items():
#     print(v + ',' + str(c))
    
    

