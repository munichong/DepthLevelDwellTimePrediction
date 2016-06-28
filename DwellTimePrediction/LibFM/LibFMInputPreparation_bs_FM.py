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
from sklearn.metrics import mean_squared_error


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
fresh_libfm_output = open('../data_bs/fresh.libfm', 'w')
keyword_libfm_output = open('../data_bs/keyword.libfm', 'w')
topic_libfm_output = open('../data_bs/topic.libfm', 'w')


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
fresh_train_output = open('../data_bs/fresh.train', 'w')
keyword_train_output = open('../data_bs/keyword.train', 'w')
topic_train_output = open('../data_bs/topic.train', 'w')
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
fresh_test_output = open('../data_bs/fresh.test', 'w')
keyword_test_output = open('../data_bs/keyword.test', 'w')
topic_test_output = open('../data_bs/topic.test', 'w')
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
    if key not in lookup:
        lookup[key] = len(lookup)
        output.write('0')
        for i, v in zip(indice, vals):
            output.write(' ' + str(i) + ':' + str(v))
        output.write('\n')

def discretize_pixel_area(pixels):
    if pixels == 'unknown':
        return pixels
    return 'x'.join([str(int(p)/100) for p in pixels.split('x')])

def get_area_text(top, bottom, fulltext):
    length = len(fulltext.split())
    from_top = length * top / 100
    to_bottom = length * bottom / 100 + 1
    return fulltext.split()[from_top : to_bottom]
#     return [ ' '.join(fulltext.split()[from_top : to_bottom]) ]

def get_lda_topic(text):
    topic_dist = ttg.lda[ttg.dictionary.doc2bow(text)]
#     print(topic_dist)
#     print(max(topic_dist, key=operator.itemgetter(1)))
    return max(topic_dist, key=operator.itemgetter(1)) # (topic_index, probability)



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
fresh_index_lookup = defaultdict(int)
keyword_index_lookup = defaultdict(int)
topic_index_lookup = defaultdict(int)

''' For BASELINE '''
dwell_depth_list = defaultdict(list)
dwell_user_depth_list = defaultdict(dict)
dwell_page_depth_list = defaultdict(dict)


# screen_val_dist = defaultdict(int)
# viewport_val_dist = defaultdict(int)



print("FM: Iterating through training data")
for training_pv in ttg.training_set:
    for training_depth in training_pv.depth_level_rows:
        dwell, uid, url, depth, top, bottom, screen, viewport, user_geo, agent, weekday, hour, length, channel, fresh = training_depth
        
        
        ''' transform features if necessary '''
        screen = discretize_pixel_area(screen)
        viewport = discretize_pixel_area(viewport) 
        screen_height = screen.split('x')[0] if screen != 'unknown' else 'unknown'
        screen_width = screen.split('x')[1] if screen != 'unknown' else 'unknown'
        viewport_height = viewport.split('x')[0] if viewport != 'unknown' else 'unknown'
        viewport_width = viewport.split('x')[1] if viewport != 'unknown' else 'unknown'
        
        area_text = get_area_text(top, bottom, ttg.all_body_text[url]) # e.g. ['abc', 'adsf', 'asde']
#         print(ttg.tfidf_vectorizer.transform(area_text))
        topic, _ = get_lda_topic(area_text)
        topic = str(topic)
        
        
        area_text = [' '.join(area_text)] # e.g. ['abc adsf asde']
        keyword_indice = sorted(list( ttg.tfidf_vectorizer.transform(area_text).nonzero()[1] ))
#         print(len(keyword_indice))
        if not keyword_indice:
#             print("NO TF-IDF KEYWORDS ARE FOUND.")
            area_text = get_area_text(top, bottom, ttg.all_body_text[url])
        keyword_values = [1] * len(keyword_indice)
        keyword_key = ' '.join([str(i) for i in keyword_indice])
        
        
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
        write_new_featval(fresh, fresh_index_lookup, fresh_libfm_output)
        write_new_featval_multi(keyword_indice, keyword_values, keyword_key, keyword_index_lookup, keyword_libfm_output)
        write_new_featval(topic, topic_index_lookup, topic_libfm_output)
        
        
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
        fresh_train_output.write(str(fresh_index_lookup[fresh]) + '\n')
        keyword_train_output.write(str(keyword_index_lookup[keyword_key]) + '\n')
        topic_train_output.write(str(topic_index_lookup[topic]) + '\n')
        
    
        y_train_output.write(str(dwell) + '\n')


        ''' For baselines '''
        dwell_depth_list[depth].append(dwell) 
        
        if depth in dwell_user_depth_list[uid]:
            dwell_user_depth_list[uid][depth].append(dwell)
        else:
            dwell_user_depth_list[uid][depth] = [dwell]
            
        if depth in dwell_page_depth_list[url]:
            dwell_page_depth_list[url][depth].append(dwell)
        else:
            dwell_page_depth_list[url][depth] = [dwell]


''' For baselines '''
dwell_globaldepth_mean = defaultdict(float)
dwell_globaldepth_median = defaultdict(float)
for depth in dwell_depth_list:
    dwell_globaldepth_mean[depth] = mean(dwell_depth_list[depth])
    dwell_globaldepth_median[depth] = median(dwell_depth_list[depth])


dwell_userdepth_mean = defaultdict(dict)
dwell_userdepth_median = defaultdict(dict)
for uid in dwell_user_depth_list:
    for depth in dwell_user_depth_list[uid]:
        dwell_userdepth_mean[uid][depth] = mean(dwell_user_depth_list[uid][depth])
        dwell_userdepth_median[uid][depth] = median(dwell_user_depth_list[uid][depth])

dwell_pagedepth_mean = defaultdict(dict)
dwell_pagedepth_median = defaultdict(dict)
for url in dwell_page_depth_list:
    for depth in dwell_page_depth_list[url]:
        dwell_pagedepth_mean[url][depth] = mean(dwell_page_depth_list[url][depth])
        dwell_pagedepth_median[url][depth] = median(dwell_page_depth_list[url][depth])




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
        dwell, uid, url, depth, top, bottom, screen, viewport, user_geo, agent, weekday, hour, length, channel, fresh = test_depth
        
        ''' transform features if necessary '''
        screen = discretize_pixel_area(screen)
        viewport = discretize_pixel_area(viewport) 
        screen_height = screen.split('x')[0] if screen != 'unknown' else 'unknown'
        screen_width = screen.split('x')[1] if screen != 'unknown' else 'unknown'
        viewport_height = viewport.split('x')[0] if viewport != 'unknown' else 'unknown'
        viewport_width = viewport.split('x')[1] if viewport != 'unknown' else 'unknown'
        
        area_text = get_area_text(top, bottom, ttg.all_body_text[url])
        topic, _ = get_lda_topic(area_text)
        topic = str(topic)
        
        
        area_text = [' '.join(area_text)] # e.g. ['abc adsf asde']
        keyword_indice = sorted(list( ttg.tfidf_vectorizer.transform(area_text).nonzero()[1] ))
        keyword_values = [1] * len(keyword_indice)
        keyword_key = ' '.join([str(i) for i in keyword_indice])
    
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
#              body_length not in length_index_lookup
            ):
            continue
        
        ''' Output features to .libfm files '''
        write_new_featval_multi(keyword_indice, keyword_values, keyword_key, keyword_index_lookup, keyword_libfm_output)
        
        
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
        fresh_test_output.write(str(fresh_index_lookup[fresh]) + '\n')
        keyword_test_output.write(str(keyword_index_lookup[keyword_key]) + '\n')
        topic_test_output.write(str(topic_index_lookup[topic]) + '\n')
    
        y_test_output.write(str(dwell) + '\n')
    
        y_pred_globalMEAN.append( dwell_globaldepth_mean[depth] )
        y_pred_globalMEDIAN.append( dwell_globaldepth_median[depth] )
        y_pred_userMEAN.append( dwell_userdepth_mean[uid][depth] )
        y_pred_userMEDIAN.append( dwell_userdepth_median[uid][depth] )
        y_pred_pageMEAN.append( dwell_pagedepth_mean[url][depth] )
        y_pred_pageMEDIAN.append( dwell_pagedepth_median[url][depth] )
        y_true.append( dwell ) 


print("Finish outputting\n")

print('RMSD_gloablMEAN =', sqrt(mean_squared_error(y_true, y_pred_globalMEAN)))
print('RMSD_gloablMEDIAN =', sqrt(mean_squared_error(y_true, y_pred_globalMEDIAN)))
print('RMSD_userMEAN =', sqrt(mean_squared_error(y_true, y_pred_userMEAN)))
print('RMSD_userMEDIAN =', sqrt(mean_squared_error(y_true, y_pred_userMEDIAN)))
print('RMSD_pageMEAN =', sqrt(mean_squared_error(y_true, y_pred_pageMEAN)))
print('RMSD_pageMEDIAN =', sqrt(mean_squared_error(y_true, y_pred_pageMEDIAN)))

# print("SCREEN")
# for s, c in screen_val_dist.items():
#     print(s + ',' + str(c))
#     
# print("VIEWPORT")
# for v, c in viewport_val_dist.items():
#     print(v + ',' + str(c))
    
    

