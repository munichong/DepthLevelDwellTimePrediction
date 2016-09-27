'''
Created on Aug 25, 2016

@author: munichong
'''
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from Scenario1 import train_test_split as tts


def discretize_pixel_area(pixels):
    if pixels == 'unknown':
        return pixels, pixels
    return [str(int(p)//100) for p in pixels.split('x')]

unique_feature_names = set()

def training_input_builder(pageviews):
    X = [] # reusable; [ [{...}, {...}], [{...}], ... ]
    y = [] # [ [2, 4], [3], ...]
    for pv_indx, pv in enumerate(pageviews):
#         print(pv_indx+1, "/", len(pageviews))
        pv_X = []
        pv_y = []
        for (dwell, uid, url, top, bottom, screen, viewport, geo, agent,
                weekday, hour, length, channel, section, channel_group, section_group, 
                    fresh, device, os, browser) in pv.depth_level_rows:            
            
            feature_dict = defaultdict(int)
            feature_dict['='.join(['top', str(top)])], feature_dict['='.join(['bot', str(bottom)])] = 1, 1
            
            vp_wid, vp_hei = discretize_pixel_area(viewport)
            feature_dict['='.join(['vp_wid', vp_wid])], feature_dict['='.join(['vp_hei', vp_hei])] = 1, 1
            
            feature_dict[geo] = 1
#             print(geo)
#             if geo is None:
#                 print("@@@")
    #         feature_dict['='.join(['weekday', weekday])], feature_dict['='.join(['hour', hour])] = 1, 1
            feature_dict['='.join(['length', length])] = 1
    #         feature_dict['='.join(['channel', channel])], feature_dict['='.join(['section', section])] = 1, 1
            
            for cha in channel_group:
                feature_dict[cha] = 1
            for sec in section_group:
                feature_dict[sec] = 1
            
            feature_dict['='.join(['fresh', fresh])] = 1
            feature_dict[device], feature_dict[os], feature_dict[browser] = 1, 1, 1
            
            
            pv_X.append(feature_dict)
#             pv_y.append(dwell)
            pv_y.append([dwell]) # required for 'many to many'; Example: http://stackoverflow.com/questions/38294046/simple-recurrent-neural-network-with-keras
            unique_feature_names.update(feature_dict.keys())
            
        X.append(pv_X)
        y.append(pv_y)
    return X, y       
    
X_train, y_train = training_input_builder(tts.training_set)  
print(len(unique_feature_names), "unique feature names") 

# print(y_train)
# print()
# print(len(X_train[0]))
# print(len(y_train[0]))

del tts.all_training_text


vectorizer = DictVectorizer() 

# print(None in unique_feature_names)
# print('' in unique_feature_names)
# print('none' in unique_feature_names)

vectorizer.fit([{feat_name:1 for feat_name in unique_feature_names}])
print("The length of each vector will be", len(vectorizer.feature_names_))


def Xy_gen(X, y, batch_size=3):
    X_batch = []
    y_batch = []
    for Xdict_pv, y_pv in zip(X, y):
#         print(type(vectorizer.transform(Xdict_pv).toarray()))
        X_batch.append(vectorizer.transform(Xdict_pv).toarray())
        y_batch.append(y_pv)
        if len(X_batch) == batch_size:
#             print(np.array(X_batch).shape)
            yield X_batch, y_batch
            X_batch.clear()
            y_batch.clear()
            
    if len(X_batch) != 0:
        yield X_batch, y_batch
        
 
