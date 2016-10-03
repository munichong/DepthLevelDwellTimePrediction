'''
Created on Aug 25, 2016

@author: munichong
'''
import random, numpy as np
from collections import defaultdict, Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
import train_test_split as tts
from gensim.models import Doc2Vec
from urllib.parse import urlparse

def categorize_vp_wid(raw_vp_wid):
    if raw_vp_wid <= 5:
        return '<=5'
    elif raw_vp_wid >= 20:
        return '>=20'
    else:
        return str(raw_vp_wid)

def categorize_vp_hei(raw_vp_hei):
    if raw_vp_hei <= 4:
        return '<=4'
    elif raw_vp_hei >= 11:
        return '>=11'
    else:
        return str(raw_vp_hei) 

def discretize_pixel_area(pixels):
    if pixels == 'unknown':
        return pixels, pixels
    wid, hei = [int(p)//100 for p in pixels.split('x')]
    return categorize_vp_wid(wid), categorize_vp_hei(hei)
   
def add_vector_features(feat_dict, name_head, vector):
    for i in range(len(vector)):
        feat_dict[ ''.join([name_head, str(i)]) ] = vector[i]

def remove_url_parameters(raw_url):
    ''' remove parameters in the raw_url 
        but keep page numbers '''
    parse_result = urlparse(raw_url)
    clean_url = '{0}://{1}{2}'.format(parse_result.scheme, parse_result.netloc, parse_result.path)
#     print("clean_url:", clean_url)
    return clean_url



class X_instance():
    def __init__(self, cntxt, dep, u, p):
        self.user = u
        self.page = p
        self.depth = dep
        self.context = cntxt # dict
    
    def dense_depth(self):
        dense_vec = [0] * 100
        dense_vec[self.depth - 1] = 1
        return np.array(dense_vec)



unique_feature_names = set()
unique_users = set() # include both training and test users
unique_pages = set() # include both training and test pages



# doc2vec = Doc2Vec.load('../doc2vec_models_py3/d2v_model_urlinuserlog_20.doc2vec')

vp_wid_counter = Counter()
vp_hei_counter = Counter()
def input_vector_builder(pageviews):
    X = [] # reusable; [ [{...}, {...}], [{...}], ... ]
    y = [] # [ [2, 4], [3], ...]
    no_d2v_dep_num = 0
    for pv_index, pv in enumerate(pageviews):
#         print(pv_index+1, "/", len(pageviews))
        pv_X = []
        pv_y = []
        for index, (dwell, uid, url, 
#              top, bottom,
              screen, viewport, geo, agent,
                weekday, hour, length, channel, section, channel_group, section_group, 
                    fresh, device, os, browser, page_type, templateType, blogType, storyType, 
                    image, writtenByForbesStaff, commentCount) in enumerate(pv.depth_level_rows):            
            
            
            feature_dict = defaultdict(int)
            
            clean_url = remove_url_parameters(url)
            
            unique_users.add(uid)
            unique_pages.add(clean_url)
            
            
            '''
            Add Doc2Vec vector of this page
            '''
#             try:
#                 d2v_vec = doc2vec.docvecs[url] # the url here is the URL_IN_USERLOG 
# #                 print(d2v_vec)
#                 add_vector_features(feature_dict, 'd2v', d2v_vec)
#             except KeyError:
#                 # the missing will be "0,0,0,0,0"
#                 no_d2v_dep_num += 1
#                 continue
            
            
#             feature_dict['depth'] = index + 1
            depth = index + 1 # range: [1, 100], must be an integer
            
            
            vp_wid, vp_hei = discretize_pixel_area(viewport)
            
            feature_dict['='.join(['vp_wid', vp_wid])] = 1
            feature_dict['='.join(['vp_hei', vp_hei])] = 1
            
            vp_wid_counter.update([vp_wid])
            vp_hei_counter.update([vp_hei])
            
            
            if geo in tts.geo_convert2OTHER:
                feature_dict['='.join(['geo', 'other'])] = 1
            else:
                feature_dict['='.join(['geo', geo])] = 1
            

            feature_dict['='.join(['weekday', weekday])] = 1
            feature_dict['='.join(['hour', hour])] = 1
            feature_dict['='.join(['length', length])] = 1
#             feature_dict['='.join(['channel', channel])] = 1
#             feature_dict['='.join(['section', section])] = 1
            
            for cha in channel_group:
                feature_dict['='.join(['channel', cha])] = 1
            for sec in section_group:
                feature_dict['='.join(['section', sec])] = 1
            
            feature_dict['='.join(['fresh', fresh])] = 1
            feature_dict['='.join(['page_type', page_type])] = 1
            feature_dict['='.join(['tempType', templateType])] = 1
            feature_dict['='.join(['blogType', blogType])] = 1
            feature_dict['='.join(['storyType', storyType])] = 1
            feature_dict['='.join(['image', image])] = 1
            feature_dict['='.join(['forbesStaff', writtenByForbesStaff])] = 1
            feature_dict['='.join(['commentCount', commentCount])] = 1
            
            
            
            feature_dict['='.join(['device', device])] = 1
            if os in tts.os_convert2OTHER:
                feature_dict['='.join(['os', 'other'])] = 1
            else:
                feature_dict['='.join(['os', os])] = 1
            if browser in tts.browser_convert2OTHER:
                feature_dict['='.join(['browser', 'other'])] = 1
            else:
                feature_dict['='.join(['browser', browser])] = 1
            
            x_inst = X_instance(feature_dict, depth, uid, clean_url)
            
            
            pv_X.append(x_inst)
#             pv_y.append(float(dwell))
            pv_y.append([float(dwell)]) # required for 'many to many'; Example: http://stackoverflow.com/questions/38294046/simple-recurrent-neural-network-with-keras
            unique_feature_names.update(feature_dict.keys())
        
        if pv_X: # if the body is not valid, pv_X will be [] (if "continue")
            X.append(pv_X)
            y.append(pv_y)
        
#     print("%d in %d (%f) pageviews have no valid body content" % 
#         (no_d2v_dep_num/100, pv_index, no_d2v_dep_num/100/pv_index))
    
    return X, y

print("Building training input vectors ...")
X_train, y_train = input_vector_builder(tts.training_set) 
del tts.training_set 
X_val, y_val = input_vector_builder(tts.validate_set)  
del tts.validate_set
X_test, y_test = input_vector_builder(tts.test_set) 
del tts.test_set 
# print(np.array(y_train).shape)
print(len(unique_feature_names), "unique feature names") 
print(unique_feature_names)

print()
print("X_train contains %d training examples" % len(X_train))
print("X_val contains %d validation examples" % len(X_val))
print("X_test contains %d test examples" % len(X_test))
print()



print("\n*************** The Distribution of vp_wid ***************")
total = sum(vp_wid_counter.values())
for vp_wid, count in sorted(vp_wid_counter.items(), key=lambda x: x[0]):
    print(vp_wid, "\t", count/100, "\t", count/total)
print("******************************")
del vp_wid_counter

print("\n*************** The Distribution of vp_hei ***************")
total = sum(vp_hei_counter.values())
for vp_hei, count in sorted(vp_hei_counter.items(), key=lambda x: x[0]):
    print(vp_hei, "\t", count/100, "\t", count/total)
print("******************************")


unique_users_num = len(unique_users)
unique_pages_num = len(unique_pages)
print("All three partitions contains %d unique users and %d unique pages" % (unique_users_num, unique_pages_num))
del unique_users
del unique_pages


del tts.geo_convert2OTHER
del tts.os_convert2OTHER
del tts.browser_convert2OTHER

# del doc2vec
# del tts.all_training_text


vectorizer = DictVectorizer(dtype=np.float32) 

# print(None in unique_feature_names)
# print('' in unique_feature_names)
# print('none' in unique_feature_names)

print("Fitting feature names")
vectorizer.fit([{feat_name:1 for feat_name in unique_feature_names}]) # dummy input
print("The length of each vector will be", len(vectorizer.feature_names_))


def Xy_gen(X, y, batch_size=10):
    X_batch_ctx = []
    X_batch_dep = []
    y_batch = []
#     print(len(y))
    for Xinst_pv, y_pv in random.sample(list(zip(X, y)), len(y)): # shuffle pageviews
        # Xinst_pv is about one pageview which has 100 X_instance
        X_batch_ctx.append( vectorizer.transform(
                                                    (x.context for x in Xinst_pv)
                                                      ).toarray()
                            )
        X_batch_dep.append( [x.depth for x in Xinst_pv] )
        y_batch.append( y_pv )
        if len(y_batch) == batch_size:
#             print(np.array(X_batch_ctx).shape)
#             print(np.array(X_batch_dep).shape)
#             print(np.array(y_batch).shape)
            yield np.array(X_batch_ctx, dtype='float32'), \
                  np.array(X_batch_dep, dtype='float32'), \
                  np.array(y_batch, dtype='float32')
            X_batch_ctx.clear()
            X_batch_dep.clear()
            y_batch.clear()
            
    if len(y_batch) != 0:
        yield np.array(X_batch_ctx, dtype='float32'), \
              np.array(X_batch_dep, dtype='float32'), \
              np.array(y_batch, dtype='float32')
        
 
