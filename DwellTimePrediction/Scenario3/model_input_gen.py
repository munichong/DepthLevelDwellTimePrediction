'''
Created on Aug 25, 2016

@author: munichong
'''
import random, numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import train_test_split as tts
from gensim.models import Doc2Vec


def discretize_pixel_area(pixels):
    if pixels == 'unknown':
        return pixels, pixels
    return [str(int(p)//100) for p in pixels.split('x')]

def add_vector_features(feat_dict, name_head, vector):
    for i in range(len(vector)):
        feat_dict[ ''.join([name_head, str(i)]) ] = vector[i]

def clean_url():
    pass



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

def input_vector_builder(pageviews):
    X = [] # reusable; [ [{...}, {...}], [{...}], ... ]
    y = [] # [ [2, 4], [3], ...]
    no_d2v_dep_num = 0
    for pv_index, pv in enumerate(pageviews):
        print(pv_index+1, "/", len(pageviews))
        pv_X = []
        pv_y = []
        for index, (dwell, uid, url, 
#              top, bottom,
              screen, viewport, geo, agent,
                weekday, hour, length, channel, section, channel_group, section_group, 
                    fresh, device, os, browser) in enumerate(pv.depth_level_rows):            
            
            
            feature_dict = defaultdict(int)
            
            
            unique_users.add(uid)
            unique_pages.add(url)
            
            
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
            
            feature_dict[geo] = 1
#             print(geo)
#             if geo is None:
#                 print("@@@")

            feature_dict['='.join(['weekday', weekday])] = 1
            feature_dict['='.join(['hour', hour])] = 1
            feature_dict['='.join(['length', length])] = 1
#             feature_dict['='.join(['channel', channel])] = 1
#             feature_dict['='.join(['section', section])] = 1
            
            for cha in channel_group:
                feature_dict[cha] = 1
            for sec in section_group:
                feature_dict[sec] = 1
            
            feature_dict['='.join(['fresh', fresh])] = 1
            feature_dict[device] = 1
            feature_dict[os] = 1
            feature_dict[browser] = 1
            
            x_inst = X_instance(feature_dict, depth, uid, url)
            
            
            pv_X.append(x_inst)
#             pv_y.append(float(dwell))
            pv_y.append([float(dwell)]) # required for 'many to many'; Example: http://stackoverflow.com/questions/38294046/simple-recurrent-neural-network-with-keras
            unique_feature_names.update(feature_dict.keys())
        
        if pv_X: # if the body is not valid, pv_X will be [] (if "continue")
            X.append(pv_X)
            y.append(pv_y)
        
    print("%d in %d (%f) pageviews have no valid body content" % 
        (no_d2v_dep_num/100, pv_index, no_d2v_dep_num/100/pv_index))
    
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
# print(unique_feature_names)

print()
print("X_train contains %d training examples" % len(X_train))
print("X_val contains %d validation examples" % len(X_val))
print("X_test contains %d test examples" % len(X_test))
print()

unique_users_num = len(unique_users)
unique_pages_num = len(unique_pages)
print("All three partitions contains %d unique users and %d unique pages" % (unique_users_num, unique_pages_num))
del unique_users
del unique_pages


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
        
 
