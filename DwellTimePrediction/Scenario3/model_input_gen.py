'''
Created on Aug 25, 2016
@author: munichong
'''
import random, numpy as np, hashlib
from collections import defaultdict, Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
import train_test_split as tts
from gensim.models import Doc2Vec


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

def hashstr(s):
#     return hash(s)
    return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16)%(1e+6-1)+1




class X_instance():
    def __init__(self, cntxt, dep, u, p):
        self.user = u
        self.page = p
        self.depth = dep
        self.context = cntxt # dict
        
    def gen_more_feats(self):
        return self.context
    
#     def dense_depth(self):
#         dense_vec = [0] * 100
#         dense_vec[self.depth - 1] = 1
#         return np.array(dense_vec)



unique_feature_names = set()
user2index = {} # include both training and test users
page2index = {} # include both training and test pages



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
        for index, (dwell, uid, clean_url, 
#              top, bottom,
              screen, viewport, geo, agent,
                weekday, hour, length, channel, section, channel_group, section_group, 
                    fresh, device, os, browser, page_type, templateType, blogType, storyType, 
                    image, writtenByForbesStaff, commentCount) in enumerate(pv.depth_level_rows):            
            
            
            '''
            Depth, User, Page indices all start from 1, not 0 !!!
            '''
            #             feature_dict['depth'] = index + 1
            depth = index + 1 # range: [1, 100], must be an integer
            
            hashed_uid = hashstr(uid)
            hashed_url = hashstr(clean_url)
            if hashed_uid not in user2index:
                user2index[hashed_uid] = len(user2index) + 1
            if hashed_url not in page2index:
                page2index[hashed_url] = len(page2index) + 1
            
            
            features = []
            
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
            
                      
            
            if geo in tts.geo_convert2OTHER:
                features.append(hashstr('='.join(['geo', 'other'])))
            else:
                features.append(hashstr('='.join(['geo', geo])))
            
            features.append(hashstr('='.join(['length', length])))
            
#             features.append(hashstr('='.join(['channel', channel])))
#             features.append(hashstr('='.join(['section', section])))
            
            for cha in channel_group:
                features.append(hashstr('='.join(['channel', cha])))
            for sec in section_group:
                features.append(hashstr('='.join(['section', sec])))
            
            features.append(hashstr('='.join(['fresh', fresh])))
            features.append(hashstr('='.join(['page_type', page_type])))
            features.append(hashstr('='.join(['tempType', templateType])))
            features.append(hashstr('='.join(['blogType', blogType])))
            features.append(hashstr('='.join(['storyType', storyType])))
            features.append(hashstr('='.join(['image', image])))
            features.append(hashstr('='.join(['forbesStaff', writtenByForbesStaff])))
            features.append(hashstr('='.join(['commentCount', commentCount])))
            
            
            if device in tts.device_convert2OTHER:
                features.append(hashstr('='.join(['device', device])))
            else:
                features.append(hashstr('='.join(['device', 'OTHER'])))
            if os in tts.os_convert2OTHER:
                features.append(hashstr('='.join(['os', 'other'])))
            else:
                features.append(hashstr('='.join(['os', os])))
            if browser in tts.browser_convert2OTHER:
                features.append(hashstr('='.join(['browser', 'other'])))
            else:
                features.append(hashstr('='.join(['browser', browser])))
            
            
            features.append(hashstr('='.join(['weekday', weekday])))
            features.append(hashstr('='.join(['hour', hour])))
            
            vp_wid, vp_hei = discretize_pixel_area(viewport)
            
            features.append(hashstr('='.join(['vp_wid', vp_wid])))
            features.append(hashstr('='.join(['vp_hei', vp_hei])))
            
            vp_wid_counter.update([vp_wid])
            vp_hei_counter.update([vp_hei])
            
            
            features = tuple(features)
            
            x_inst = X_instance(features, depth, hashed_uid, hashed_url)
            
            
            pv_X.append(x_inst)
#             pv_y.append(float(dwell))
            pv_y.append([float(dwell)]) # required for 'many to many'; Example: http://stackoverflow.com/questions/38294046/simple-recurrent-neural-network-with-keras
            unique_feature_names.update(features)
        
        if pv_X: # if the body is not valid, pv_X will be [] (if "continue")
            X.append(pv_X)
            y.append(pv_y)
        
#     print("%d in %d (%f) pageviews have no valid body content" % 
#         (no_d2v_dep_num/100, pv_index, no_d2v_dep_num/100/pv_index))
    
    return X, y



print("Building training vectors ...")
X_train, y_train = input_vector_builder(tts.training_set) 
del tts.training_set

print("Building validation vectors ...")
X_val, y_val = input_vector_builder(tts.validate_set)  
del tts.validate_set

print("Building test vectors ...")
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


unique_users_num = len(user2index)
unique_pages_num = len(page2index)
print("All three partitions contains %d unique users and %d unique pages" % (unique_users_num, unique_pages_num))



del tts.geo_convert2OTHER
del tts.os_convert2OTHER
del tts.browser_convert2OTHER

# del doc2vec
# del tts.all_training_text


# vectorizer = DictVectorizer(dtype=np.float32) 

class Vectorizer:
    def __init__(self, feature_names):
        self.feat_dict = {feature : index for index, feature in enumerate(feature_names)}
        
    def transform(self, X_batch): # X_batch is a list of depth in one single page view, len(X_batch) = 100
        X = []
        for x in X_batch:
            vec = [0] * len(self.feat_dict)
#             for f, v in x.gen_more_feats():
#                 vec[self.feat_dict[f]] = v
            for f in x.gen_more_feats():
                vec[self.feat_dict[f]] = 1
            
            X.append(vec)
        return np.array(X, dtype='float32')

vectorizer = Vectorizer(unique_feature_names) 

del unique_feature_names

# print(None in unique_feature_names)
# print('' in unique_feature_names)
# print('none' in unique_feature_names)

# print("Fitting feature names")
# vectorizer.fit([{feat_name:1 for feat_name in unique_feature_names}]) # dummy input
# print("The length of each vector will be", len(vectorizer.feature_names_))


def Xy_gen(X, y, batch_size=10):
    X_batch_u = []
    X_batch_p = []
    X_batch_ctx = []
    X_batch_dep = []
    y_batch = []
#     print(len(y))
    for Xinst_pv, y_pv in random.sample(list(zip(X, y)), len(y)): # shuffle pageviews
        # Xinst_pv is about one pageview which has 100 X_instance
#         X_batch_ctx.append( vectorizer.transform(
#                                                     (x.context for x in Xinst_pv)
#                                                       ).toarray()
#                             )
        X_batch_ctx.append( vectorizer.transform(Xinst_pv) )
        user_index = user2index[Xinst_pv[0].user]
        page_index = page2index[Xinst_pv[0].page]
        X_batch_u.append(np.array([user_index] * 100))
        X_batch_p.append(np.array([page_index] * 100))
        X_batch_dep.append( [x.depth for x in Xinst_pv] )
        y_batch.append( y_pv )
        if len(y_batch) == batch_size:
#             print(np.array(X_batch_ctx).shape)
#             print(np.array(X_batch_dep).shape)
#             print(np.array(y_batch).shape)
            yield np.array(X_batch_ctx, dtype='float32'), \
                  np.array(X_batch_dep, dtype='float32'), \
                  np.array(X_batch_u, dtype='float32'), \
                  np.array(X_batch_p, dtype='float32'), \
                  np.array(y_batch, dtype='float32')
            X_batch_ctx.clear()
            X_batch_dep.clear()
            X_batch_u.clear()
            X_batch_p.clear()
            y_batch.clear()
            
    if len(y_batch) != 0:
        yield np.array(X_batch_ctx, dtype='float32'), \
              np.array(X_batch_dep, dtype='float32'), \
              np.array(X_batch_u, dtype='float32'), \
              np.array(X_batch_p, dtype='float32'), \
              np.array(y_batch, dtype='float32')
        X_batch_ctx.clear()
        X_batch_dep.clear()
        X_batch_u.clear()
        X_batch_p.clear()
        y_batch.clear()
        
        