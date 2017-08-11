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
    def __init__(self, dep, u, p, u_feat, p_feat, c_feat):
        self.user = u
        self.page = p
        self.depth = dep
        self.user_features = u_feat
        self.page_features = p_feat
        self.context_features = c_feat # np.array
#         print('=====================')
#         print(cntxt)
#         print('=====================')
        
    def gen_more_feats(self):
        return self.user_features, self.page_features, self.context_features
    
#     def dense_depth(self):
#         dense_vec = [0] * 100
#         dense_vec[self.depth - 1] = 1
#         return np.array(dense_vec)



unique_feature_names = set()
user2index = {} # include both training and test users
page2index = {} # include both training and test pages

unique_user_feat = set()
unique_page_feat = set()
unique_ctx_feat = set()

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
            
                      
            ''' USER FEATURES '''
            hashed_geo = hashstr('='.join(['geo', 'other'])) if geo in tts.geo_convert2OTHER else hashstr('='.join(['geo', geo]))
            
            USER_FEATURES = ((hashed_geo, 1), )
            unique_user_feat.add(hashed_geo)
            
            
            ''' PAGE FEATURES '''
#             features.append(hashstr('='.join(['channel', channel])))
#             features.append(hashstr('='.join(['section', section])))
            
            chan_sum = sum(channel_group.values())
            hashed_cha_group = tuple((hashstr('='.join(['chans', cha])), channel_group[cha] / chan_sum) for cha in channel_group)
#             hashed_cha_group = tuple((hashstr('='.join(['chans', cha])), 1) for cha in channel_group)
            
            sec_sum = sum(section_group.values())
#             hashed_sec_group = tuple((hashstr('='.join(['secs', sec])), 1) for sec in section_group)
            hashed_sec_group = tuple((hashstr('='.join(['secs', sec])), section_group[sec] / sec_sum) for sec in section_group)

#             if len(channel_group) > 1 and len(section_group) > 1:
#                 print()
            
            hashed_page_meta = ((hashstr('='.join(['length', length])), 1),
                                (hashstr('='.join(['fresh', fresh])), 1), 
                                (hashstr('='.join(['page_type', page_type])), 1),
                                (hashstr('='.join(['tempType', templateType])), 1), 
                                (hashstr('='.join(['blogType', blogType])), 1),
                                (hashstr('='.join(['storyType', storyType])), 1), 
                                (hashstr('='.join(['image', image])), 1),
                                (hashstr('='.join(['forbesStaff', writtenByForbesStaff])), 1),
                                (hashstr('='.join(['commentCount', commentCount])), 1)
                                )
            
            PAGE_FEATURES = hashed_cha_group + hashed_sec_group + hashed_page_meta
            unique_page_feat.update(f for f, v in PAGE_FEATURES)

            
            ''' CONTEXT FEATURES '''
            hashed_device = hashstr('='.join(['device', 'OTHER'])) if device in tts.device_convert2OTHER else hashstr('='.join(['device', device]))
            
            hashed_os = hashstr('='.join(['os', 'OTHER'])) if os in tts.os_convert2OTHER else hashstr('='.join(['os', os]))
            
            hashed_browser = hashstr('='.join(['browser', 'OTHER'])) if browser in tts.browser_convert2OTHER else hashstr('='.join(['browser', browser]))
            
            vp_wid, vp_hei = discretize_pixel_area(viewport)
            vp_wid_counter.update([vp_wid])
            vp_hei_counter.update([vp_hei])
            
            CONTEXT_FEATURES = ((hashed_device, 1), (hashed_os, 1), (hashed_browser, 1), 
                                (hashstr('='.join(['weekday', weekday])), 1),
                                (hashstr('='.join(['hour', hour])), 1),
                                (hashstr('='.join(['vp_wid', vp_wid])), 1),
                                (hashstr('='.join(['vp_hei', vp_hei])), 1)
                                )
            unique_ctx_feat.update(f for f, v in CONTEXT_FEATURES)
            
            
            ALL_FEATURES = np.array(USER_FEATURES + PAGE_FEATURES + CONTEXT_FEATURES)
            
            
            x_inst = X_instance(depth, hashed_uid, hashed_url, np.array(USER_FEATURES), np.array(PAGE_FEATURES), np.array(CONTEXT_FEATURES))
            
            
            pv_X.append(x_inst)
#             pv_y.append(float(dwell))
            pv_y.append([float(dwell)]) # required for 'many to many'; Example: http://stackoverflow.com/questions/38294046/simple-recurrent-neural-network-with-keras
            unique_feature_names.update(f for f, _ in ALL_FEATURES)
        
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
    def __init__(self, unique_user_feat, unique_page_feat, unique_ctx_feat):
        self.feat_dict = {feature : index for index, feature in enumerate(unique_user_feat | unique_page_feat | unique_ctx_feat)}
        self.u_feat_dict = {feature : index for index, feature in enumerate(unique_user_feat)}
        self.p_feat_dict = {feature : index for index, feature in enumerate(unique_page_feat)}
        self.c_feat_dict = {feature : index for index, feature in enumerate(unique_ctx_feat)}
        
    def transform(self, X_batch): # X_batch contains 100 depths in one single page view, len(X_batch) = 100
        X_u = []
        X_p = []
        X_c = []
        for x in X_batch:
            u_vec = [0] * len(self.u_feat_dict)
            for f, v in x.user_features:
                u_vec[self.u_feat_dict[f]] = v
            X_u.append(u_vec)
            
            p_vec = [0] * len(self.p_feat_dict)
            for f, v in x.page_features:
                p_vec[self.p_feat_dict[f]] = v
            X_p.append(p_vec)
            
            c_vec = [0] * len(self.c_feat_dict)
            for f, v in x.context_features:
                c_vec[self.c_feat_dict[f]] = v
            X_c.append(c_vec)
            
        return np.array(X_u, dtype='float32'), np.array(X_p, dtype='float32'), np.array(X_c, dtype='float32')

vectorizer = Vectorizer(unique_user_feat, unique_page_feat, unique_ctx_feat) 

del unique_feature_names

# print(None in unique_feature_names)
# print('' in unique_feature_names)
# print('none' in unique_feature_names)

# print("Fitting feature names")
# vectorizer.fit([{feat_name:1 for feat_name in unique_feature_names}]) # dummy input
# print("The length of each vector will be", len(vectorizer.feature_names_))

# np.set_printoptions(threshold=np.nan)

def Xy_gen(X, y, batch_size=10):
    X_batch_u = []
    X_batch_uf = []
    X_batch_p = []
    X_batch_pf = []
    X_batch_ctx = []
    X_batch_dep = []
    y_batch = []

    for Xinst_pv, y_pv in random.sample(list(zip(X, y)), len(y)): # shuffle pageviews
#     for Xinst_pv, y_pv in list(zip(X, y)):
        ''' Xinst_pv is about one pageview which has 100 X_instance '''
        batch_vecs_uf, batch_vecs_pf, batch_vecs_cf = vectorizer.transform(Xinst_pv)
        X_batch_uf.append(batch_vecs_uf)
        X_batch_pf.append(batch_vecs_pf)
        X_batch_ctx.append(batch_vecs_cf)
#         print('batch_vecs_uf', batch_vecs_uf)
#         print('batch_vecs_pf', batch_vecs_pf)
#         print('batch_vecs_cf', batch_vecs_cf)

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
                  np.array(X_batch_uf, dtype='float32'), \
                  np.array(X_batch_pf, dtype='float32'), \
                  np.array(y_batch, dtype='float32')
            X_batch_ctx.clear()
            X_batch_dep.clear()
            X_batch_u.clear()
            X_batch_p.clear()
            X_batch_uf.clear()
            X_batch_pf.clear()
            y_batch.clear()
            
    if len(y_batch) != 0:
        yield np.array(X_batch_ctx, dtype='float32'), \
              np.array(X_batch_dep, dtype='float32'), \
              np.array(X_batch_u, dtype='float32'), \
              np.array(X_batch_p, dtype='float32'), \
              np.array(X_batch_uf, dtype='float32'), \
              np.array(X_batch_pf, dtype='float32'), \
              np.array(y_batch, dtype='float32')
        X_batch_ctx.clear()
        X_batch_dep.clear()
        X_batch_u.clear()
        X_batch_p.clear()
        X_batch_uf.clear()
        X_batch_pf.clear()
        y_batch.clear()
        
        