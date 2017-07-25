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



class X_pageview():
    def __init__(self):
        ''' PAGE-LEVEL FEATURES '''
        self.user_index = None
        self.page_index = None
        self.user_features = None # np.array
        self.page_features = None
        self.context_features = None
        self.depth_truth = None

    def aux_feats(self):
#         print(self.user_features.shape)
#         print(self.page_features.shape)
#         print(self.context_features.shape)
        return np.concatenate((self.user_features, self.page_features, self.context_features), axis=0)

    def depths(self):
        return [d for d in range(1, 101)] # depth index stars from 1

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
    '''
    "pageviews" contains a list of Pageview instances.
    '''
    pv_examples = []

    for pv_index, pv in enumerate(pageviews):
#         print(pv_index+1, "/", len(pageviews))

        hashed_uid = hashstr(pv.uid)
        if hashed_uid not in user2index:
            user2index[hashed_uid] = len(user2index) + 1

        hashed_url = hashstr(pv.url)
        if hashed_url not in page2index:
            page2index[hashed_url] = len(page2index) + 1


        X_pv = X_pageview()
        X_pv.user_index = user2index[hashed_uid]
        X_pv.page_index = page2index[hashed_url]

        ''' USER FEATURES '''
        geo = pv.pagelevel_auxfeats['geo']
        hashed_geo = hashstr('='.join(['geo', 'other'])) if geo in tts.geo_convert2OTHER else hashstr('='.join(['geo', geo]))
        X_pv.user_features = np.array([(hashed_geo, 1), ], dtype='float32')


        ''' PAGE FEATURES '''
#         displayChannel = [(hashstr('='.join(['disChan', pv.pagelevel_auxfeats['channel']])), 1), ]
#         displaySection = [(hashstr('='.join(['disSec', pv.pagelevel_auxfeats['section']])), 1), ]
        
        channel_group = pv.pagelevel_auxfeats['channel_group']
        chan_sum = sum(channel_group.values())
        hashed_cha_group = [(hashstr('='.join(['chans', cha])), channel_group[cha] / chan_sum) for cha in channel_group]
#         hashed_cha_group = [(hashstr('='.join(['chans', cha])), 1) for cha in channel_group]

        section_group = pv.pagelevel_auxfeats['section_group']
        sec_sum = sum(section_group.values())
#         hashed_sec_group = [(hashstr('='.join(['secs', sec])), 1) for sec in section_group]
        hashed_sec_group = [(hashstr('='.join(['secs', sec])), section_group[sec] / sec_sum) for sec in section_group]


        hashed_page_meta = [
                            (hashstr('='.join(['disChan', pv.pagelevel_auxfeats['channel']])), 1),
                            (hashstr('='.join(['disSec', pv.pagelevel_auxfeats['section']])), 1),
                            (hashstr('='.join(['len', pv.pagelevel_auxfeats['length']])), 1),
                            (hashstr('='.join(['frsh', pv.pagelevel_auxfeats['fresh']])), 1),
                            (hashstr('='.join(['page_t', pv.pagelevel_auxfeats['page_type']])), 1),
                            (hashstr('='.join(['temp', pv.pagelevel_auxfeats['templateType']])), 1),
                            (hashstr('='.join(['blog', pv.pagelevel_auxfeats['blogType']])), 1),
                            (hashstr('='.join(['story', pv.pagelevel_auxfeats['storyType']])), 1),
                            (hashstr('='.join(['image', pv.pagelevel_auxfeats['image']])), 1),
                            (hashstr('='.join(['staff', pv.pagelevel_auxfeats['writtenByForbesStaff']])), 1),
                            (hashstr('='.join(['comment', pv.pagelevel_auxfeats['calledOutCommentCount']])), 1)
                            ]

        X_pv.page_features = np.array(hashed_cha_group + hashed_sec_group + hashed_page_meta, dtype='float32')


        ''' CONTEXT FEATURES '''
        device, os, browser = pv.pagelevel_auxfeats['device'], pv.pagelevel_auxfeats['os'], pv.pagelevel_auxfeats['browser']
        hashed_device = hashstr('='.join(['device', 'OTHER'])) if device in tts.device_convert2OTHER else hashstr('='.join(['device', device]))
        hashed_os = hashstr('='.join(['os', 'OTHER'])) if os in tts.os_convert2OTHER else hashstr('='.join(['os', os]))
        hashed_browser = hashstr('='.join(['browser', 'OTHER'])) if browser in tts.browser_convert2OTHER else hashstr('='.join(['browser', browser]))

        vp_wid, vp_hei = discretize_pixel_area(pv.pagelevel_auxfeats['viewport'])
        vp_wid_counter.update([vp_wid])
        vp_hei_counter.update([vp_hei])

        X_pv.context_features = np.array([(hashed_device, 1), (hashed_os, 1), (hashed_browser, 1),
                                    (hashstr('='.join(['vp_wid', vp_wid])), 1),
                                    (hashstr('='.join(['vp_hei', vp_hei])), 1),
                                    (hashstr('='.join(['weekday', pv.pagelevel_auxfeats['weekday']])), 1),
                                    (hashstr('='.join(['hour', pv.pagelevel_auxfeats['hour']])), 1)
                                    ], dtype='float32')


        ''' TARGET VARIABLE '''
        X_pv.depth_truth = np.array([[float(target)] for target in pv.depth_truth], dtype='float32') # required for 'many to many'; Example: http://stackoverflow.com/questions/38294046/simple-recurrent-neural-network-with-keras


        unique_feature_names.update(f for f, _ in X_pv.aux_feats())


        pv_examples.append(X_pv)

    return pv_examples



print("Building training vectors ...")
training_examples = input_vector_builder(tts.training_set)
del tts.training_set

print("Building validation vectors ...")
validation_examples = input_vector_builder(tts.validate_set)
del tts.validate_set

print("Building test vectors ...")
test_examples = input_vector_builder(tts.test_set)
del tts.test_set

print(len(unique_feature_names), "unique feature names")
print(unique_feature_names)

print()
print("training_examples contains %d training examples" % len(training_examples))
print("validation_examples contains %d validation examples" % len(validation_examples))
print("test_examples contains %d test examples" % len(test_examples))
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



class Vectorizer:
    def __init__(self, feature_names):
        self.feat_dict = {feature : index for index, feature in enumerate(feature_names)}

    def transform(self, X_batch):
        '''
        X_batch is aX_pageview representing one single page view
        '''
        X = []
        for d in X_batch.depths(): # ranging from [1, 100]
            vec = [0] * len(self.feat_dict)
            for f, v in X_batch.aux_feats():
                vec[self.feat_dict[f]] = v

            X.append(vec)
        return np.array(X, dtype='float32')

vectorizer = Vectorizer(unique_feature_names)

del unique_feature_names


# np.set_printoptions(threshold=np.nan)


def Xy_gen(input_examples, batch_size=10):
    '''
    "input_examples" contains a list of X_pageview instances
    '''
    X_batch_u = []
    X_batch_p = []
    X_batch_ctx = []
    X_batch_dep = []
    y_batch = []

    for Xinst_pv in random.sample(input_examples, len(input_examples)): # shuffle pageviews
        ''' Xinst_pv is one pageview '''
        X_batch_ctx.append( vectorizer.transform(Xinst_pv) )
#         print(vectorizer.transform(Xinst_pv))
        
        X_batch_u.append(np.array([Xinst_pv.user_index] * 100))
        X_batch_p.append(np.array([Xinst_pv.page_index] * 100))

        X_batch_dep.append( Xinst_pv.depths() )
        y_batch.append( Xinst_pv.depth_truth )
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