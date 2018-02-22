'''
Created on Jun 23, 2017

@author: Wang
'''
import hashlib
import random, numpy as np
from collections import defaultdict, Counter
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
    nr_bin = 1e+6
    return str(int(int(hashlib.md5(s.encode('utf8')).hexdigest(), 16)%(nr_bin - 1) + 1))



# doc2vec = Doc2Vec.load('../doc2vec_models_py3/d2v_model_urlinuserlog_20.doc2vec')



def fm_file_writer(pageviews, outfile):
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
            #             features['depth'] = index + 1
            depth = index + 1 # range: [1, 100], must be an integer
            
            user_cell = ':'.join([hashstr(uid), '1'])
            page_cell = ':'.join([hashstr(clean_url), '1'])
            depth_cell = ':'.join([str(depth), '1'])
            
            vp_wid, vp_hei = discretize_pixel_area(viewport)
            viewport_cell = ':'.join([hashstr('x'.join([vp_wid, vp_hei])), '1'])
#             channel_cell = ':'.join([hashstr('='.join(['cha', channel])), '1'])
            
            
            
            '''
            Add Doc2Vec vector of this page
            '''
#             try:
#                 d2v_vec = doc2vec.docvecs[url] # the url here is the URL_IN_USERLOG 
# #                 print(d2v_vec)
#                 add_vector_features(features, 'd2v', d2v_vec)
#             except KeyError:
#                 # the missing will be "0,0,0,0,0"
#                 no_d2v_dep_num += 1
#                 continue
            

            
            output = ' '.join([str(dwell), user_cell, page_cell, depth_cell, viewport_cell])
#             print(output)
            
            outfile.write(output + '\n')
    
   

print("Building FM training input ...")
outfile_train = open('fm_training.txt', 'w')
fm_file_writer(tts.training_set, outfile_train) 


print("Building FM validation input ...")
outfile_vali = open('fm_validation.txt', 'w')
fm_file_writer(tts.validate_set, outfile_vali)  
 
 
print("Building FM test input ...")
outfile_test = open('fm_test.txt', 'w')
fm_file_writer(tts.test_set, outfile_test) 


print("Finish")
 
