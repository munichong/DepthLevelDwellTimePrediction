'''
Created on Mar 25, 2016

@author: Wang
'''
import numpy as np
from numpy import mean
from collections import defaultdict
from math import log
from collections import Counter

# page_dwell_time_threshold = 120
MIN_SCREEN_DWELL_TIME = 60
MAX_SEQ_LEN = 10


def is_valid_screen_dwell(dwell):
    if dwell > MIN_SCREEN_DWELL_TIME or dwell <= 0:
        return False
    return True

viewport_dwell_counter = Counter()
seq_len_counter = Counter()

def viewport_behaviors(loglist):
#     print(loglist)
    pv_summary = [] # [[screen_top, screen_bottom, dwell_time], [ ... ]]
    skip_pageview = False
    for index, log_dict in enumerate(loglist):
        additionalinfo = log_dict['additionalinfo']
        
        
        if log_dict["eventname"] == "Post reading":
            if 'Percentage reading from' not in additionalinfo:
                skip_pageview = True
                break
            screen_top = additionalinfo['Percentage reading from']
            screen_bottom = additionalinfo['Percentage of reading']
            dwell_time = additionalinfo['Time on article']
            
#             if dwell_time > page_dwell_time_threshold or dwell_time < 0:
#                 skip_pageview = True
#                 break
            
            if dwell_time < 0:
                skip_pageview = True
                break
                
            if len(pv_summary) > 0: 
                ''' If this is NOT the first log in this pageview '''
                ''' calculate the dwell time of the previous screen '''
                pv_summary[-1][2] = dwell_time - pv_summary[-1][2] # calculate the dwell time of the previous screen
                if not is_valid_screen_dwell(pv_summary[-1][2]):
                    skip_pageview = True
                    break
            if index < len(loglist) - 1: 
                ''' If this log is NOT the last log in this pageview, add it momentarily, 
                the dwell time is supposed to be updated in the next iteration. '''
                pv_summary.append([screen_top, screen_bottom, dwell_time])
                    
        elif log_dict["eventname"] == "Text selection":
            if index == len(loglist) - 1 and len(pv_summary) > 0: 
                ''' If this log is the last log in this pageview '''
                dwell_time = additionalinfo['Time on article']
                pv_summary[-1][2] = dwell_time - pv_summary[-1][2]
                if not is_valid_screen_dwell(pv_summary[-1][2]):
                    skip_pageview = True
                    break
            ''' If this log is NOT the last log in this pageview, skip this row '''
            continue    
        elif log_dict["eventname"] == "Probably left":
            skip_pageview = True
            break
        elif log_dict["eventname"] == "Load new article":
            skip_pageview = True
            break
        elif log_dict["eventname"] == "Page leave":
            if len(pv_summary) > 0:
                ''' calculate the dwell time of the previous screen '''
                pv_summary[-1][2] = additionalinfo['Total time'] - pv_summary[-1][2]
                if not is_valid_screen_dwell(pv_summary[-1][2]):
                    skip_pageview = True
                    break
            else:
                skip_pageview = True
            break
        else:
            continue
           
           
    ''' Remove the pageviews which have no action or have too many actions '''
    
    if skip_pageview or not pv_summary:
        return None
    
    seq_len_counter.update([len(pv_summary)])
    if len(pv_summary) > MAX_SEQ_LEN:
        return None
        
    viewport_dwell_counter.update( [s[2] for s in pv_summary] )
    
    depth_dwell_stats = get_depth_dwell_stats(pv_summary)
    
    
#     if np.count_nonzero(depth_dwell_stats) < 99:
#         return None
            
    return depth_dwell_stats


def print_seq_len_dist():
    print("\n*************** The Distribution of The Action Counts ***************")
    global seq_len_counter
    total = sum(seq_len_counter.values())
    for seq_len, count in seq_len_counter.items():
        print(seq_len, '\t', count, '\t', count/total)
    print("******************************")
    del seq_len_counter

def print_viewport_dwell_dist():
    print("\n*************** The Distribution of The Viewport-level Dwell Time ***************")
    global viewport_dwell_counter
    total = sum(viewport_dwell_counter.values())
    for viewport_dwell, count in viewport_dwell_counter.items():
        print(viewport_dwell, '\t', count, '\t', count/total)
    print("******************************")
    del viewport_dwell_counter


def get_depth_dwell_stats(pv_summary):    
#     segment_boudaries = sorted(list(set([top for top, _, _ in pv_summary] + 
#                                         [bottom for _, bottom, _ in pv_summary])))
    
    ''' accumulate the dwell time of each page depth 
        (The depth that was not viewed will be filled with 0 here)'''
    depth_dwell_stats = {}
    for depth in range(1, 101):
        accumu_dwell = 0
        for top, bottom, dwell in pv_summary:
            if depth >= top and depth <= bottom:
#             if depth >= top and (depth == 100 or depth < bottom):
                accumu_dwell += dwell
        depth_dwell_stats[depth] = accumu_dwell

    ''' Find the corresponding viewport window for each page depth '''
#     for depth in range(1, 101):
#         for i in range(len(segment_boudaries)):
#             ''' if this depth is at the end of the page which was not viewed '''
#             if i+1 == len(segment_boudaries):
#                 depth_dwell_stats[depth] = (depth_dwell_stats[depth], segment_boudaries[i], 100)
#                 break
#             ''' if this depth is at the beginning of the page which was not viewed '''
#             if depth < segment_boudaries[0]:
#                 depth_dwell_stats[depth] = (depth_dwell_stats[depth], 1, segment_boudaries[i])
#                 break
#             ''' if the depth is in the area which was viewed '''
#             if depth >= segment_boudaries[i] and depth < segment_boudaries[i+1]:
#                 depth_dwell_stats[depth] = (depth_dwell_stats[depth], segment_boudaries[i], segment_boudaries[i+1])
#                 break  
    
#     return depth_dwell_stats
    return tuple(depth_dwell_stats.values())
