'''
Created on Mar 25, 2016

@author: Wang
'''
from collections import defaultdict
from math import log

# page_dwell_time_threshold = 120
MIN_SCREEN_DWELL_TIME = 180

def is_valid_screen_dwell(dwell):
    if dwell > MIN_SCREEN_DWELL_TIME or dwell < 0:
        return False
    return True

def get_depth_dwell_time(loglist):
#     print(loglist)
    pv_summary = [] # [[screen_top, screen_bottom, dwell_time], [ ... ]]
    skip_pageview = False
    for index, log_dict in enumerate(loglist):
        additionalinfo = log_dict['additionalinfo']
        
#         try:
#             print(log_dict)
#         except UnicodeEncodeError:
#             print('UnicodeEncodeError')
        
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
        
    
                    
    if skip_pageview or not pv_summary:
        return None
    
    
#     depth_dwell_stats = get_depth_dwell_stats_conditional(pv_summary)
    depth_dwell_stats = get_depth_dwell_stats(pv_summary)

#             if dwell_time == 0:
#                 dwell_time = 0.01
#             depth_dwell_stats[depth] = log(dwell_time, 2)
            
    return depth_dwell_stats


def get_depth_dwell_stats_conditional(pv_summary):
    depth_dwell_stats = defaultdict(int)
    for screen_top, screen_bottom, dwell_time in pv_summary:
        for depth in range(screen_top, screen_bottom + 1):
            depth_dwell_stats[depth] = (dwell_time, screen_top, screen_bottom)
    return depth_dwell_stats

def get_depth_dwell_stats(pv_summary):
    mean_area_size = []
    depth_dwell_stats = defaultdict(tuple)
#     print(pv_summary)
    for screen_top, screen_bottom, dwell_time in pv_summary:
        mean_area_size.append(screen_bottom - screen_top)
        for depth in range(screen_top, screen_bottom + 1):
            if depth in depth_dwell_stats:
                depth_dwell_stats[depth] = (depth_dwell_stats[depth][0]+dwell_time, min(depth_dwell_stats[depth][1], screen_top), max(depth_dwell_stats[depth][2], screen_bottom))
            else:
                depth_dwell_stats[depth] = (dwell_time, screen_top, screen_bottom)
    mean_area_size = sum(mean_area_size) / len(mean_area_size)
            
    ''' fill the screens which were not scrolled '''
    for d in range(0, 101):
        if d not in depth_dwell_stats:
            simulated_top = d - mean_area_size if d - mean_area_size >= 0 else 0
            simulated_bottom = d + mean_area_size if d - mean_area_size <= 100 else 100
            depth_dwell_stats[d] = (0, simulated_top, simulated_bottom)
    return depth_dwell_stats
