'''
Created on Mar 25, 2016

@author: Wang
'''
from collections import defaultdict
from math import log

# page_dwell_time_threshold = 120
screen_dwell_time_threshold = 120

def get_depth_dwell_time(loglist):
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
                pv_summary[-1][2] = dwell_time - pv_summary[-1][2]
                if pv_summary[-1][2] > screen_dwell_time_threshold or pv_summary[-1][2] < 0:
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
                if pv_summary[-1][2] > screen_dwell_time_threshold or pv_summary[-1][2] < 0:
                    skip_pageview = True
                    break
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
                if pv_summary[-1][2] > screen_dwell_time_threshold or pv_summary[-1][2] < 0:
                    skip_pageview = True
                    break
            else:
                skip_pageview = True
            break
        else:
            continue
        
    
                    
    if skip_pageview:
        return None
    
    
    for screen_top, screen_bottom, dwell_time in pv_summary:
        if dwell_time > 200 or dwell_time < 0:
            print()
    
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
            depth_dwell_stats[depth] = dwell_time
    return depth_dwell_stats

def get_depth_dwell_stats(pv_summary):
    max_scroll_depth = 0
    depth_dwell_stats = defaultdict(int)
    for screen_top, screen_bottom, dwell_time in pv_summary:
        max_scroll_depth = screen_bottom
        for depth in range(screen_top, screen_bottom + 1):
            depth_dwell_stats[depth] = dwell_time
    
    for d in range(max_scroll_depth + 1, 101):
        depth_dwell_stats[d] = 0
    return depth_dwell_stats
