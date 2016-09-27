'''
Created on Mar 25, 2016

@author: Wang
'''
from numpy import mean
from collections import defaultdict
from math import log

# page_dwell_time_threshold = 120
MIN_SCREEN_DWELL_TIME = 180

def is_valid_screen_dwell(dwell):
    if dwell > MIN_SCREEN_DWELL_TIME or dwell <= 0:
        return False
    return True

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
           
                    
    if skip_pageview or not pv_summary:
        return None
    
    return pv_summary


