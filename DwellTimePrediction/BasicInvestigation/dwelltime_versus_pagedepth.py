'''
Created on Jan 3, 2016

@author: Wang
'''
from pymongo import MongoClient 

from numpy import array
import numpy as np
# import pylab as plt
import matplotlib.pyplot as plt

client = MongoClient()
featVec_DB = client['Forbes_Dec2015']['FreqUserLogPV']

def total_dwell_depth_distribution():
    ''' initialization '''
    depth_dwell_stats = {}
    for depth in range(101):
        depth_dwell_stats[depth] = 0
        
    n = 0    
    for pv_doc in featVec_DB.find():
    #     ''' skip pageviews which only has 0 or 1 log '''
    #     if len(pv_doc['loglist']) <= 1:
    #         continue
        print(n)
        n += 1
        
        pv_summary = [] # [[screen_top, screen_bottom, dwell_time], [ ... ]]
        skip_pageview = False
        for index, log_dict in enumerate(pv_doc['loglist']):
            additionalinfo = log_dict['additionalinfo']
            
            if log_dict["eventname"] == "Post reading":
                if 'Percentage reading from' not in additionalinfo:
                    skip_pageview = True
                    break
                screen_top = additionalinfo['Percentage reading from']
                screen_bottom = additionalinfo['Percentage of reading']
                dwell_time = additionalinfo['Time on article']
                if dwell_time > 120:
                    skip_pageview = True
                    break
                
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = dwell_time - pv_summary[-1][2]
                if index < len(pv_doc['loglist']) - 1:
                    pv_summary.append([screen_top, screen_bottom, dwell_time])
                    
            elif log_dict["eventname"] == "Text selection":
                continue    
            elif log_dict["eventname"] == "Probably left":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Load new article":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Page leave":
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = additionalinfo['Total time'] - pv_summary[-1][2]
                else:
                    skip_pageview = True
                break
            else:
                continue
            
        if not skip_pageview:
            ''' Content area overlap is allowed '''
            for screen_top, screen_bottom, dwell_time in pv_summary:
                for depth in range(screen_top, screen_bottom + 1):
                    depth_dwell_stats[depth] += dwell_time
                    
    
    print()
    for depth in range(101):
        print(depth, ',', depth_dwell_stats[depth])
    
    
def mean_dwell_depth_distribution():
    ''' initialization '''
    depth_dwell_stats = {}
    for depth in range(101):
        depth_dwell_stats[depth] = []
        
    n = 0    
    valid_total_num = 0
    for pv_doc in featVec_DB.find():
    #     ''' skip pageviews which only has 0 or 1 log '''
    #     if len(pv_doc['loglist']) <= 1:
    #         continue
        print(n)
        n += 1
        
        if n > 300000:
            break
        
        pv_summary = [] # [[screen_top, screen_bottom, dwell_time], [ ... ]]
        skip_pageview = False
        for index, log_dict in enumerate(pv_doc['loglist']):
            additionalinfo = log_dict['additionalinfo']
            
            if log_dict["eventname"] == "Post reading":
                if 'Percentage reading from' not in additionalinfo:
                    skip_pageview = True
                    break
                screen_top = additionalinfo['Percentage reading from']
                screen_bottom = additionalinfo['Percentage of reading']
                dwell_time = additionalinfo['Time on article']
                if dwell_time > 180:
                    skip_pageview = True
                    break
                
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = dwell_time - pv_summary[-1][2]
                if index < len(pv_doc['loglist']) - 1:
                    pv_summary.append([screen_top, screen_bottom, dwell_time])
                    
            elif log_dict["eventname"] == "Text selection":
                continue    
            elif log_dict["eventname"] == "Probably left":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Load new article":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Page leave":
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = additionalinfo['Total time'] - pv_summary[-1][2]
                else:
                    skip_pageview = True
                break
            else:
                continue
            
        if not skip_pageview:
            ''' Content area overlap is allowed '''
            valid_total_num += 1
            for screen_top, screen_bottom, dwell_time in pv_summary:
                for depth in range(screen_top, screen_bottom + 1):
                    depth_dwell_stats[depth].append(dwell_time)
                    
    
    print()
    mean_dwells = []
    stds = []
    depth_dwells = []
    for depth in range(101):
#         depth_dwells.append(depth_dwell_stats[depth] + [0] * (valid_total_num - len(depth_dwell_stats[depth])))
        mean_dwells.append(np.sum(depth_dwell_stats[depth]) / float(valid_total_num))
        stds.append(np.std(depth_dwell_stats[depth] + [0] * (valid_total_num - len(depth_dwell_stats[depth])) ))
#         print(depth, depth_dwell_stats[depth])
    plt.bar(range(101), mean_dwells, color='r', tick_label=range(101), yerr=stds)
    
    plt.show()

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.boxplot(depth_dwells)
#     ax.set_xticklabels(list(range(101)))
# 
#     plt.show()

def post_reading_last_log_distribution():
    ''' initialization '''
    last_position_counts = {}
    for depth in range(101):
        last_position_counts[depth] = 0

    n = 0
    for pv_doc in featVec_DB.find():
    #     ''' skip pageviews which only has 0 or 1 log '''
    #     if len(pv_doc['loglist']) <= 1:
    #         continue
        print(n)
        n += 1
        
        last_log = pv_doc['loglist'][-1]
        
        if last_log["eventname"] != "Post reading":
            continue
        
        additionalinfo = last_log['additionalinfo']
        if 'Percentage reading from' not in additionalinfo:
            continue
        screen_top = additionalinfo['Percentage reading from']
        screen_bottom = additionalinfo['Percentage of reading']
        
        ''' record '''
        for depth in range(screen_top, screen_bottom + 1):
            last_position_counts[depth] += 1
                    
    print()
    for depth in range(101):
        print(depth, ',', last_position_counts[depth])                

def page_depth_low_dwell_counts():
    ''' initialization '''
    depth_counts = {}
    for depth in range(101):
        depth_counts[depth] = 0
    
    n = 0
    valid_pv_counts = {}
    for depth in range(101):
        valid_pv_counts[depth] = 0
        
    for pv_doc in featVec_DB.find():
    #     ''' skip pageviews which only has 0 or 1 log '''
    #     if len(pv_doc['loglist']) <= 1:
    #         continue
        print(n)
        
#         if n == 3:
#             break
        
        n += 1
        
        pv_summary = [] # [[screen_top, screen_bottom, dwell_time], [ ... ]]
        skip_pageview = False
        max_scroll_depth = 0
        for index, log_dict in enumerate(pv_doc['loglist']):
            additionalinfo = log_dict['additionalinfo']
            
            if log_dict["eventname"] == "Post reading":
                if 'Percentage reading from' not in additionalinfo:
                    skip_pageview = True
                    break
                screen_top = additionalinfo['Percentage reading from']
                screen_bottom = additionalinfo['Percentage of reading']
                
                
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = additionalinfo['Time on article'] - pv_summary[-1][2]
                if index < len(pv_doc['loglist']) - 1:
                    pv_summary.append([screen_top, screen_bottom, additionalinfo['Time on article']])
                    if screen_bottom > max_scroll_depth:
                        ''' UNCOMMENT this to get During or After scrolling '''
                        max_scroll_depth = 100
                        ''' UNCOMMENT this to get During scrolling '''
#                         max_scroll_depth = screen_bottom 
                    
            elif log_dict["eventname"] == "Text selection":
                continue    
            elif log_dict["eventname"] == "Probably left":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Load new article":
                skip_pageview = True
                break
            elif log_dict["eventname"] == "Page leave":
                if len(pv_summary) > 0:
                    pv_summary[-1][2] = additionalinfo['Total time'] - pv_summary[-1][2]
                else:
                    skip_pageview = True
                    break
            else:
                continue
        
        
        if not skip_pageview:
            for depth in range(max_scroll_depth + 1):
                valid_pv_counts[depth] += 1
            
            for depth in range(max_scroll_depth + 1):
                depth_counts[depth] += 1
            
#             if n == 2:
#                 print()
            
            ''' Content area overlap is allowed '''
            has_recorded = []
#             print(pv_summary)
            for screen_top, screen_bottom, dwell_time in pv_summary:
                ''' if the rest of the page has low dwell time because it is not scrolled to, do NOT count it. '''
                for depth in range(screen_top, screen_bottom + 1):
                    ''' LOW DWELL TIME THRESHOLD '''
                    if depth not in has_recorded and dwell_time >= 10:
                        depth_counts[depth] -= 1
#                         depth_counts[depth] += 1
                        has_recorded.append(depth)
#                 print()
#         if valid_pv_counts[100] != 0:
#             print()
#     print(depth_counts)
#     print()
#     print(valid_pv_counts)      
    
    print()
    for depth in range(101):
        if float(valid_pv_counts[depth]) == 0:
            print(depth, ',', 0)
        else:
            print(depth, ',', depth_counts[depth] / float(valid_pv_counts[depth]))
        
    

if __name__ == "__main__":
    mean_dwell_depth_distribution()
#     post_reading_last_log_distribution()             
#     page_depth_low_dwell_counts()