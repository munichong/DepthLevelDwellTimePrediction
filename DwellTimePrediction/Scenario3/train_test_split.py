'''
Created on Apr 1, 2016

@author: Wang
'''
import re, numpy as np
from pymongo import MongoClient
from pymongo.errors import CursorNotFound
from dwell_time_calculation import viewport_behaviors, print_viewport_dwell_dist
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from user_agents import parse


client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']
articleInfo = client['Forbes_Dec2015']['ArticleInfo']


COLD_START_THRESHOLD = 5



client = MongoClient()
user_freq_table = client['Forbes_Dec2015']['UserFreq_all']
page_freq_table = client['Forbes_Dec2015']['PageFreq_all']

def get_freq_uids(threshold=0):
    return user_freq_table.find({"freq": {"$gte":threshold}}, {'uid':1})

def get_freq_urls(threshold=0):
    return page_freq_table.find({"freq": {"$gte":threshold}}, {'url':1})

# freq_uids = get_freq_uids(COLD_START_THRESHOLD) # The uids of freq users
# freq_urls = get_freq_urls(COLD_START_THRESHOLD) # The urls of freq pages
# 
# print(freq_uids.count(), "unique users")
# print(freq_urls.count(), "unique urls")
# print()


freq_page_set = set()
for page_doc in get_freq_urls(COLD_START_THRESHOLD):
    freq_page_set.add(page_doc['url'])

def simplify_version(raw_version):
    # mainly for browsers
    if '.' in raw_version:
        return raw_version[:raw_version.index('.')]
    return raw_version

def remove_version(raw_str):
    sim_str = simplify_version(raw_str)
    if sim_str.split()[-1].isdigit():
        return ' '.join(sim_str.split()[:-1])
    return sim_str
    
def get_info_from_agent(ua_string):
    user_agent = parse(ua_string)
#     print(str(user_agent).split(' / '))
#     print([simplify_version(string) for string in str(user_agent).split(' / ')])
    device, os, browser = str(user_agent).lower().split(' / ')
    os  = remove_version(os)
    browser = remove_version(browser)
    return device, os, browser
#     return [simplify_version(string) for string in str(user_agent).split(' / ')]
#     print(user_agent.device.family, user_agent.device.brand, user_agent.device.model)
#     print(user_agent.os.family, simplify_version(user_agent.os.version_string))
#     print(user_agent.browser.family, simplify_version(user_agent.browser.version_string))
#     return (' '.join([user_agent.device.family, user_agent.device.brand, user_agent.device.model]),
#             ' '.join([user_agent.os.family, simplify_version(user_agent.os.version_string)]),
#             ' '.join([user_agent.browser.family, simplify_version(user_agent.browser.version_string)])
#             )

def get_user_geo(country, state):
    if country and country != 'US':
        return country
    if country == 'US' and state:
        return '-'.join((country, state))
    else:
        return 'US'


def get_freshness(doc, pv_start_time):
    if 'date' in doc:
        pub_time = int(str(doc['date'])[:-3])
        freshness_hour = int((pv_start_time - pub_time) / 86400) # unit: day
        if freshness_hour > 10:
            freshness_hour = '>10d'
        return str(freshness_hour)
    else:
        return 'unknown'

def get_channel_group(doc):
    if 'channelSection' not in doc:
        return ['unknown']
    channel_group = set()
    for chanSec_dict in doc['channelSection']:
        channel_group.add(chanSec_dict['channelId'].lower())
    return list(channel_group)

def get_section_group(doc):
    if 'channelSection' not in doc:
        return ['unknown']
    section_group = set()
    for chanSec_dict in doc['channelSection']:
        if 'sectionId' in chanSec_dict:
            section_group.add(chanSec_dict['sectionId'].lower())
    return list(section_group)

def get_article_info(userlog_url, pv_start_time):
    for doc in articleInfo.find({'URL_IN_USERLOG':userlog_url}):
        
        if 'body' in doc:
            try:
                body_text = BeautifulSoup(doc['body']).getText()
            except TypeError:
                break
            
            body_text = re.sub(r'\[.*?\]', ' ', body_text)
#             print(body_text)
            body_length = str(int(len(body_text.split()) / 100))
        else:
            body_text = 'unknown'
            body_length = 'unknown'
        channel = doc['displayChannel'] if 'displayChannel' in doc else 'unknown'
        section = doc['displaySection'] if 'displaySection' in doc else 'unknown'
        channel_group = get_channel_group(doc) # list
        section_group = get_section_group(doc) # list
        freshness = get_freshness(doc, pv_start_time)
        
        return body_length, channel.lower(), section.lower(), channel_group, section_group, freshness
#     return ['unknown', 'unknown', 'unknown', ['unknown'], ['unknown'], 'unknown']
    return None
    
    
def get_body_text(userlog_url):
    for doc in articleInfo.find({'URL_IN_USERLOG':userlog_url}):
        
        if 'body' in doc:
            body_text = BeautifulSoup(doc['body']).getText()
            body_text = re.sub(r'\[.*?\]', ' ', body_text)
        else:
            body_text = 'unknown'
        
        return body_text.lower()
    return 'unknown'


class Pageview:
    def __init__(self, uid, url, depth_dwell, **auxiliary):
        self.uid = uid
        self.url = url
        self.screen, self.viewport = auxiliary['screen'], auxiliary['viewport']
        self.create_depth_level(uid, url, depth_dwell, auxiliary)
    
    def create_depth_level(self, uid, url, depth_dwell, auxiliary):
        """ convert pageview level data to depth level data
        depth level data are used to train the predictive model """
        self.depth_level_rows = []
        for depth, dwell in enumerate(depth_dwell, start=1):
            self.depth_level_rows.append((dwell, uid, url, 
#                                           top, bottom, 
                                          auxiliary['screen'], auxiliary['viewport'], 
                                          auxiliary['geo'], auxiliary['agent'],
                                          auxiliary['weekday'], auxiliary['hour'], 
                                          auxiliary['length'], auxiliary['channel'], auxiliary['section'],
                                          auxiliary['channel_group'], auxiliary['section_group'],
                                          auxiliary['fresh'], auxiliary['device'], auxiliary['os'],
                                          auxiliary['browser']
                                          ))

    
        


valid_pv_num = 0
user_freq = defaultdict(int); page_freq = defaultdict(int);


user_num = 0
all_pageviews = []
done = False
depth_dwell_counter = Counter()
device_counter = Counter()
os_counter = Counter()
browser_counter = Counter()
while not done:
    try:
        freq_uids = get_freq_uids(COLD_START_THRESHOLD)
        freq_uids.skip(user_num)
        for user_doc in freq_uids: # for each unique user
            uid = user_doc['uid']
            
            user_num += 1
            print(user_num)
            
            """ for each page view """
            for pv_doc in userlog.find({'uid':uid}):
                url = pv_doc['url']
                unix_start_time = pv_doc['unix_start_time']
                      
                 
                if unix_start_time < 1449792000 or unix_start_time > 1450656000:
                    continue
                
                """ if this page is not a frequent page """
#                 if url not in freq_page_set:
#                     continue
                
        
                depth_dwell = viewport_behaviors(pv_doc['loglist']) # [[screen_top, screen_bottom, dwell_time], [ ... ]]
#                 print(depth_dwell)
                if depth_dwell is None or all(v == 0 for v in depth_dwell):
                    continue                
                
                
                depth_dwell_counter.update(depth_dwell)
                
                
                ''' AUXILIARY FEATURES '''
                screen_size = pv_doc['loglist'][0]['additionalinfo']['screenSize'] if 'screenSize' in pv_doc['loglist'][0]['additionalinfo'] else 'unknown'
                
                viewport_size = pv_doc['loglist'][0]['additionalinfo']['viewportSize'] if 'viewportSize' in pv_doc['loglist'][0]['additionalinfo'] else 'unknown'
                
                user_geo = get_user_geo(pv_doc['country'], pv_doc['state'])
                
                isodate = pv_doc['local_start_time']
                if isodate:
                    local_weekday, local_hour = str(isodate.weekday()), str(isodate.hour) # The range of weekday is [0, 6]
                else:
                    local_weekday, loca_hour = 'unknown', 'unknown'
                
                article_info = get_article_info(url, pv_doc['unix_start_time'])
                if not article_info:
                    continue
                body_length, channel, section, channel_group, section_group, freshness = article_info
                
                
                device, os, browser = get_info_from_agent(pv_doc['ua'])
                device_counter.update([device])
                os_counter.update([os])
                browser_counter.update([browser])
            
                
                ''' this is a valid page view '''
                valid_pv_num += 1
                user_freq[uid] += 1
                page_freq[url] += 1
                
                pageview = Pageview(uid, url, depth_dwell, screen=screen_size, 
                                    viewport=viewport_size, geo=user_geo, agent=pv_doc['ua'], 
                                    weekday=local_weekday, hour=local_hour, 
                                    length=body_length, channel=channel, section=section,
                                    channel_group=channel_group, section_group=section_group, 
                                    fresh=freshness, device=device, os=os, browser=browser
                                    )
                all_pageviews.append(pageview)
        done = True
    except CursorNotFound:
        print("pymongo.errors.CursorNotFound")
        print("Will start from", user_num)
        
print()
print("=============== Statistics of Initial Data ================")
print("valid_pv_num =", valid_pv_num)
print(len(user_freq), " unique users and ", len(page_freq), " unique pages")
print("density =", valid_pv_num/float(len(user_freq) * len(page_freq)))
print()



print("\n*************** The Distribution of The Depth-level Dwell Time ***************")
total = sum(depth_dwell_counter.values())
for depth_dwell, count in sorted(depth_dwell_counter.items(), key=lambda x: x[0]):
    print(depth_dwell, "\t", count, "\t", count/total)
print("******************************")
del depth_dwell_counter

print("\n*************** The Distribution of Devices ***************")
total = sum(device_counter.values())
for device, count in sorted(device_counter.items(), key=lambda x: x[1], reverse=True):
    print(device, "\t", count, "\t", count/total)
print("******************************")
del device_counter

print("\n*************** The Distribution of OS ***************")
total = sum(os_counter.values())
for os, count in sorted(os_counter.items(), key=lambda x: x[1], reverse=True):
    print(os, "\t", count, "\t", count/total)
print("******************************")
del os_counter

print("\n*************** The Distribution of Devices ***************")
total = sum(browser_counter.values())
for browser, count in sorted(browser_counter.items(), key=lambda x: x[1], reverse=True):
    print(browser, "\t", count, "\t", count/total)
print("******************************")
del browser_counter

print_viewport_dwell_dist()



user_freq2 = defaultdict(int)
page_freq2 = defaultdict(int)
valid_pv_num2 = 0
def filter_pageviews_by_minPVnum(pvs):
    FURTHER_COLD_START_THRESHOLD = 0 # fups.COLD_START_THRESHOLD
    filtered_dataset = []
    global valid_pv_num2, user_freq2, page_freq2
    for pv in pvs:
        uid = pv.uid
        url = pv.url
        if ( user_freq[uid] < FURTHER_COLD_START_THRESHOLD or 
             page_freq[url] < FURTHER_COLD_START_THRESHOLD ):
            continue
        user_freq2[uid] += 1
        page_freq2[url] += 1 
        valid_pv_num2 += 1
        filtered_dataset.append(pv)
    return filtered_dataset


all_pageviews = filter_pageviews_by_minPVnum(all_pageviews)

# print()
# print("=============== Statistics of Further Data ================")
# print("valid_pv_num2 =", valid_pv_num2)
# print(len(user_freq2), "unique users and", len(page_freq2), "unique pages")
# print("density =", valid_pv_num2/float(len(user_freq2) * len(page_freq2)))
# print()

# print("(count, freqOfCount)")
# print( Counter(user_freq2.values()).most_common() )
# print( Counter(page_freq2.values()).most_common() )
# 
# print(sorted(Counter(user_freq2.values()).most_common(), key=lambda x:x[0], reverse=False))
# print(sorted(Counter(page_freq2.values()).most_common(), key=lambda x:x[0], reverse=False))



print("\n=============== Separating Training and Test Data ================")
""" Randomly pick training, validation and test instances """
training_set = []
validate_set = []
test_set = []
# all_training_text = defaultdict(str) # url -> body_text
# all_test_text = defaultdict(str)
# users_have_been_in_test = set()
# pages_have_been_in_test = set()

# np.random.shuffle(all_pageviews)
for pv in all_pageviews:
#     if ( user_freq2[uid] < fups.COLD_START_THRESHOLD or 
#         page_freq2[url] < fups.COLD_START_THRESHOLD ):
#         continue
#     body_text = get_body_text(pv.url)

#     if ( user_freq2[pv.uid] > COLD_START_THRESHOLD and 
#          page_freq2[pv.url] > COLD_START_THRESHOLD and
    if ( user_freq2[pv.uid] > 1 and 
         page_freq2[pv.url] > 1 and
         len(test_set) / len(all_pageviews) <= 0.1):
        test_set.append(pv)
#         if body_text != 'unknown':
#             all_test_text[pv.url] = body_text
        user_freq2[pv.uid] -= 1
        page_freq2[pv.url] -= 1
    elif (user_freq2[pv.uid] > 1 and 
         page_freq2[pv.url] > 1 and
         len(validate_set) / len(all_pageviews) <= 0.1):
        validate_set.append(pv)
#         if body_text != 'unknown':
#             all_test_text[pv.url] = body_text
        user_freq2[pv.uid] -= 1
        page_freq2[pv.url] -= 1
    else:
        training_set.append(pv)
#         if body_text != 'unknown':
#             all_training_text[pv.url] = body_text
            
        
print()
print(len(training_set), "pageviews in the training set")
print(len(validate_set), "pageviews in the validation set")
print(len(test_set), "pageviews in the test set")
print("The fraction of training data is", len(training_set) / float(valid_pv_num2))
print("The fraction of validation data is", len(validate_set) / float(valid_pv_num2))
print("The fraction of test data is", len(test_set) / float(valid_pv_num2))
print()


del userlog
del articleInfo

del user_freq_table
del page_freq_table
del user_freq
del page_freq
del user_freq2
del page_freq2

del all_pageviews


