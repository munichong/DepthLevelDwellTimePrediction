'''
Created on Apr 1, 2016

@author: Wang
'''
import re, numpy as np
from pprint import pprint
from pymongo import MongoClient
from pymongo.errors import CursorNotFound
from dwell_time_calculation import viewport_behaviors, print_viewport_dwell_dist, print_seq_len_dist
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from user_agents import parse
from urllib.parse import urlparse
from models_training import TASK, VIEWABILITY_THRESHOLD



DATABASE = 'Forbes_Apr2016'
# DATABASE = 'Forbes_Dec2015'

client = MongoClient()
userlog = client[DATABASE]['FreqUserLogPV']
articleInfo = client[DATABASE]['ArticleInfo']


COLD_START_THRESHOLD = 10



client = MongoClient()
user_freq_table = client[DATABASE]['UserFreq_all']
page_freq_table = client[DATABASE]['PageFreq_all']

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

def categorize_device(raw_device):
#     if raw_device[:len('lg')] == 'lg' or raw_device == 'vk700':
#         return 'lg'
#     if raw_device[:len('sgp')] == 'sgp':
#         return 'sgp'
#     if raw_device[:len('rct')] == 'rct':
#         return 'rct'
#     if raw_device[:len('qmv')] == 'qmv':
#         return 'qmv'
    if raw_device[:len('ipad')] == 'ipad': # e.g., 'ipad', 'ipad4,1', 'ipad5,3'
        return 'ipad'
#     if raw_device[:len('hudl')] == 'hudl':
#         return 'hudl'
#     if raw_device[:len('lenovo')] == 'lenovo':
#         return 'lenovo'
    if raw_device[:len('blackberry')] == 'blackberry':
        return 'blackberry'
    if raw_device[:len('samsung sm')] == 'samsung sm':
        return 'samsung sm'
    if raw_device[:len('samsung gt')] == 'samsung gt':
        return 'samsung OTHER'
    if raw_device[:len('samsung sch')] == 'samsung sch':
        return 'samsung OTHER'
    if raw_device[:len('samsung sgh')] == 'samsung sgh':
        return 'samsung OTHER'
    if raw_device[:len('samsung sph')] == 'samsung sph':
        return 'samsung OTHER'
    if raw_device[:len('playstation')] == 'playstation':
        return 'playstation'
    if raw_device[:len('kindle fire')] == 'kindle fire':
        return 'kindle fire'
    if raw_device[:len('asus')] == 'asus': # e.g., 'asus nexus 7', 'asus nexus 10', 'asus me173x'
        return 'asus'
    
    
#     if raw_device in ['cw-vi8', 'kfsawi', 'd101', 'b1-750', 'pixel c', 'a1-840fhd', 'k010', 'pro7d', 'ns-15t8lte', 'a1',
#                       'vk410', 'vk810 4g', 'qtaqz3', 'motorola xoom', 'p01m', 'b1-750', 'le pan tc802a', 'asus me173x',
#                       'xiaomi mi pad', 'dpm7827', 'trio axs 4g', 'b1-720', 'at7-c', 'a3-a20', 'hp slate 7 voice tab',
#                       'ns-p16at10', 'a200', 'telpad qs', 'hp 10', 'nexus 9', 'hive v 3g' ] \
#                       or raw_device[:len('lenovo')] == 'lenovo' or raw_device[:len('hudl')] == 'hudl' \
#                       or  raw_device[:len('rct')] == 'rct' or raw_device[:len('qmv')] == 'qmv':
#         return 'generic tablet'
#     if raw_device[:len('venue')] == 'venue':
#         return 'generic tablet'
    if ' tablet' in raw_device:
        return 'generic tablet'
    
#     if raw_device in ['microsoft lumia 735', 'k007', 'htc 0p6b180'] or raw_device[:len('sgp')] == 'sgp' or raw_device[:len('lg')] == 'lg' or raw_device == 'vk700':
#         return 'generic smartphone'
    
    return raw_device

def categorize_browser(raw_browser):
    if raw_browser == 'firefox ios' or raw_browser == 'firefox mobile':
        return 'firefox ios/mobile'
    return raw_browser
    
def get_info_from_agent(ua_string):
    user_agent = parse(ua_string)
#     print(str(user_agent).split(' / '))
#     print([simplify_version(string) for string in str(user_agent).split(' / ')])
    device, os, browser = str(user_agent).lower().split(' / ')
    ''' remove version '''
    os  = remove_version(os)
    browser = remove_version(browser)
    ''' categorize device, os, and browser '''
    device = categorize_device(device)
    browser = categorize_browser(browser)
    
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
        freshness = int((pv_start_time - pub_time) / 86400) # unit: day
        if freshness > 10:
            freshness = '>10d'
        return str(freshness)
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

def get_commentCount(count):
    if count == 0:
        return '0'
    if count == 1:
        return '1'
    if count == 2:
        return '2'
    elif count > 2 and count <= 5:
        return '3<=5'
    elif count > 5 and count <= 10:
        return '5<=10'
    elif count > 10 and count <= 20:
        return '10<=20'
    else:
        return '>20'

def categorize_length(raw_length):
    length = int(raw_length / 100)
    if length < 20:
        return str(length)
    elif length >= 20 and length < 25:
        return '20=<25'
    elif length >= 25 and length < 30:
        return '25=<30'
    elif length >= 30 and length < 40:
        return '30=<40'
    else:
        return '>40'
    
        

def get_article_info(userlog_url, pv_start_time):
    for doc in articleInfo.find({'URL_IN_USERLOG':userlog_url}):
        
        if 'body' in doc:
            try:
                body_text = BeautifulSoup(doc['body'], "lxml").getText()
            except TypeError:
                break
            
            body_text = re.sub(r'\[.*?\]', ' ', body_text)
#             print(body_text)
            body_length = categorize_length(len(body_text.split()))
        else:
            break
#             body_text = 'unknown'
#             body_length = 'unknown'
        channel = doc['displayChannel'] if 'displayChannel' in doc else 'unknown'
        section = doc['displaySection'] if 'displaySection' in doc else 'unknown'
        channel_group = get_channel_group(doc) # list
        section_group = get_section_group(doc) # list
        freshness = get_freshness(doc, pv_start_time)
        
        page_type = doc['type'].lower() if 'type' in doc else 'unknown'
        templateType = doc['templateType'].lower() if 'templateType' in doc else 'unknown'
        blogType = doc['blogType'].lower() if 'blogType' in doc else 'unknown'
        storyType = doc['storyType'].lower() if 'storyType' in doc else 'unknown'
        image = 'true' if 'image' in doc else 'false'
        writtenByForbesStaff = str(doc['writtenByForbesStaff']).lower() if 'writtenByForbesStaff' in doc else 'unknown'
        calledOutCommentCount = get_commentCount(doc['calledOutCommentCount']) if 'calledOutCommentCount' in doc else 'unknown'
        
        return body_length, channel.lower(), section.lower(), channel_group, section_group, freshness, \
            page_type, templateType, blogType, storyType, image, writtenByForbesStaff, calledOutCommentCount
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

def remove_url_parameters(raw_url):
    ''' remove parameters in the raw_url 
        but keep page numbers '''
    parse_result = urlparse(raw_url)
    clean_url = '{0}://{1}{2}'.format(parse_result.scheme, parse_result.netloc, parse_result.path)
#     print("clean_url:", clean_url)
    return clean_url


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
                                          auxiliary['browser'], auxiliary['page_type'], auxiliary['templateType'],
                                          auxiliary['blogType'], auxiliary['storyType'], auxiliary['image'], 
                                          auxiliary['writtenByForbesStaff'], auxiliary['calledOutCommentCount']
                                          ))

    
        


valid_pv_num = 0
user_freq = defaultdict(int); page_freq = defaultdict(int);


user_num = 0
all_pageviews = []
done = False
depth_dwell_counter = Counter()
geo_counter = Counter()
channel_counter = Counter()
section_counter = Counter()
length_counter = Counter()
device_counter = Counter()
os_counter = Counter()
browser_counter = Counter()
commentCount_counter = Counter()
while not done:
    try:
        freq_uids = get_freq_uids(COLD_START_THRESHOLD)
        freq_uids.skip(user_num)
        for user_doc in freq_uids: # for each unique user
            uid = user_doc['uid']
                        
            if user_num % 2000 == 0:
                print(user_num)
            user_num += 1
            
            
            """ for each page view """
            for pv_doc in userlog.find({'uid':uid}):
                url = pv_doc['url']
                unix_start_time = pv_doc['unix_start_time']
                      
                 
#                 if unix_start_time < 1449792000 or unix_start_time > 1450656000:
#                     continue
                
                """ if this page is not a frequent page """
#                 if url not in freq_page_set:
#                     continue
                
        
                depth_dwell = viewport_behaviors(pv_doc['loglist']) 
#                 print(depth_dwell)
                if depth_dwell is None:
                    continue     
                
                if TASK == 'r' and all(v == 0 for v in depth_dwell):
                    continue
                
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
                
#                 print(article_info)
                body_length, channel, section, channel_group, section_group, freshness, \
                page_type, templateType, blogType, storyType, image, writtenByForbesStaff, calledOutCommentCount = article_info
                                
                
                device, os, browser = get_info_from_agent(pv_doc['ua'])
                
                if browser == 'moatbot': # suspect robot
                    continue
                
                
                depth_dwell_counter.update(depth_dwell)
                geo_counter.update([user_geo])
                channel_counter.update(channel_group)
                section_counter.update(section_group)
                length_counter.update([body_length])
                
                
                device_counter.update([device])
                os_counter.update([os])
                browser_counter.update([browser])
                commentCount_counter.update([calledOutCommentCount])
            
                
                ''' this is a valid page view '''
                valid_pv_num += 1
                user_freq[uid] += 1
                clean_url = remove_url_parameters(url)
                page_freq[clean_url] += 1
                
                pageview = Pageview(uid, clean_url, depth_dwell, screen=screen_size, 
                                    viewport=viewport_size, geo=user_geo, agent=pv_doc['ua'], 
                                    weekday=local_weekday, hour=local_hour, 
                                    length=body_length, channel=channel, section=section,
                                    channel_group=channel_group, section_group=section_group, 
                                    fresh=freshness, device=device, os=os, browser=browser,
                                    page_type=page_type, templateType=templateType, blogType=blogType,
                                    storyType=storyType, image=image, writtenByForbesStaff=writtenByForbesStaff,
                                    calledOutCommentCount=calledOutCommentCount
                                    )
                all_pageviews.append(pageview)
        done = True
    except CursorNotFound:
        print("pymongo.errors.CursorNotFound")
        print("Will start from", user_num)
        




print("\n*************** The Distribution of The Depth-level Dwell Time ***************")
total = sum(depth_dwell_counter.values())
for depth_dwell, count in sorted(depth_dwell_counter.items(), key=lambda x: x[0]):
    print(depth_dwell, "\t", count, "\t", count/total)
print("******************************")
del depth_dwell_counter

print_seq_len_dist()

'''
print_viewport_dwell_dist()
'''

print("\n*************** The Distribution of User Geo ***************")
total = sum(geo_counter.values())
geo_convert2OTHER = set()
for geo, count in sorted(geo_counter.items(), key=lambda x: x[1], reverse=True):
    if count/total < 0.001:
        geo_convert2OTHER.add(geo)
#         print(geo, "\t", count, "\t", count/total, '\t', "CONVERT TO 'other'")
    else:
#         print(geo, "\t", count, "\t", count/total)
        pass
print("******************************")
del geo_counter

'''
print("\n*************** The Distribution of Channel_Group ***************")
total = sum(channel_counter.values())
for channel, count in sorted(channel_counter.items(), key=lambda x: x[1], reverse=True):
    print(channel, "\t", count, "\t", count/total)
print("******************************")
del channel_counter

print("\n*************** The Distribution of Section_Group ***************")
total = sum(section_counter.values())
for section, count in sorted(section_counter.items(), key=lambda x: x[1], reverse=True):
    print(section, "\t", count, "\t", count/total)
print("******************************")
del section_counter
'''

print("\n*************** The Distribution of Body Length ***************")
total = sum(length_counter.values())
for length, count in sorted(length_counter.items(), key=lambda x: x[0]):
#     print(length, "\t", count, "\t", count/total)
    pass
print("******************************")
del length_counter


print("\n*************** The Distribution of Devices ***************")
total = sum(device_counter.values())
device_convert2OTHER = set()
for device, count in sorted(device_counter.items(), key=lambda x: x[1], reverse=True):
    if count/total < 0.0005:
        device_convert2OTHER.add(device)
        print(device, "\t", count, "\t", count/total, '\t', "CONVERT TO 'other'")
    else:
        print(device, "\t", count, "\t", count/total)
print("******************************")
del device_counter

print("\n*************** The Distribution of OS ***************")
total = sum(os_counter.values())
os_convert2OTHER = set()
for os, count in sorted(os_counter.items(), key=lambda x: x[1], reverse=True):
    if count < os_counter['other'] or count < 50:
        os_convert2OTHER.add(os)
        print(os, "\t", count, "\t", count/total, '\t', "CONVERT TO 'other'")
    else:
        print(os, "\t", count, "\t", count/total)
print("******************************")
del os_counter

print("\n*************** The Distribution of Browsers ***************")
total = sum(browser_counter.values())
browser_convert2OTHER = set()
for browser, count in sorted(browser_counter.items(), key=lambda x: x[1], reverse=True):
    if count < browser_counter['other']:
        browser_convert2OTHER.add(browser)
        print(browser, "\t", count, "\t", count/total, '\t', "CONVERT TO 'other'")
    else:
        print(browser, "\t", count, "\t", count/total)
print("******************************")
del browser_counter

print("\n*************** The Distribution of calledOutCommentCount ***************")
total = sum(commentCount_counter.values())
for commentCount, count in sorted(commentCount_counter.items(), key=lambda x: x[1], reverse=True):
    print(commentCount, "\t", count, "\t", count/total)
print("******************************")
del commentCount_counter







print()
print("=============== Statistics of Initial Data ================")
print("valid_pv_num =", valid_pv_num)
print(len(user_freq), " unique users and ", len(page_freq), " unique pages")
print("density =", valid_pv_num/float(len(user_freq) * len(page_freq)))
print()


user_freq2 = defaultdict(int)
page_freq2 = defaultdict(int)
valid_pv_num2 = 0
def filter_pageviews_by_minPVnum(pvs):
    FURTHER_COLD_START_THRESHOLD = 4 # fups.COLD_START_THRESHOLD
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

print()
print("=============== Statistics of Further Data ================")
print("valid_pv_num2 =", valid_pv_num2)
print(len(user_freq2), "unique users and", len(page_freq2), "unique pages")
print("density =", valid_pv_num2/float(len(user_freq2) * len(page_freq2)))
print()



print("\n=============== Separating Training and Test Data ================")
""" Randomly pick training, validation and test instances """
training_set = []
validate_set = []
test_set = []
# all_training_text = defaultdict(str) # url -> body_text
# all_test_text = defaultdict(str)
users_in_train = defaultdict(int)
pages_in_train = defaultdict(int)
users_in_val = defaultdict(int)
pages_in_val = defaultdict(int)

# np.random.shuffle(all_pageviews)
for pv in all_pageviews:
    uid = pv.uid
    url = pv.url
    
    if uid not in users_in_train or url not in pages_in_train:
        ''' if this is the first time that we see this user or this page '''
        training_set.append(pv)
#         if uid == '49bdb98c-d8c4-1bce-bb92-a2e0f9f3a3d5':
#             print('1', users_in_train[uid] / user_freq2[uid], pages_in_train[url] / page_freq2[url])
        ''' mark that the user and the page has been added to training data once '''
        users_in_train[uid] += 1
        pages_in_train[url] += 1
        
    elif (users_in_train[uid] / user_freq2[uid] < 0.7 or
        pages_in_train[url] / page_freq2[url] < 0.7):
        ''' if we have seen this user and page, but not enough history in the training data '''
        training_set.append(pv)
#         if uid == '49bdb98c-d8c4-1bce-bb92-a2e0f9f3a3d5':
#             print('2', users_in_train[uid] / user_freq2[uid], pages_in_train[url] / page_freq2[url])
        ''' mark that the user and the page has been added to training data once '''
        users_in_train[uid] += 1
        pages_in_train[url] += 1
        
    else:
        if (users_in_val[uid] / (user_freq2[uid] - users_in_train[uid]) < 0.95 and
        pages_in_val[url] / (page_freq2[url] - pages_in_train[url]) < 0.95):
            validate_set.append(pv)
            users_in_val[uid] += 1
            pages_in_val[url] += 1
#             if uid == '49bdb98c-d8c4-1bce-bb92-a2e0f9f3a3d5':
#                 print('3')
        else:
            test_set.append(pv)



if TASK == 'c':
    positive_num_train = {0:0, 1:0}
    for pv in training_set:
        for depth_row in pv.depth_level_rows:
            positive_num_train[depth_row[0]] += 1
    print("\nIn the training data, %f depth dwell times are at least %d seconds" % (positive_num_train[1]/sum(positive_num_train.values()), VIEWABILITY_THRESHOLD))
    
    positive_num_val = {0:0, 1:0}
    for pv in validate_set:
        for depth_row in pv.depth_level_rows:
            positive_num_val[depth_row[0]] += 1
    print("In the validation data, %f depth dwell times are at least %d seconds" % (positive_num_val[1]/sum(positive_num_val.values()), VIEWABILITY_THRESHOLD))



print()
if not (users_in_val.keys() - users_in_train.keys()):
    print('The users in validation data are also in the training data')
else:
    print('!!! Some users in validation data are NOT in the training data !!!')
if not (pages_in_val.keys() - pages_in_train.keys()):
    print('The pages in validation data are also in the training data')
else:
    print('!!! Some pages in validation data are NOT in the training data !!!')
print()      



# print("Users in the training data")
# print("(num_of occurrence, num_of_users)")
# pprint(sorted(Counter(users_in_train.values()).most_common(), key=lambda x:x[0], reverse=False))
# print("Pages in the training data")
# print("(num_of occurrence, num_of_pages)")
# pprint(sorted(Counter(pages_in_train.values()).most_common(), key=lambda x:x[0], reverse=False))


val_userFreq_in_train = Counter()
val_pageFreq_in_train = Counter()
checked_users = set()
checked_pages = set()
for pv in validate_set:
    uid = pv.uid
    url = pv.url
    if uid not in checked_users:
        val_userFreq_in_train.update([users_in_train[uid]])
        checked_users.add(uid)
    if url not in checked_pages:
        val_pageFreq_in_train.update([pages_in_train[url]])
        checked_pages.add(url)

print()
print("The frequency of validation users in the training data:")
for num_in_train, user_count in sorted(val_userFreq_in_train.items(), key=lambda x: x[0]):
    print("%d users in validation set have %d record in training data" % (user_count, num_in_train))
print()
print("The frequency of validation pages in the training data:")
for num_in_train, page_count in sorted(val_pageFreq_in_train.items(), key=lambda x: x[0]):
    print("%d pages in validation set have %d record in training data" % (page_count, num_in_train))
print()


# for u, c in users_in_train.items():
#     if c > 60:
# #     if users_in_train[u] == 1:
#         print(u, ':',  c, 'in train,', users_in_val[u], 'in val')


del val_userFreq_in_train
del val_pageFreq_in_train
del checked_users
del checked_pages


        
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
del users_in_train
del pages_in_train
del users_in_val
del pages_in_val

del all_pageviews


