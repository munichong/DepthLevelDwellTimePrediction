'''
Created on Mar 25, 2016

@author: Wang
'''
from math import sqrt
from pymongo import MongoClient
from BasicInvestigation.dwell_time_calculation import get_depth_dwell_time
from LibFM import FreqUserPageSearcher as fups
from httpagentparser import detect
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss

from LibFM import Training_test_generator as ttg

client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']
article_DB = client['Forbes_Dec2015']['ArticleInfo']


required_dwell_time = 7
    
    
def get_device(ua_string):
    """ {'platform': {'version': None, 'name': 'Linux'}, 'browser':{'version':'5.0', 'name': 'Chrome'}, 
             'os':{'name':'Linux'}, 'bot':False}
        If the ua_string cannot be parsed, return {'platform':{'version': None, 'name':None}} 
    """
    parsed_ua = detect(ua_string)
    os_name = parsed_ua['platform']['name']
    if not os_name and 'BB10' in ua_string:
        os_name = 'BlackBerry'
    if not os_name:
        os_name = 'unknown'   
    
    if os_name in ['iOS', 'Android', 'BlackBerry']:
        return 'mobile'
    elif os_name in ['Mac OS', 'Linux', 'PlayStation', 'Windows', 'ChromeOS', ' ChromeOS']:
        return 'non-mobile'
    return 'unknown' 


"""
Features:
Device, length, Channel
Depth
"""

print("Iterating through training data")
X_train = []; y_train_regress = []; y_train_classify = []
X_test = []; y_test_regress = []; y_test_classify = []
training_users = set()
training_pages = set()
for train_data in ttg.training_set:
    dwell, uid, url, depth, _, viewport, _, body_length, channel, _, user_agent = train_data
    
    device = get_device(user_agent)
    training_users.add(uid)
    training_pages.add(url)
    
    length_missing = 0
    length = 0
    if body_length == "unknown":
        length_missing = 1
        length = 0
    else:
        length = int(body_length)
    
#     X_train.append( {'device': device, 'length': length, 'channel': channel, 'depth':float(depth)} )
#     X_train.append( {'device': device, 'length': length, 'channel': channel, 'viewport1':int(viewport.split('x')[0]), 'viewport2':int(viewport.split('x')[1]), 'depth':float(depth)} )
#     X_train.append( {'device': device, 'length': length, 'channel': channel, 'viewport':viewport, 'depth':float(depth)} )
#     X_train.append( {'viewport':viewport, 'depth':float(depth)} )
    X_train.append( {'viewport1':int(viewport.split('x')[0]), 'viewport2':int(viewport.split('x')[1]), 'depth':float(depth)} )
    y_train_regress.append(float(dwell))
    y_train_classify.append(1 if dwell >= required_dwell_time else 0)



print("Iterating through test data")
for test_data in ttg.test_set:
    dwell, uid, url, depth, _, viewport, _, body_length, channel, _, user_agent = test_data
    
    if ( uid not in training_users or url not in training_pages ):
        continue
    
    device = get_device(user_agent)
    training_users.add(uid)
    training_pages.add(url)
    
    length_missing = 0
    length = 0
    if body_length == "unknown":
        length_missing = 1
        length = 0
    else:
        length = int(body_length)
    
#     X_test.append( {'device': device, 'length': length, 'channel': channel, 'depth':float(depth)} )
#     X_test.append( {'device': device, 'length': length, 'channel': channel, 'viewport1':int(viewport.split('x')[0]), 'viewport2':int(viewport.split('x')[1]), 'depth':float(depth)} )
#     X_test.append( {'device': device, 'length': length, 'channel': channel, 'viewport':viewport, 'depth':float(depth)} )
    X_test.append( {'viewport1':int(viewport.split('x')[0]), 'viewport2':int(viewport.split('x')[1]), 'depth':float(depth)} )
    y_test_regress.append(float(dwell))
    y_test_classify.append(1 if dwell >= required_dwell_time else 0)
    
    


vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print('X has been vectorized.')

# print("HAS VIEWPORT")
print('required_dwell_time =', required_dwell_time)
clf = LinearRegression()
clf.fit(X_train, y_train_regress)
y_pred_regress = clf.predict(X_test)
print('RMSD_beyondclick =', sqrt(mean_squared_error(y_test_regress, y_pred_regress)))

clf = LogisticRegression()
clf.fit(X_train, y_train_classify)
y_pred_classify = clf.predict_proba(X_test)
# print(y_test_classify)
# print(y_pred_classify)
print('Log-loss_beyondclick =', log_loss(y_test_classify, y_pred_classify))



# y_pred_classify = clf.predict_proba(X_train)
# print('Log-loss_beyondclick TRAINING =', log_loss(y_train_classify, y_pred_classify))
