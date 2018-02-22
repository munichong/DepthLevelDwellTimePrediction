'''
Created on Jun 2, 2017

@author: Wang
'''
import operator
from pymongo import MongoClient
from _collections import defaultdict
from LibFM import Training_test_generator as ttg
from math import sqrt
from numpy import mean, median, array
from sklearn.metrics import mean_squared_error, log_loss
from gensim.models import Doc2Vec, ldamodel
from gensim import corpora
from sklearn.preprocessing import normalize
import scipy as sp



client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']

