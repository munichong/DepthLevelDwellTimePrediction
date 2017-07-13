'''
Created on Aug 27, 2016

@author: munichong
'''
import pandas as pd, hashlib, sys
from random import random

import theano

print(theano.__version__)

import pydot_ng

def hashstr(str, nr_bin):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bin-1)+1

a = hashstr("asd", 1e+6)
print(a, sys.getsizeof(a))
a = hashstr("asd", 1e+3)
print(a, sys.getsizeof(a))


