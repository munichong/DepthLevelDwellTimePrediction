'''
Created on Jun 24, 2017

@author: Wang
'''
TASK = 'c'
VIEWABILITY_THRESHOLD = 1


STEP_DECAY = True
'''
LR_RATES[i]: The learning rate that is set at the beginning of the $i$th Epoch
'''
LR_RATES = [0.1] * 2
LR_RATES = [0.01] * 30
LR_RATES += [0.001] * 10
LR_RATES += [0.0001] * 100

