'''
Created on Jun 13, 2017

@author: Wang
'''
THRESHOLD = 'dwell'

root = 'I:/Desktop/Desktop/libfm-1.42.src/'

ground_truth = []
# with open(root + 'data_bs_' + THRESHOLD + '/y.test', 'r') as infile_true:
with open('I:/Desktop/ground_truth_dwell.csv', 'r') as infile_true:
    n = 0
    for line in infile_true:
        ground_truth.append(float(line))
        n+= 1
    print(n)
    
origin_pred = []
# with open(root + 'prediction_' + THRESHOLD + '.txt', 'r') as infile_pred:
with open('I:/Desktop/output_ffm_dwell', 'r') as infile_pred:
    n = 0
    for line in infile_pred:
        origin_pred.append(float(line))
        n+=1
    print(n) 
# print(ground_truth[:10])
# print(origin_pred[:10])
# print(len(ground_truth))
# print(len(origin_pred))

# ground_truth = ground_truth[:101]
# origin_pred = origin_pred[:101]
# print(ground_truth)
# print(origin_pred)




from numpy import mean, median, percentile
from math import sqrt
from sklearn.metrics import log_loss, mean_squared_error

def calculate_newVal(l):
#     return max(l)
    return percentile(l, 75)

if THRESHOLD == 'dwell':
    print("Original RMSD:", round(sqrt(mean_squared_error(ground_truth, origin_pred)), 4))
else:
    print("Original Logloss:", round(log_loss(ground_truth, origin_pred), 4))

    


# for d in [1,2,3,4,5,6,7,8,9,10]:
for d in range(1, 21):
    pv_num = 0
    filtered_pred = []
#     bucket1_pred = []
#     bucket1_true = []
#     bucket2_pred = []
#     bucket2_true = []
#     bucket3_pred = []
#     bucket3_true = []
#     bucket4_pred = []
#     bucket4_true = []
    while pv_num < len(ground_truth) / 101:
        start_index = pv_num * 101
        pv_num += 1
        end_index = pv_num * 101
        
        group = 0
#         percent = 0
        while group * d < 101:
#             print(group)
            l = origin_pred[start_index: end_index][group * d: group * d + d]
            new_val = calculate_newVal(l)
            filtered_pred.extend([new_val] * len(l))
            group += 1
#             for i in range(len(l)):
#                 if percent <= 25:
#                     bucket1_pred.append(new_val)
#                     bucket1_true.append(new_val)
#                 elif percent <= 50:
#                     bu
    
    if THRESHOLD == 'dwell':
        print("Filtered RMSD (d="+ str(d)  +"):", round(sqrt(mean_squared_error(ground_truth, filtered_pred)), 4))
    else:
        print("Filtered Logloss (d="+ str(d)  +"):", round(log_loss(ground_truth, filtered_pred), 4)) 
    
#     if d == 7:    
#         print("Bucket1:")
#         pv_num = 0
#         ground_truth1 = []
#         filtered_pred1 = []
#         while pv_num * 101 < n:
#             ground_truth1.extend(ground_truth[pv_num * 101+1: pv_num * 101 + 26])
#             filtered_pred1.extend(filtered_pred[pv_num * 101+1: pv_num * 101 + 26])
#             pv_num += 1
#         print(round(sqrt(mean_squared_error(ground_truth1, filtered_pred1)), 4))
#         
#         
#         print("Bucket2:")
#         pv_num = 0
#         ground_truth2 = []
#         filtered_pred2 = []
#         while pv_num * 101 < n:
#             ground_truth2.extend(ground_truth[pv_num * 101+26: pv_num * 101 + 51])
#             filtered_pred2.extend(filtered_pred[pv_num * 101+26: pv_num * 101 + 51])
#             pv_num += 1
#         print(round(sqrt(mean_squared_error(ground_truth2, filtered_pred2)), 4))
#         
#         print("Bucket3:")
#         pv_num = 0
#         ground_truth3 = []
#         filtered_pred3 = []
#         while pv_num * 101 < n:
#             ground_truth3.extend(ground_truth[pv_num * 101+51: pv_num * 101 + 76])
#             filtered_pred3.extend(filtered_pred[pv_num * 101+51: pv_num * 101 + 76])
#             pv_num += 1
#         print(round(sqrt(mean_squared_error(ground_truth3, filtered_pred3)), 4))
#         
#         
#         print("Bucket4:")
#         pv_num = 0
#         ground_truth4 = []
#         filtered_pred4 = []
#         while pv_num * 101 < n:
#             ground_truth4.extend(ground_truth[pv_num * 101+76: pv_num * 101 + 101])
#             filtered_pred4.extend(filtered_pred[pv_num * 101+76: pv_num * 101 + 101])
#             pv_num += 1
#         print(round(sqrt(mean_squared_error(ground_truth4, filtered_pred4)), 4))
