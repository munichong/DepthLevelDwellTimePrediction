'''
Created on Jun 2, 2017

@author: Wang
'''
from collections import defaultdict

fields = ['user', 'page', 'depth', 'channel', 'viewport', 
            'fresh', 'geo', 'hour', 'length', 'weekday', 'keyword',
            'topic_10', 'topic_20', 'topic_30', 'topic_40', 
            'topic_group_10', 'topic_group_20', 'topic_group_30', 'topic_group_40',
            'doc2vec_50', 'doc2vec_150']
start_indice = [0] * len(fields)
feat_idx_dict = [defaultdict(str) for _ in range(len(fields))]


MIN_DWELL_TIME = "dwell"
root = 'I:/Desktop/Desktop/libfm-1.42.src/data_bs_' + str(MIN_DWELL_TIME) + '/'

outfile_train = open('I:/Desktop/FFMInput_' + str(MIN_DWELL_TIME) + 's_train.csv', 'w')
outfile_test = open('I:/Desktop/FFMInput_' + str(MIN_DWELL_TIME) + 's_test.csv', 'w')

''' calculate the start_indice for each field '''
for f_index in range(len(fields) - 1):
    max_index = 0
    for line in open(root + fields[f_index] + '.libfm'):
        line = line.strip()
        for cell in line.split(' ')[1:]:
            i = int(cell.split(':')[0])
            max_index = max(max_index, i)
    start_indice[f_index + 1] = max_index + 1 + start_indice[f_index]
print('start_indice:', start_indice)


''' build feature index dictionaries from.libfm files '''
for f_index in range(len(fields)):
    start_index = start_indice[f_index]
    line_index = 0
    for line in open(root + fields[f_index] + '.libfm'):
        line = line.strip()
        cells = line.split(' ')[1:]
        new_cells = []
        for cell in cells:
            i = start_index + int(cell.split(':')[0])
            new_cells.append(str(f_index) + ':' + str(i) + ':' + cell.split(':')[1])
        new_cells = ' '.join(new_cells)
        feat_idx_dict[f_index][line_index] = new_cells
        line_index += 1
# print(feat_idx_dict[2])
# print(feat_idx_dict[-1])


training_data = []
for line in open(root + 'y.train', 'r'):
    training_data.append(line.strip())
for f_index in range(len(features)):
    i = 0
    for line in open(root + features[f_index] + '.train'):
#         if int(line.strip()) not in feat_idx_dict[f_index]:
#             print("New Feature!!!")
        training_data[i] = training_data[i] + ' ' + feat_idx_dict[f_index][int(line.strip())]
        i += 1
for instance in training_data:
    outfile_train.write(instance + '\n')


test_data = []
for line in open(root + 'y.test', 'r'):
    test_data.append(line.strip())
for f_index in range(len(features)):
    i = 0
    for line in open(root + features[f_index] + '.test'):
        test_data[i] = test_data[i] + ' ' + feat_idx_dict[f_index][int(line.strip())]
        i += 1
for instance in test_data:
    outfile_test.write(instance + '\n')


    