'''
Created on Jul 25, 2017

@author: Wang
'''
import matplotlib.pyplot as plt

def read_result(path):
    train_errors = []
    valid_errors =[]
    valid_best_epoch = 0
    test_errors = []
    with open(path) as result_file:
        for line in result_file:
            line = line.strip()
            if 'The training error of RNN' in line:
                ll_train = float(line.split('ll=')[1][:6])
                train_errors.append(ll_train)
            elif 'The validation error of RNN' in line:
                ll_valid = float(line.split('ll=')[1][:6])
                valid_errors.append(ll_valid)
            elif 'The test error of RNN' in line:
                ll_test = float(line.split('ll=')[1][:6])
                test_errors.append(ll_test)
            elif 'RNN got the best performance at Epoch' in line:
                valid_best_epoch = int(line[-2:])
            
    return train_errors, valid_errors, valid_best_epoch, test_errors

training_errors1, validation_errors1, validation_best_epoch1, test_errors1 = read_result('I:/Desktop/userpagedepth_regular0.001.txt')
# training_errors2, validation_errors2, validation_best_epoch2, test_errors2 = read_result('I:/Desktop/userpage_regular0.001 (2).txt')

test_errors1 = test_errors1[:-1]
# test_errors2 = test_errors2[:-1]

validation_best_epoch1 = 17

print(training_errors1)
print(validation_errors1)
print(validation_best_epoch1)
print(test_errors1)

print()

# print(training_errors2)
# print(validation_errors2)
# print(validation_best_epoch2)
# print(test_errors2)


def plot_figure(training_errors, validation_errors, validation_best_epoch, test_errors, marker, linestyle, markersize=18, linewidth=1):
    training, = plt.plot(range(1, len(training_errors) + 1), training_errors,'r', 
                         marker=marker, linestyle=linestyle, markersize=markersize, linewidth=linewidth,
#                          markerfacecolor='none', 
                         markeredgecolor='r')
    validation, = plt.plot(range(1, len(validation_errors) + 1), validation_errors,'b', 
                           marker=marker, linestyle=linestyle, markersize=markersize, linewidth=linewidth, 
#                            markerfacecolor='none', 
                           markeredgecolor='b')
    test, = plt.plot(range(1, len(test_errors) + 1), test_errors,'g', 
                           marker=marker, linestyle=linestyle, markersize=markersize, linewidth=linewidth, 
#                            markerfacecolor='none', 
                           markeredgecolor='g')
    plt.plot([validation_best_epoch], [min(validation_errors)], 'magenta', 
             marker=marker, linestyle=linestyle, markersize=markersize)
    plt.annotate(min(validation_errors), (validation_best_epoch, min(validation_errors)), color='magenta', fontsize=18)
    
    print(min(validation_errors))
    
    plt.plot([test_errors.index(min(test_errors)) + 1], [min(test_errors)], 'magenta', 
             marker=marker, linestyle=linestyle, markersize=markersize)
    plt.annotate(min(test_errors), (test_errors.index(min(test_errors)) + 1, min(test_errors)), color='magenta', fontsize=18)
    
    print(min(test_errors))
    
    for i in range(len(training_errors)):
        plt.annotate(training_errors[i], (i+1, training_errors[i]), color='r', fontsize=16)
    for i in range(len(validation_errors)):
        plt.annotate(validation_errors[i], (i+1, validation_errors[i]), color='b', fontsize=16)
    for i in range(len(test_errors)):
        plt.annotate(test_errors[i], (i+1, test_errors[i]), color='g', fontsize=16)
    
    return training, validation, test

training1, validation1, test1 = plot_figure(training_errors1, validation_errors1, validation_best_epoch1, test_errors1, '.', '-')
# training2, validation2, test2 = plot_figure(training_errors2, validation_errors2, validation_best_epoch2, test_errors2, 's', ':', markersize=10)

plt.xlim(0, 35)
plt.ylim(0.58, 0.65)
plt.ylabel('Logloss', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.title('Test Error (1-layer; userpagedepth_regular0.001; lr=0.1*2|0.01*38) = ' + str(test_errors1[validation_best_epoch1-1]),
#         + '\nTest Error (userpage_regular0.001; lr=0.1*2|0.01*17) = ' + str(test_errors2[validation_best_epoch2-1]),
           fontsize=20)

plt.legend([
            training1, validation1, test1,
#             training2, validation2, test2
            ], 
           ['Training Errors', 'Validation Errors', 'Test Errors',
#             'Training Errors (2)', 'Validation Errors (2)', 'Test Errors (2)'
            ], 
           fontsize=16)

plt.show()




'''

import matplotlib.pyplot as plt

def read_result(path):
    train_errors = []
    valid_errors =[]
    valid_best_epoch = 0
    test_error = ''
    with open(path) as result_file:
        for line in result_file:
            line = line.strip()
            if 'The training error of RNN' in line:
                ll_train = float(line.split('ll=')[1][:6])
                train_errors.append(ll_train)
            elif 'The validation error of RNN' in line:
                ll_valid = float(line.split('ll=')[1][:6])
                valid_errors.append(ll_valid)
            elif 'The test error of RNN' in line:
                test_error = line.split('ll=')[1][:6]
            elif 'RNN got the best performance at Epoch' in line:
                valid_best_epoch = int(line[-2:])
    return train_errors, valid_errors, valid_best_epoch, test_error

# training_errors1, validation_errors1, validation_best_epoch1, test_error1 = read_result('I:/Desktop/0.1^1-0.01^1-0.001^2.txt')
# training_errors2, validation_errors2, validation_best_epoch2, test_error2 = read_result('I:/Desktop/0.1^2-0.001^3.txt')
# training_errors3, validation_errors3, validation_best_epoch3, test_error3 = read_result('I:/Desktop/0.1^2-0.0001^5.txt')
training_errors4, validation_errors4, validation_best_epoch4, test_error4 = read_result('I:/Desktop/0.1^1-0.01^2-0.001^5-0.0001^4.txt')
# training_errors5, validation_errors5, validation_best_epoch5, test_error5 = read_result('I:/Desktop/0.1^2-0.01^1.txt')
training_errors6, validation_errors6, validation_best_epoch6, test_error6 = read_result('I:/Desktop/0.01^3-0.001^9.txt')




# print(training_errors)
# print(validation_errors)
# print(validation_best_epoch)
# print(test_error)



def plot_figure(training_errors, validation_errors, validation_best_epoch, test_error, marker, linestyle, markersize=18, linewidth=1):
    training, = plt.plot(range(1, len(training_errors) + 1), training_errors,'r', 
                         marker=marker, linestyle=linestyle, markersize=markersize, linewidth=linewidth,
#                          markerfacecolor='none', 
                         markeredgecolor='r')
    validation, = plt.plot(range(1, len(validation_errors) + 1), validation_errors,'b', 
                           marker=marker, linestyle=linestyle, markersize=markersize, linewidth=linewidth, 
#                            markerfacecolor='none', 
                           markeredgecolor='b')
    plt.plot([validation_best_epoch], [min(validation_errors)], 'magenta', 
             marker=marker, linestyle=linestyle, markersize=markersize)
    for i in range(len(training_errors)):
        plt.annotate(training_errors[i], (i+1, training_errors[i]), color='r', fontsize=14)
    for i in range(len(validation_errors)):
        plt.annotate(validation_errors[i], (i+1, validation_errors[i]), color='b', fontsize=14)
    plt.annotate(min(validation_errors), (validation_best_epoch, min(validation_errors)), color='magenta', fontsize=16)
    return training, validation

# training1, validation1 = plot_figure(training_errors1, validation_errors1, validation_best_epoch1, test_error1, '.', '-')
# training2, validation2 = plot_figure(training_errors2, validation_errors2, validation_best_epoch2, test_error2, 's', '--', markersize=10)
# training3, validation3 = plot_figure(training_errors3, validation_errors3, validation_best_epoch3, test_error3, 'x', '-')
training4, validation4 = plot_figure(training_errors4, validation_errors4, validation_best_epoch4, test_error4, '^', '-', markersize=10)
# training5, validation5 = plot_figure(training_errors5, validation_errors5, validation_best_epoch5, test_error5, 'd', ':', markersize=10)
training6, validation6 = plot_figure(training_errors6, validation_errors6, validation_best_epoch6, test_error6, 's', ':', markersize=10)

plt.xlim(0, 13)
plt.ylim(0.56, 0.68)
plt.ylabel('Logloss', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
# plt.title('Test Error (lr=0.01, decay=1e-4) = ' + test_error1 + '\nTest Error (lr=0.01, decay=1e-6) = ' + test_error2, fontsize=25)
plt.title("Evaluation of Step Decay\n(SGD(lr=LR_RATES[0], decay=0, momentum=0.99, nesterov=True)\n[Results are good]")

plt.legend([
#             training1, validation1, training2, validation2,
#             training3, validation3,
            training4, validation4,
#             training5, validation5, 
            training6, validation6
            ], 
           [
#             'Training Errors (lr=0.1*1-0.01*1-0.001*2)', 'Validation Errors (lr=0.1*1-0.01*1-0.001*2)',
#             'Training Errors (lr=0.1*2-0.001*3)', 'Validation Errors (lr=0.1*2-0.001*3)',
#             'Training Errors (lr=0.1*2-0.0001*5)', 'Validation Errors (lr=0.1*2-0.0001*5)',
            'Training Errors (lr=0.1*1-0.01*2-0.001*5-0.0001*4)', 'Validation Errors (lr=0.1*1-0.01*2-0.001*5-0.0001*4)',
#             'Training Errors (lr=0.1*2-0.01*1)', 'Validation Errors (lr=0.1*2-0.01*1)',
            'Training Errors (lr=0.01*3-0.001*9)', 'Validation Errors (lr=0.01*3-0.001*9)',
            ], 
           fontsize=16)

plt.show()

'''