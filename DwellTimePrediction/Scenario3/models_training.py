'''
Created on Aug 28, 2016

@author: munichong
'''
import sys, math, time, pickle
import numpy as np
from keras import backend as K
from math import sqrt
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

start_time = time.time()

import model_input_gen as mig # This will run other modules
from keras_models import FNN_onestep_r, FNN_onestep_c, RNN_upc_embed_r, RNN_upc_embed_c, linearRegression_keras, logisticRegression_keras
from keras.models import load_model
from prespecified_parameters import TASK, VIEWABILITY_THRESHOLD, STEP_DECAY, LR_RATES

# BATCH_SIZE = len(mig.X_train)
BATCH_SIZE = 128
NUM_EPOCH = 12





class GlobalAverage:
    def __init__(self):
        self.training_pageviews_num = 0
        self.accumulator = np.array([0.0] * 100)
    
    def update(self, y_batch):
        for y100 in y_batch:
            self.accumulator += np.array([dw[0] for dw in y100]) 
            self.training_pageviews_num += 1
    
    def predict(self, batch_size):
        # The prediction array should have the same shape as the y input of Keras RNN 
        return [np.array([[y_hat] for y_hat in self.accumulator / self.training_pageviews_num])
                for _ in range(batch_size)]
        
        

class regression_metric_batch:
    def __init__(self):
        self.total_numerator = 0
        self.total_denominator = 0
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_batch_true, y_batch_pred):
        for y_true, y_pred in zip(y_batch_true, y_batch_pred):
            self.total_numerator += ((np.array([y[0] for y in y_pred]) - 
                                          np.array([y[0] for y in y_true])
                                          ) ** 2).sum()
                
            self.total_denominator += len(y_true)
            self.y_true.extend([y[0] for y in y_true])
            self.y_pred.extend([y[0] for y in y_pred])
    
    def RMSD(self):
        return math.sqrt(self.total_numerator / self.total_denominator)
                
    def result(self):
        return "%.4f" % self.RMSD()


class classification_metric_batch:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_batch_true, y_batch_pred):
        for y_true, y_pred in zip(y_batch_true, y_batch_pred):  
            self.y_true.extend([y[0] for y in y_true])
            self.y_pred.extend([y[0] for y in y_pred])
    
    def LogLoss(self):
        return log_loss(self.y_true, self.y_pred)
    
    def result(self):
        ll = log_loss(self.y_true, self.y_pred)
        acc = accuracy_score(self.y_true, np.where(np.array(self.y_pred)>=0.5, 1, 0))
        auc = roc_auc_score(self.y_true, self.y_pred)
        

        
        return "(ll=%.4f, acc=%.4f, auc=%.4f; 1-25%%=%s, 26-75%%=%s, 76-100%%=%s" % (ll, acc, auc, 
                                                                                     self.bucket_results(0, 24), 
                                                                                     self.bucket_results(25, 74), 
                                                                                     self.bucket_results(75, 99))

    def bucket_results(self, start, end):
        '''
        start=0, end=24
        start=25, end=74
        start=75, end=99
        '''
        y_true_bucket = []
        y_pred_bucket = []
        for i in range(len(self.y_true)):
            if i % 100 >= start and i % 100 <= end:
                y_true_bucket.append(self.y_true[i])
                y_pred_bucket.append(self.y_pred[i])
        return "(ll=%.4f, acc=%.4f, auc=%.4f)" % (log_loss(y_true_bucket, y_pred_bucket), 
                                                  accuracy_score(y_true_bucket, np.where(np.array(y_pred_bucket)>=0.5, 1, 0)), 
                                                  roc_auc_score(y_true_bucket, y_pred_bucket))

def merge_Xs(*Xs):
    return np.dstack(Xs)

class pooling_filter():
    def __init__(self):
        self.functions = [np.min, self.quartile25, np.mean, np.median, self.quartile75, np.max]
                
    def quartile25(self, y):
        return np.percentile(y, 25)
    
    def quartile75(self, y):
        return np.percentile(y, 75)
    
    def train_filter(self, origin_pred, ground_truth, origin_result):
        ''' refresh these three every time when better validation error is found '''
        self.best_func = None
        self.best_interval = None
        self.best_stride = None
        
        best_result = origin_result
        for calculate_newVal in self.functions:
            for d in range(2, 10):
                pv_num = 0
                filtered_pred = []
                while pv_num < len(ground_truth) / 100:
                    start_index = pv_num * 100
                    pv_num += 1
                    end_index = pv_num * 100
                    group = 0
                    while group * d < 100:
                        l = origin_pred[start_index: end_index][group * d: group * d + d]    
                        new_val = calculate_newVal(l)
                        filtered_pred.extend([new_val] * len(l))
                        group += 1
                
                if TASK == 'c':
                    res = round(log_loss(ground_truth, filtered_pred), 4)
                    print("Filtered Logloss (", calculate_newVal, "d =", d, "):", res)
                elif TASK == 'r':
                    res = round(sqrt(mean_squared_error(ground_truth, filtered_pred)), 4)
                    print("Filtered RMSD (", calculate_newVal, "d =", d, "):", res)
                
                if res < best_result:
                    best_result = res
                    self.best_func = calculate_newVal
                    self.best_interval = d
        print("\nThe best result is", best_result, "\nThe best function is", self.best_func, "\nThe best interval is", self.best_interval)
                    
    def predict_filter(self, origin_pred, ground_truth, origin_result):            
        if not self.best_func:
            print("No saved filter !!!")
            return
            
        d = self.best_interval
        calculate_newVal = self.best_func
        
        pv_num = 0
        filtered_pred = []
        while pv_num < len(ground_truth) / 100:
            start_index = pv_num * 100
            pv_num += 1
            end_index = pv_num * 100
            group = 0
            while group * self.best_interval < 100:
                l = origin_pred[start_index: end_index][group * d: group * d + d]    
                new_val = calculate_newVal(l)
                filtered_pred.extend([new_val] * len(l))
                group += 1
        
        res = None
        if TASK == 'c':
#             res = round(log_loss(ground_truth, filtered_pred), 4)
            print("Filtered Logloss (", calculate_newVal, "d =", d, "):")
            res = classification_metric_batch()
        elif TASK == 'r':
#             res = round(sqrt(mean_squared_error(ground_truth, filtered_pred)), 4)
            print("Filtered RMSD (", calculate_newVal, "d =", d, "):")
            res = regression_metric_batch()
        res.y_true = ground_truth
        res.y_pred = filtered_pred
        print(res.result())
            
            
def make_prediction(X, y):
    if TASK == 'r':
        globalAvg_err = regression_metric_batch()
        lr_err = regression_metric_batch()
        rnn_err = regression_metric_batch()
    elif TASK == 'c':
        globalAvg_err = classification_metric_batch()
        lr_err = classification_metric_batch()
        rnn_err = classification_metric_batch()
  
    for X_batch_ctx, X_batch_dep, X_batch_u, X_batch_p, y_batch in mig.Xy_gen(X, y, batch_size=BATCH_SIZE):
        globalAvg_err.update( y_batch, globalAverage.predict(BATCH_SIZE) )
        lr_err.update(y_batch, lr.predict_on_batch(merge_Xs(X_batch_ctx, X_batch_dep)))
        rnn_err.update( y_batch, rnn.predict_on_batch({'dep_input':X_batch_dep,
                                                            'ctx_input':X_batch_ctx,
                                                            'user_input': X_batch_u,
                                                            'page_input': X_batch_p}) )
    return globalAvg_err, lr_err, rnn_err            
            
            
            
            

if __name__ == "__main__":
    
    num_batch = math.ceil( len(mig.X_train) / BATCH_SIZE )
    best_epoch_lr = 0
    best_epoch_rnn = 0
    
    # sgdRegressor = SGDRegressor()
    
    print('\nBuild model...')
    # rnn = RNN_simple(len(mig.vectorizer.feat_dict))
    if TASK == 'r':
        rnn = RNN_upc_embed_r(len(mig.vectorizer.feat_dict), mig.unique_users_num, mig.unique_pages_num)
#         rnn = FNN_onestep_r(len(mig.vectorizer.feat_dict), mig.unique_users_num, mig.unique_pages_num)
    elif TASK =='c':
        rnn = RNN_upc_embed_c(len(mig.vectorizer.feat_dict), mig.unique_users_num, mig.unique_pages_num)
#         rnn = FNN_onestep_c(len(mig.vectorizer.feat_dict), mig.unique_users_num, mig.unique_pages_num)
    
    if TASK == 'r':
        lr = linearRegression_keras(len(mig.vectorizer.feat_dict) + 1)
    elif TASK == 'c':
        lr = logisticRegression_keras(len(mig.vectorizer.feat_dict) + 1)
    
    
    globalAverage = GlobalAverage()
    
    post_filter = pooling_filter()
    
    # _array = []
    train_error_history = []
    val_error_history = []
    test_error_history = []
    learning_rate_history = [(1, LR_RATES[0])]
    print("\n****** Iterating over each batch of the training data ******")
    for epoch in range(1, NUM_EPOCH+1):
        batch_index = 0
        
        for X_batch_ctx, X_batch_dep, X_batch_u, X_batch_p, y_batch in mig.Xy_gen(mig.X_train, mig.y_train, batch_size=BATCH_SIZE):
            batch_index += 1
            
        #     sgdRegressor.partial_fit(X_batch, y_batch)
            if epoch == 1:
                ''' Baseline - GlobalAverage '''
                globalAverage.update(y_batch)
                

            
            ''' Baseline - Regression '''
            loss_lr = lr.train_on_batch(merge_Xs(X_batch_ctx, X_batch_dep), y_batch)
        
            ''' RNN '''
#             print(X_batch_dep.shape, X_batch_ctx.shape, X_batch_u.shape, X_batch_p.shape)
#             print((len(mig.vectorizer.feat_dict), mig.unique_users_num, mig.unique_pages_num))
            loss_rnn = rnn.train_on_batch({'dep_input':X_batch_dep,
                                           'ctx_input':X_batch_ctx,
                                           'user_input': X_batch_u,
                                           'page_input': X_batch_p,
                                           }, y_batch)
    #         print(loss.history)
    #         print(rnn.metrics_names)
    #         print(loss_rnn)
            if TASK == 'r':
                print("Epoch %d/%d : Batch %d/%d | %s = %.4f | root_%s = %.4f" %
                      (epoch, NUM_EPOCH, batch_index, num_batch, 
                       rnn.metrics_names[0], loss_rnn[0], rnn.metrics_names[1], np.sqrt(loss_rnn[1])))
            elif TASK == 'c':
                print("Epoch %d/%d : Batch %d/%d | %s = %.4f | %s = %.4f | %s = %.4f" %
                      (epoch, NUM_EPOCH, batch_index, num_batch, 
                       rnn.metrics_names[0], loss_rnn[0], rnn.metrics_names[1], loss_rnn[1], rnn.metrics_names[2], loss_rnn[2]))
        
        
        
        ''' Tracking the learning rate '''
        lrate = rnn.optimizer.lr.get_value()
        lrate *= (1. / (1. + rnn.optimizer.decay.get_value() * rnn.optimizer.iterations.get_value()))
        print('The Learning rate at the end of epoch', epoch, ":", lrate)
        
        learning_rate_history.append((epoch+1, lrate))
        if STEP_DECAY:
            K.set_value(rnn.optimizer.lr, LR_RATES[epoch])
        
        
        
        
        ''' 
        Use the RNN trained in the epoch to predict and compute the training error of this epoch 
        '''    
        if TASK == 'r':
            rnn_train_err = regression_metric_batch()
        elif TASK == 'c':
            rnn_train_err = classification_metric_batch()
        
        for X_batch_ctx, X_batch_dep, X_batch_u, X_batch_p, y_batch in mig.Xy_gen(mig.X_train, mig.y_train, batch_size=BATCH_SIZE):  
            prediction_batch = rnn.predict_on_batch({'dep_input':X_batch_dep,
                                                     'ctx_input':X_batch_ctx,
                                                     'user_input': X_batch_u,
                                                     'page_input': X_batch_p})
                    
            if np.count_nonzero(prediction_batch) == 0:
                print("All zero")
            elif (prediction_batch < 0).any():
                print("Has negative")
                print("Converted negatives to zeros")
                prediction_batch[prediction_batch < 0] = 0
            else:
                print("The prediction contains", np.count_nonzero(prediction_batch),"non-zeros")
                print(prediction_batch)
            
            rnn_train_err.update(y_batch, prediction_batch)
    
        train_error_history.append(rnn_train_err.result())
        
        
        
        print()
        print("****** Epoch", epoch, ": training_error =", rnn_train_err.result(), '\n')
        
        
        '''
        Use the RNN trained in the epoch to predict and calculate the validation error of this epoch
        '''
        globalAvg_val_err, lr_val_err, rnn_val_err = make_prediction(mig.X_val, mig.y_val)
        val_error_history.append((globalAvg_val_err.result(), 
                                   lr_val_err.result(), 
                                   rnn_val_err.result()))
        
        
        
        
        print()    
        print("================= Performance on the Validation Set =======================")
        print("****** Epoch", epoch, ": The validation error of GlobalAverage =", globalAvg_val_err.result())
        print("****** Epoch", epoch, ": The validation error of Linear Regression =", lr_val_err.result())
        print("****** Epoch", epoch, ": The validation error of RNN =", rnn_val_err.result())
        print("===========================================================================")
        print()
        
        
        '''
        Use the RNN trained in the epoch to predict and calculate the test error of this epoch
        '''
        globalAvg_test_err, lr_test_err, rnn_test_err = make_prediction(mig.X_test, mig.y_test)
        test_error_history.append((globalAvg_test_err.result(),
                                   lr_test_err.result(),
                                   rnn_test_err.result()))
        
        print()
        print("================= Performance on the Test Set =======================")
        print("****** Epoch", epoch, ": The test error of GlobalAverage =", globalAvg_test_err.result())
        print("****** Epoch", epoch, ": The test error of Linear Regression =", lr_test_err.result())
        print("****** Epoch", epoch, ": The test error of RNN =", rnn_test_err.result())
        print("===========================================================================")
        print()
        
        
        
        
        ''' Check if the validation error of this epoch is the lowest so far '''
        if lr_val_err.result() == min(lr_val for _, lr_val, _ in val_error_history):
            lr.save('lr.h5')
            best_epoch_lr = epoch
            print("A LR model has been saved.")
            
        
        ''' Check if the validation error of this epoch is the lowest so far '''
        if rnn_val_err.result() == min(rnn_val for _, _, rnn_val in val_error_history):
            rnn.save('rnn.h5')
            best_epoch_rnn = epoch
            print("An RNN model has been saved.")
            
            ''' TRAIN POST-FILTER '''
            if TASK == 'c':
                post_filter.train_filter(rnn_val_err.y_pred, rnn_val_err.y_true, rnn_val_err.LogLoss())
            elif TASK == 'r':
                post_filter.train_filter(rnn_val_err.y_pred, rnn_val_err.y_true, rnn_val_err.RMSD())
                
        print()
                
        print("The best performance of RNN so far is", min(rnn_val for _, _, rnn_val in val_error_history))
        print()
        print()
        
        
        
        
        epoch = 0
        print("The Performance of All Epochs So Far:")
        for train_error_rnn, val_error_all, test_error_all in zip(train_error_history, val_error_history, test_error_history):
            epoch += 1
            print("============ Epoch %d =============" % epoch)
            print("The training error of RNN =", train_error_rnn)
            print()
            print("The validation error of GlobalAverage =", val_error_all[0])
            print("The validation error of Linear Regression", val_error_all[1])
            print("The validation error of RNN =", val_error_all[2])
            print()
            print("The test error of GlobalAverage =", test_error_all[0])
            print("The test error of Linear Regression", test_error_all[1])
            print("The test error of RNN =", test_error_all[2])
            print()
        
    
    print("LR got the best performance at Epoch", best_epoch_lr)
    print("RNN got the best performance at Epoch", best_epoch_rnn)
    
        
    print()
    print("LEARNING RATES:", learning_rate_history[:-1])
    print()    
        
        
        
    rnn_best = load_model('rnn.h5')
    lr_best = load_model('lr.h5')   
    
    '''
    Load the RNN and LR have the lowest validation error
    Predict and calculate the test error of this epoch
    '''
    
    if TASK == 'r':
        globalAvg_test_err = regression_metric_batch()
        lr_test_err = regression_metric_batch()
        rnn_test_err = regression_metric_batch()
    elif TASK == 'c':
        globalAvg_test_err = classification_metric_batch()
        lr_test_err = classification_metric_batch()
        rnn_test_err = classification_metric_batch()
    
    
#     X_test = pickle.load(open('X_test.p', 'rb'))
#     y_test = pickle.load(open('y_test.p', 'rb'))
        

    for X_batch_ctx, X_batch_dep, X_batch_u, X_batch_p, y_batch in mig.Xy_gen(mig.X_test, mig.y_test, batch_size=BATCH_SIZE):
        globalAvg_test_err.update( y_batch, globalAverage.predict(BATCH_SIZE) )
        lr_test_err.update(y_batch, lr_best.predict_on_batch(merge_Xs(X_batch_ctx, X_batch_dep)))
        rnn_test_err.update( y_batch, rnn_best.predict_on_batch({'dep_input': X_batch_dep, 
                                                                  'ctx_input': X_batch_ctx,
                                                                  'user_input': X_batch_u,
                                                                  'page_input': X_batch_p}) )
#     del X_test
#     del y_test
    
    
    print()
    print("================= Performance on the Test Set =======================")
    print("The test error of GlobalAverage =", globalAvg_test_err.result())
    print("The test error of LR =", lr_test_err.result())
    print("The test error of RNN =", rnn_test_err.result())
    print("=====================================================================")
    
    ''' APPLY SAVED POST-FILTERING '''
    if TASK == 'c':
#         print(rnn_test_err.y_pred)
#         print(rnn_test_err.y_true)
        post_filter.predict_filter(rnn_test_err.y_pred, rnn_test_err.y_true, rnn_test_err.LogLoss())
    elif TASK == 'r':
        post_filter.predict_filter(rnn_test_err.y_pred, rnn_test_err.y_true, rnn_test_err.RMSD())
    
    
    
    
    ''' Measure runtime '''
    print()
    runtime = time.time() - start_time
    print("--- TOTAL RUNTIME: %d hours %d minutes %d seconds ---" % (runtime // 3600 % 60, runtime // 60 % 60, runtime % 60))
    print("--- AVERAGE RUNTIME: %d hours %d minutes %d seconds ---" % (runtime // NUM_EPOCH // 3600 % 60, 
                                                      runtime // NUM_EPOCH // 60 % 60, 
                                                      runtime // NUM_EPOCH % 60))
    