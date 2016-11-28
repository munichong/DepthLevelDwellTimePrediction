'''
Created on Aug 28, 2016

@author: munichong
'''
import sys, math
import numpy as np
import model_input_gen as mig
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from keras_models import FNN_onestep_r, FNN_onestep_c, RNN_upc_embed_r, RNN_upc_embed_c, linearRegression_keras, logisticRegression_keras
from keras.models import load_model



TASK = 'c'
VIEWABILITY_THRESHOLD = 1
# BATCH_SIZE = len(mig.X_train)
BATCH_SIZE = 128
NUM_EPOCH = 20





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
    
    def update(self, y_batch_true, y_batch_pred):
        for y_true, y_pred in zip(y_batch_true, y_batch_pred):
            self.total_numerator += ((np.array([y[0] for y in y_pred]) - 
                                      np.array([y[0] for y in y_true])
                                      ) ** 2).sum()
            
            self.total_denominator += len(y_true)

                
    def result(self):
        return math.sqrt(self.total_numerator / self.total_denominator)


class classification_metric_batch:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_batch_true, y_batch_pred):
        for y_true, y_pred in zip(y_batch_true, y_batch_pred):
            self.y_true.extend([y[0] for y in y_true])
            self.y_pred.extend([y[0] for y in y_pred])
    
    def result(self):
        ll = log_loss(self.y_true, self.y_pred)
        auc = roc_auc_score(self.y_true, self.y_pred)
        acc = accuracy_score(self.y_true, np.where(np.array(self.y_pred)>=0.5, 1, 0))
        return (ll, auc, acc, 
                (self.bucket_results(0, 24),
                 self.bucket_results(25, 74),
                 self.bucket_results(75, 99)))

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
        return (log_loss(y_true_bucket, y_pred_bucket),
                roc_auc_score(y_true_bucket, y_pred_bucket),
                accuracy_score(y_true_bucket, np.where(np.array(y_pred_bucket)>=0.5, 1, 0))
                )

def merge_Xs(*Xs):
    return np.dstack(Xs)




if __name__ == "__main__":
    
    num_batch = math.ceil( len(mig.X_train) / BATCH_SIZE )
    best_epoch_lr = 0
    best_epoch_rnn = 0
    
    # sgdRegressor = SGDRegressor()
    
    print('\nBuild model...')
    # rnn = RNN_simple(len(mig.vectorizer.feature_names_))
    if TASK == 'r':
        rnn = RNN_upc_embed_r(len(mig.vectorizer.feature_names_), mig.unique_users_num, mig.unique_pages_num)
#         rnn = FNN_onestep_r(len(mig.vectorizer.feature_names_), mig.unique_users_num, mig.unique_pages_num)
    elif TASK =='c':
        rnn = RNN_upc_embed_c(len(mig.vectorizer.feature_names_), mig.unique_users_num, mig.unique_pages_num)
#         rnn = FNN_onestep_c(len(mig.vectorizer.feature_names_), mig.unique_users_num, mig.unique_pages_num)
    
    if TASK == 'r':
        lr = linearRegression_keras(len(mig.vectorizer.feature_names_) + 1)
    elif TASK == 'c':
        lr = logisticRegression_keras(len(mig.vectorizer.feature_names_) + 1)
    
    
    globalAverage = GlobalAverage()
    
    
    # _array = []
    train_error_history = []
    val_error_history = []
    test_error_history = []
    print("\n****** Iterating over each batch of the training data ******")
    for epoch in range(1, NUM_EPOCH+1):
        batch_index = 0
        
        for X_batch_ctx, X_batch_dep, X_batch_u, X_batch_p, y_batch in mig.Xy_gen(mig.X_train, mig.y_train, batch_size=BATCH_SIZE):
            batch_index += 1
            
        #     sgdRegressor.partial_fit(X_batch, y_batch)
            if epoch == 1:
                ''' Baseline - GlobalAverage '''
                globalAverage.update(y_batch)
                
    #         print(X_batch_dep.shape)
    #         print(X_batch_ctx.shape)
            
            ''' Baseline - Regression '''
            loss_lr = lr.train_on_batch(merge_Xs(X_batch_ctx, X_batch_dep), y_batch)
        
            ''' RNN '''
            loss_rnn = rnn.train_on_batch({'dep_input':X_batch_dep,
                                           'ctx_input':X_batch_ctx,
                                           'user_input': X_batch_u,
                                           'page_input': X_batch_p,
                                           },y_batch)
    #         print(loss.history)
    #         print(rnn.metrics_names)
    #         print(loss_rnn)
            if TASK == 'r':
                print("Epoch %d/%d : Batch %d/%d | %s = %f | root_%s = %f" %
                      (epoch, NUM_EPOCH, batch_index, num_batch, 
                       rnn.metrics_names[0], loss_rnn[0], rnn.metrics_names[1], np.sqrt(loss_rnn[1])))
            elif TASK == 'c':
                print("Epoch %d/%d : Batch %d/%d | %s = %f | %s = %f | %s = %f" %
                      (epoch, NUM_EPOCH, batch_index, num_batch, 
                       rnn.metrics_names[0], loss_rnn[0], rnn.metrics_names[1], loss_rnn[1], rnn.metrics_names[2], loss_rnn[2]))
        
        
        
        
        
        
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
        
        
        
        
        
        
        '''
        Use the RNN trained in the epoch to predict and calculate the validation error of this epoch
        '''
        if TASK == 'r':
            globalAvg_val_err = regression_metric_batch()
            lr_val_err = regression_metric_batch()
            rnn_val_err = regression_metric_batch()
        if TASK == 'c':
            globalAvg_val_err = classification_metric_batch()
            lr_val_err = classification_metric_batch()
            rnn_val_err = classification_metric_batch()
        
        for X_batch_ctx, X_batch_dep, X_batch_u, X_batch_p, y_batch in mig.Xy_gen(mig.X_val, mig.y_val, batch_size=BATCH_SIZE):
            globalAvg_val_err.update( y_batch, globalAverage.predict(BATCH_SIZE) )
            lr_val_err.update(y_batch, lr.predict_on_batch(merge_Xs(X_batch_ctx, X_batch_dep)))
            rnn_val_err.update( y_batch, rnn.predict_on_batch({'dep_input':X_batch_dep,
                                                                'ctx_input':X_batch_ctx,
                                                                'user_input': X_batch_u,
                                                                'page_input': X_batch_p}) )
            
        val_error_history.append((globalAvg_val_err.result(), 
                                   lr_val_err.result(), 
                                   rnn_val_err.result()))
        
        
        
        print()
        print("****** Epoch", epoch, ": training_error =", rnn_train_err.result(), '\n')
        print()    
        print("================= Performance on the Validation Set =======================")
        print("****** Epoch", epoch, ": The validation error of GlobalAverage =", globalAvg_val_err.result())
        print("****** Epoch", epoch, ": The validation error of Linear Regression =", lr_val_err.result())
        print("****** Epoch", epoch, ": The validation error of RNN =", rnn_val_err.result())
        print("===========================================================================")
        print()
        
        epoch = 0
        print("The Performance of All Epochs So Far:")
        for train_error_rnn, val_error_all in zip(train_error_history, val_error_history):
            epoch += 1
            print("============ Epoch %d =============" % epoch)
            print("The training error of RNN =", train_error_rnn)
            print()
            print("The validation error of GlobalAverage =", val_error_all[0])
            print("The validation error of Linear Regression", val_error_all[1])
            print("The validation error of RNN =", val_error_all[2])
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
        print()
                
        print("The best performance of RNN so far is", min(rnn_val for _, _, rnn_val in val_error_history))
        print()
        print()
        
        
    # Just print out
#    print("GlobalAverage:")
#    print(globalAverage.predict(1)[0])
#    print()
    
        
    # epoch = 0
    # for train_error_rnn, val_error_all in zip(train_error_history, val_error_history):
    #     epoch += 1
    #     print("============ Epoch %d =============" % epoch)
    #     print("The training error of RNN = %f" % train_error_rnn)
    #     print()
    #     print("The validation error of GlobalAverage = %f" % val_error_all[0])
    #     print("The validation error of Linear Regression = %f" % val_error_all[1])
    #     print("The validation error of RNN = %f" % val_error_all[2])
    #     print()
    
    
    print("LR got the best performance at Epoch", best_epoch_lr)
    print("RNN got the best performance at Epoch", best_epoch_rnn)
    
    """    
    rnn_best = load_model('rnn.h5')
    # lr_best = load_model('lr.h5')   
    
    '''
    Load the RNN and LR have the lowest validation error
    Predict and calculate the test error of this epoch
    '''
    globalAvg_test_err = RMSD_batch()
    rmsd_lr_test = RMSD_batch()
    rmsd_rnn_test = RMSD_batch()
    for X_batch_ctx, X_batch_dep, X_batch_u, X_batch_p, y_batch in mig.Xy_gen(mig.X_test, mig.y_test, batch_size=BATCH_SIZE):
        rmsd_globalAvg_test.update( y_batch, globalAverage.predict(BATCH_SIZE) )
    #     rmsd_lr_test.update(y_batch, lr_best.predict_on_batch(merge_Xs(X_batch_ctx, X_batch_dep)))
        rmsd_rnn_test.update( y_batch, rnn_best.predict_on_batch({'dep_input': X_batch_dep, 
                                                                  'ctx_input': X_batch_ctx,
                                                                  'user_input': X_batch_u,
                                                                  'page_input': X_batch_p}) )
    
    print()
    print("================= Performance on the Test Set =======================")
    print("GlobalAverage: RMSD = %f" % (rmsd_globalAvg_test.final_RMSD()))
    # print("Linear Regression (Epoch=%d): RMSD = %f" % (best_epoch_lr, rmsd_lr_test.final_RMSD()))
    print("RNN (Epoch=%d): RMSD = %f" % (best_epoch_rnn, rmsd_rnn_test.final_RMSD()))
    print("=====================================================================")
    """
