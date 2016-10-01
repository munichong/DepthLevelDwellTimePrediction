'''
Created on Aug 28, 2016

@author: munichong
'''
import sys, math
import numpy as np
import model_input_gen as mig
from sklearn.linear_model import SGDRegressor
from keras_models import RNN_simple, RNN_context_embed, linearRegression_keras
from keras.models import load_model


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
        
        

class RMSD_batch:
    def __init__(self):
        self.total_numerator = 0
        self.total_denominator = 0
    
    def update(self, y_batch_true, y_batch_pred):
        for y_true, y_pred in zip(y_batch_true, y_batch_pred):
            self.total_numerator += ((np.array([y[0] for y in y_pred]) - 
                                      np.array([y[0] for y in y_true])
                                      ) ** 2).sum()
            
            self.total_denominator += len(y_true)

#             if self.total_numerator > 1000000000:
#                 output_true = open('y_batch_true.csv', 'w')
#                 for y_t in [y[0] for y in y_true]:
#                     output_true.write(str(y_t))
#                     output_true.write('\n')
#                     
#                 output_pred = open('y_batch_pred.csv', 'w')
#                 for y_t in [y[0] for y in y_pred]:
#                     output_pred.write(str(y_t))
#                     output_pred.write('\n')
                
    def final_RMSD(self):
        return math.sqrt(self.total_numerator / self.total_denominator)


def merge_Xs(*Xs):
    return np.dstack(Xs)

# def rnn_dict_input():

# BATCH_SIZE = len(mig.X_train)
BATCH_SIZE = 512
NUM_EPOCH = 20

num_batch = math.ceil( len(mig.X_train) / BATCH_SIZE )
best_epoch_lr = 0
best_epoch_rnn = 0

# sgdRegressor = SGDRegressor()

print('\nBuild model...')
# rnn = RNN_simple(len(mig.vectorizer.feature_names_))
rnn = RNN_context_embed(len(mig.vectorizer.feature_names_), mig.unique_users_num, mig.unique_pages_num)

lr = linearRegression_keras(len(mig.vectorizer.feature_names_) + 1)

globalAverage = GlobalAverage()


# _array = []
train_error_history = []
val_error_history = []
test_error_history = []
print("\n****** Iterating over each batch of the training data ******")
for epoch in range(1, NUM_EPOCH+1):
    batch_index = 0
    
    for X_batch_ctx, X_batch_dep, y_batch in mig.Xy_gen(mig.X_train, mig.y_train, batch_size=BATCH_SIZE):
        batch_index += 1
        
    #     sgdRegressor.partial_fit(X_batch, y_batch)
        if epoch == 1:
            ''' Baseline - GlobalAverage '''
            globalAverage.update(y_batch)
            
#         print(X_batch_dep.shape)
#         print(X_batch_ctx.shape)
        
        ''' Baseline - Linear Regression '''
        loss_lr = lr.train_on_batch(merge_Xs(X_batch_ctx, X_batch_dep), y_batch)
    
        ''' RNN '''
        loss_rnn = rnn.train_on_batch({'dep_input':X_batch_dep, 'ctx_input':X_batch_ctx}, y_batch)
#         print(loss.history)
#         print(rnn.metrics_names)
        print("Epoch %d/%d : Batch %d/%d | %s = %f | root_%s = %f" %
              (epoch, NUM_EPOCH, batch_index, num_batch, 
               rnn.metrics_names[0], loss_rnn[0], rnn.metrics_names[1], np.sqrt(loss_rnn[1])))
    
    
    ''' 
    Use the RNN trained in the epoch to predict and compute the training error of this epoch 
    '''    
    rmsd_training = RMSD_batch()
    
    for X_batch_ctx, X_batch_dep, y_batch in mig.Xy_gen(mig.X_train, mig.y_train, batch_size=BATCH_SIZE):  
        prediction_batch = rnn.predict_on_batch({'dep_input':X_batch_dep, 'ctx_input':X_batch_ctx})
                
        if np.count_nonzero(prediction_batch) == 0:
            print("All zero")
        elif (prediction_batch < 0).any():
            print("Has negative")
            print("Converted negatives to zeros")
            prediction_batch[prediction_batch < 0] = 0
        else:
            print("The prediction contains", np.count_nonzero(prediction_batch),"non-zeros")
            print(prediction_batch)
        
        rmsd_training.update(y_batch, prediction_batch)

    train_error_history.append(rmsd_training.final_RMSD())
    
    
    
    '''
    Use the RNN trained in the epoch to predict and calculate the validation error of this epoch
    '''
    rmsd_globalAvg_val = RMSD_batch()
    rmsd_lr_val = RMSD_batch()
    rmsd_rnn_val = RMSD_batch()
    for X_batch_ctx, X_batch_dep, y_batch in mig.Xy_gen(mig.X_val, mig.y_val, batch_size=BATCH_SIZE):
        rmsd_globalAvg_val.update( y_batch, globalAverage.predict(BATCH_SIZE) )
        rmsd_lr_val.update(y_batch, lr.predict_on_batch(merge_Xs(X_batch_ctx, X_batch_dep)))
        rmsd_rnn_val.update( y_batch, rnn.predict_on_batch({'dep_input':X_batch_dep, 'ctx_input':X_batch_ctx}) )
        
    val_error_history.append((rmsd_globalAvg_val.final_RMSD(), 
                               rmsd_lr_val.final_RMSD(), 
                               rmsd_rnn_val.final_RMSD()))
    
    
    print()
    print("****** Epoch %d: RMSD(training) = %f \n" % (epoch, rmsd_training.final_RMSD()))
    print()    
    print("================= Performance on the Validation Set =======================")
    print("****** Epoch %d: GlobalAverage: RMSD = %f" % (epoch, rmsd_globalAvg_val.final_RMSD()))
    print("****** Epoch %d: Linear Regression: RMSD = %f" % (epoch, rmsd_lr_val.final_RMSD()))
    print("****** Epoch %d: RNN: RMSD = %f" % (epoch, rmsd_rnn_val.final_RMSD()))
    print("===========================================================================")
    print()
    
    epoch = 0
    print("The Performance of All Epochs So Far:")
    for train_error_rnn, val_error_all in zip(train_error_history, val_error_history):
        epoch += 1
        print("============ Epoch %d =============" % epoch)
        print("The training error of RNN = %f" % train_error_rnn)
        print()
        print("The validation error of GlobalAverage = %f" % val_error_all[0])
        print("The validation error of Linear Regression = %f" % val_error_all[1])
        print("The validation error of RNN = %f" % val_error_all[2])
        print()
    
    
    
    ''' Check if the validation error of this epoch is the lowest so far '''
    if rmsd_lr_val.final_RMSD() == min(lr_val for _, lr_val, _ in val_error_history):
        lr.save('lr.h5')
        best_epoch_lr = epoch
        print("A LR model has been saved.")
    
    ''' Check if the validation error of this epoch is the lowest so far '''
    if rmsd_rnn_val.final_RMSD() == min(rnn_val for _, _, rnn_val in val_error_history):
        rnn.save('rnn.h5')
        best_epoch_rnn = epoch
        print("An RNN model has been saved.")
    print()
    print()
    
    

# Just print out
print("GlobalAverage:")
print(globalAverage.predict(1)[0])
print()

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


print("LR got the best performance at Epoch %d" % best_epoch_lr)
print("RNN got the best performance at Epoch %d" % best_epoch_rnn)

    
rnn_best = load_model('rnn.h5')
lr_best = load_model('lr.h5')   

'''
Load the RNN and LR have the lowest validation error
Predict and calculate the test error of this epoch
'''
rmsd_globalAvg_test = RMSD_batch()
rmsd_lr_test = RMSD_batch()
rmsd_rnn_test = RMSD_batch()
for X_batch_ctx, X_batch_dep, y_batch in mig.Xy_gen(mig.X_test, mig.y_test, batch_size=BATCH_SIZE):
    rmsd_globalAvg_test.update( y_batch, globalAverage.predict(BATCH_SIZE) )
    rmsd_lr_test.update(y_batch, lr_best.predict_on_batch(merge_Xs(X_batch_ctx, X_batch_dep)))
    rmsd_rnn_test.update( y_batch, rnn_best.predict_on_batch({'dep_input':X_batch_dep, 'ctx_input':X_batch_ctx}) )

print()
print("================= Performance on the Test Set =======================")
print("GlobalAverage: RMSD = %f" % (rmsd_globalAvg_test.final_RMSD()))
print("Linear Regression (Epoch=%d): RMSD = %f" % (best_epoch_lr, rmsd_lr_test.final_RMSD()))
print("RNN (Epoch=%d): RMSD = %f" % (best_epoch_rnn, rmsd_rnn_test.final_RMSD()))
print("=====================================================================")

