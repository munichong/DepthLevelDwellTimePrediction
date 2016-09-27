'''
Created on Aug 25, 2016

@author: munichong
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization

'''
activation: softmax, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear
'''

def RNN_keras(max_timestep_len, feat_num):
#     print(max_timestep_len, feat_num)
    model = Sequential()
    model.add(Masking(mask_value=-1.0, input_shape=(max_timestep_len, feat_num)))
#     model.add(BatchNormalization())
    model.add(LSTM(input_dim=feat_num, output_dim=128, activation='relu', return_sequences=True))  
#     model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(output_dim = 1, activation='relu'))) # sequence labeling
#     model.add(Dense(output_dim = 1, activation='relu'))


    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['mean_squared_error'])
    return model

# model.fit(X_train, y_train,
#           nb_epoch=20,
#           batch_size=128)
# score = model.evaluate(X_test, y_test, batch_size=16)


