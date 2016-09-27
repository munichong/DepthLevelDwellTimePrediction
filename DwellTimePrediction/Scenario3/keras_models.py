'''
Created on Aug 25, 2016

@author: munichong
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Input, Embedding, merge
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.engine.topology import Merge
from keras.models import Model
from keras.utils.visualize_util import plot


'''
activation: softmax, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear
'''


optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
# optimizer = Adam(lr=0.0001)

def RNN_simple(feat_num, timestep_num=100):
    model = Sequential()
    model.add(LSTM(input_shape=(timestep_num, feat_num), output_dim=256, activation='tanh', return_sequences=True))
    model.add(LSTM(output_dim=64, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=1, activation='linear'))) # sequence labeling
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_squared_error'])
    
    return model



def RNN_context_embed(ctx_feat_num, user_num, page_num, timestep_num=100):

#     user_input = Input(shape=(timestep_num,), name='user_input')
#     user_embed = Embedding(input_dim=user_num+1, output_dim=50, input_length=timestep_num,
#                             weights=None)(user_input)
    
#     page_input = Input(shape=(timestep_num,), name='page_input')
#     page_embed = Embedding(input_dim=page_num+1, output_dim=50, input_length=timestep_num,
#                             weights=None)(page_input)
    
    depth_input = Input(shape=(timestep_num,), name='dep_input')
    depth_embed = Embedding(input_dim=101, output_dim=200, input_length=timestep_num,
                            weights=None)(depth_input)
    
    
    context_input = Input(shape=(timestep_num, ctx_feat_num), name='ctx_input')
    
    
    merged_model = merge([depth_embed, context_input], mode='concat')
#     merged_model = merge([user_embed, page_embed, depth_embed, context_input], mode='concat')
    
    merged_model = LSTM(output_dim=300, activation='tanh',
                          return_sequences=True)(merged_model)
    
    merged_model = LSTM(output_dim=300, activation='tanh',
                          return_sequences=True)(merged_model)
                          
    merged_model = LSTM(output_dim=50, activation='tanh',
                          return_sequences=True)(merged_model)
    
    merged_model = TimeDistributed(Dense(output_dim=1, activation='linear'))(merged_model)
    
    model = Model(input=[depth_input, context_input], output=[merged_model])
    
    model.compile(loss='mean_squared_error',
                         optimizer=optimizer,
                         metrics=['mean_squared_error'])
    
    plot(model, to_file='model.png')
    
    return model

    """
#     print(timestep_num, feat_num)

    model_dep = Sequential()
    ''' The Embedding layer will take as input an integer matrix of size (batch, input_length) 
        The input array is only 2D !!! '''
    model_dep.add(Embedding(input_dim=101, output_dim=50, input_length=timestep_num,
                            weights=None))

    



#     model_context = Sequential()
#     model.add(BatchNormalization(input_shape=(timestep_num, feat_num)))

    model_context = Input(shape=(timestep_num, ctx_feat_num))
    
    
#     model.add(BatchNormalization())  

    
    
    merged_model = Sequential()
    merged_model.add(merge([model_dep, model_context], mode='concat'))
    
    merged_model.add(LSTM(output_dim=256, activation='tanh',
                          return_sequences=True))
    
    merged_model.add(LSTM(output_dim=64, activation='tanh', return_sequences=True))
#     model.add(BatchNormalization())
    
#     model.add(Dropout(0.2))
    merged_model.add(TimeDistributed(Dense(output_dim=1, activation='linear'))) # sequence labeling
#     model.add(TimeDistributed(Dense(output_dim=1, activation='relu'))) # sequence labeling
#     model.add(LeakyReLU(alpha=0.01))
    
    
    merged_model.compile(loss='mean_squared_error',
                         optimizer=optimizer,
                         metrics=['mean_squared_error'])
    
    return merged_model

    """

# RNN_context_depth(300)

# model.fit(X_train, y_train,
#           nb_epoch=20,
#           batch_size=128)
# score = model.evaluate(X_test, y_test, batch_size=16)

def linearRegression_keras(feat_num, timestep_num=100):
    model = Sequential()

    model.add(BatchNormalization(input_shape=(timestep_num, feat_num)))
    
    model.add(TimeDistributed(Dense(output_dim=1, activation='linear'), input_shape=(timestep_num, feat_num))) # sequence labeling
    

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_squared_error'])
    
    return model
