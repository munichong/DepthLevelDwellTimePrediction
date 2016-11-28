'''
Created on Aug 25, 2016

@author: munichong
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Input, Embedding, merge
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.engine.topology import Merge
from keras.models import Model
from keras.regularizers import l2, l1
# from keras.utils.visualize_util import plot


'''
activation: softmax, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear
'''



def RNN_simple_r(feat_num, timestep_num=100):
    optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.99, nesterov=True)
# optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
# optimizer = Adam(lr=0.0001)

    
    model = Sequential()
    model.add(LSTM(input_shape=(timestep_num, feat_num), output_dim=256, activation='tanh', return_sequences=True))
    model.add(LSTM(output_dim=64, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=1, activation='linear'))) # sequence labeling
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_squared_error'])
    
    return model


def FNN_onestep_r(ctx_feat_num, user_num, page_num, timestep_num=100):
    optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.99, nesterov=True)
# optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
# optimizer = Adam(lr=0.0001)

    user_input = Input(shape=(timestep_num,), name='user_input')
    user_embed = Embedding(input_dim=user_num+1, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2
                            )(user_input)
    
    page_input = Input(shape=(timestep_num,), name='page_input')
    page_embed = Embedding(input_dim=page_num+1, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2
                            )(page_input)               
    
    depth_input = Input(shape=(timestep_num,), name='dep_input')
    depth_embed = Embedding(input_dim=101, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2
                            )(depth_input)
    
        
    context_input = Input(shape=(timestep_num, ctx_feat_num), name='ctx_input')
    

       
    merged_model = merge([user_embed, page_embed, depth_embed,
                          context_input], mode='concat')
        
    
    merged_model = TimeDistributed(Dense(output_dim=500, activation='tanh'))(merged_model)
    merged_model = Dropout(0.2)(merged_model)
                          
    merged_model = TimeDistributed(Dense(output_dim=500, activation='tanh'))(merged_model)
    merged_model = Dropout(0.2)(merged_model)
            
    
    merged_model = TimeDistributed(Dense(output_dim=1, activation='relu'))(merged_model)
    
    model = Model(input=[user_input, page_input, depth_input, context_input], output=[merged_model])
    
    model.compile(loss='mean_squared_error',
                         optimizer=optimizer,
                         metrics=['mean_squared_error'])
    
    model.summary()
    
#     plot(model, to_file='model.png')
    
    return model



def FNN_onestep_c(ctx_feat_num, user_num, page_num, timestep_num=100):
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
# optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
# optimizer = Adam(lr=0.0001)

    user_input = Input(shape=(timestep_num,), name='user_input')
    user_embed = Embedding(input_dim=user_num+1, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2
                            )(user_input)
    
    page_input = Input(shape=(timestep_num,), name='page_input')
    page_embed = Embedding(input_dim=page_num+1, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2
                            )(page_input)               
    
    depth_input = Input(shape=(timestep_num,), name='dep_input')
    depth_embed = Embedding(input_dim=101, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2
                            )(depth_input)
    
        
    context_input = Input(shape=(timestep_num, ctx_feat_num), name='ctx_input')
    

       
    merged_model = merge([user_embed, page_embed, depth_embed,
                          context_input], mode='concat')
        
    
    merged_model = TimeDistributed(Dense(output_dim=500, activation='tanh'))(merged_model)
    merged_model = Dropout(0.2)(merged_model)
                          
    merged_model = TimeDistributed(Dense(output_dim=500, activation='tanh'))(merged_model)
    merged_model = Dropout(0.2)(merged_model)
            
    
    merged_model = TimeDistributed(Dense(output_dim=1, activation='sigmoid'))(merged_model)
    
    model = Model(input=[user_input, page_input, depth_input, context_input], output=[merged_model])
    
    # "we are using the logarithmic loss function (binary_crossentropy) during training
    # the preferred loss function for binary classification problems."
    model.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=['binary_crossentropy', 'accuracy'])
    
    model.summary()
    
#     plot(model, to_file='model.png')
    
    return model



def RNN_upc_embed_r(ctx_feat_num, user_num, page_num, timestep_num=100):
    optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.99, nesterov=True)
# optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
# optimizer = Adam(lr=0.0001)

    user_input = Input(shape=(timestep_num,), name='user_input')
    user_embed = Embedding(input_dim=user_num+1, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2,
#                             init='normal',
#                            W_regularizer=l2(0.001)
                            )(user_input)
#     user_embed = Dropout(0.2)(user_embed)
    
    page_input = Input(shape=(timestep_num,), name='page_input')
    page_embed = Embedding(input_dim=page_num+1, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2,
#                             init='normal',
#                            W_regularizer=l2(0.001)
                            )(page_input)
#     page_embed = Dropout(0.2)(page_embed)                
    
    depth_input = Input(shape=(timestep_num,), name='dep_input')
    depth_embed = Embedding(input_dim=101, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2,
#                             init='normal',
#                             W_regularizer=l2(0.001)
                            )(depth_input)
#    depth_embed = Dropout(0.2)(depth_embed)  
    
        
    context_input = Input(shape=(timestep_num, ctx_feat_num), name='ctx_input')
    
    
    user_page_merge= merge([user_embed, page_embed], mode='mul', dot_axes=2)
    user_page_merge = Dropout(0.2)(user_page_merge)
    user_depth_merge = merge([user_embed, depth_embed], mode='mul', dot_axes=2)
    user_depth_merge = Dropout(0.2)(user_depth_merge)
    page_depth_merge = merge([page_embed, depth_embed], mode='mul', dot_axes=2)
    page_depth_merge = Dropout(0.2)(page_depth_merge)
    user_page_depth_merge = merge([user_embed, page_embed, depth_embed], mode='mul', dot_axes=2)
    user_page_depth_merge = Dropout(0.2)(user_page_depth_merge)
    

       
    merged_model = merge([user_embed, page_embed, depth_embed,
                        user_page_merge, user_depth_merge, page_depth_merge, 
                        user_page_depth_merge, 
                        context_input], mode='concat')
#     merged_model = Dropout(0.2)(merged_model)
    '''
    merged_model = LSTM(output_dim=500, activation='tanh',
                          return_sequences=True, consume_less='gpu',
#                           init='normal',
#                           W_regularizer=l2(0.0001)
                          )(merged_model)
#     merged_model = LeakyReLU(alpha=.01)(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    
         
    merged_model = LSTM(output_dim=500, activation='tanh',
                          return_sequences=True, consume_less='gpu',
#                           init='normal',
#                           W_regularizer=l2(0.0001)
                          )(merged_model)
#     merged_model = LeakyReLU(alpha=.01)(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    '''
    
         
    merged_model = Bidirectional(LSTM(output_dim=500, activation='tanh',
                          return_sequences=True, consume_less='gpu',
#                           init='normal',
#                           W_regularizer=l2(0.0001)
                          ), merge_mode='ave')(merged_model)
    merged_model = Dropout(0.2)(merged_model)
        
    merged_model = Bidirectional(LSTM(output_dim=500, activation='tanh',
                          return_sequences=True, consume_less='gpu',
#                           init='normal',
#                           W_regularizer=l2(0.0001)
                          ), merge_mode='ave')(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    
    
    
    merged_model = TimeDistributed(Dense(output_dim=1, activation='relu'))(merged_model)
    
    model = Model(input=[user_input, page_input, depth_input, context_input], output=[merged_model])
    
    
    model.compile(loss='mean_squared_error',
                         optimizer=optimizer,
                         metrics=['mean_squared_error'])
    
    model.summary()
    
#     plot(model, to_file='model.png')
    
    return model
    




def RNN_upc_embed_c(ctx_feat_num, user_num, page_num, timestep_num=100):
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
# optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08)
# optimizer = Adam(lr=0.0001)

    user_input = Input(shape=(timestep_num,), name='user_input')
    user_embed = Embedding(input_dim=user_num+1, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2,
#                             init='normal',
#                            W_regularizer=l2(0.001)
                            )(user_input)
#     user_embed = Dropout(0.2)(user_embed)
    
    page_input = Input(shape=(timestep_num,), name='page_input')
    page_embed = Embedding(input_dim=page_num+1, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2,
#                             init='normal',
#                            W_regularizer=l2(0.001)
                            )(page_input)
#     page_embed = Dropout(0.2)(page_embed)                
    
    depth_input = Input(shape=(timestep_num,), name='dep_input')
    depth_embed = Embedding(input_dim=101, output_dim=500, input_length=timestep_num,
                            weights=None, 
                            dropout=0.2,
#                             init='normal',
#                             W_regularizer=l2(0.001)
                            )(depth_input)
#    depth_embed = Dropout(0.2)(depth_embed)  
    
        
    context_input = Input(shape=(timestep_num, ctx_feat_num), name='ctx_input')
    
    
    user_page_merge= merge([user_embed, page_embed], mode='mul', dot_axes=2)
    user_page_merge = Dropout(0.2)(user_page_merge)
    user_depth_merge = merge([user_embed, depth_embed], mode='mul', dot_axes=2)
    user_depth_merge = Dropout(0.2)(user_depth_merge)
    page_depth_merge = merge([page_embed, depth_embed], mode='mul', dot_axes=2)
    page_depth_merge = Dropout(0.2)(page_depth_merge)
    user_page_depth_merge = merge([user_embed, page_embed, depth_embed], mode='mul', dot_axes=2)
    user_page_depth_merge = Dropout(0.2)(user_page_depth_merge)
    

       
    merged_model = merge([user_embed, page_embed, depth_embed,
                        user_page_merge, user_depth_merge, page_depth_merge, 
                        user_page_depth_merge, 
                        context_input], mode='concat')
#     merged_model = Dropout(0.2)(merged_model)
    
    merged_model = LSTM(output_dim=500, activation='tanh',
                          return_sequences=True, consume_less='gpu',
#                           init='normal',
#                           W_regularizer=l2(0.0001)
                          )(merged_model)
#     merged_model = LeakyReLU(alpha=.01)(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    
         
    merged_model = LSTM(output_dim=500, activation='tanh',
                          return_sequences=True, consume_less='gpu',
#                           init='normal',
#                           W_regularizer=l2(0.0001)
                          )(merged_model)
#     merged_model = LeakyReLU(alpha=.01)(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    '''
    
    merged_model = Bidirectional(LSTM(output_dim=500, activation='tanh',
                          return_sequences=True, consume_less='gpu',
#                           init='normal',
#                           W_regularizer=l2(0.0001)
                          ), merge_mode='ave')(merged_model)
#     merged_model = LeakyReLU(alpha=.01)(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    
         
    merged_model = Bidirectional(LSTM(output_dim=500, activation='tanh',
                          return_sequences=True, consume_less='gpu',
#                           init='normal',
#                           W_regularizer=l2(0.0001)
                          ), merge_mode='ave')(merged_model)
#     merged_model = LeakyReLU(alpha=.01)(merged_model)
    merged_model = Dropout(0.2)(merged_model)
        
    merged_model = Bidirectional(LSTM(output_dim=500, activation='tanh',
                          return_sequences=True, consume_less='gpu',
#                           init='normal',
#                           W_regularizer=l2(0.0001)
                          ), merge_mode='ave')(merged_model)
#     merged_model = LeakyReLU(alpha=.01)(merged_model)
    merged_model = Dropout(0.2)(merged_model)
    '''
    
    
    merged_model = TimeDistributed(Dense(output_dim=1, activation='sigmoid'))(merged_model)
    
    model = Model(input=[user_input, page_input, depth_input, context_input], output=[merged_model])
    
    # "we are using the logarithmic loss function (binary_crossentropy) during training
    # the preferred loss function for binary classification problems."
    model.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=['binary_crossentropy', 'accuracy'])
    
    model.summary()
    
#     plot(model, to_file='model.png')
    
    return model

def linearRegression_keras(feat_num, timestep_num=100):
    optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.99, nesterov=True)
    
    model = Sequential()

    model.add(TimeDistributed(Dense(output_dim=1, activation='relu'), input_shape=(timestep_num, feat_num))) # sequence labeling
    
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_squared_error'])
    
    return model

def logisticRegression_keras(feat_num, timestep_num=100):
    optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.99, nesterov=True)
# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
# optimizer = Adam(lr=0.001)
    
    model = Sequential()

    model.add(TimeDistributed(Dense(output_dim=1, activation='sigmoid'), input_shape=(timestep_num, feat_num))) # sequence labeling
    
#    model.add(TimeDistributed(Dense(output_dim=1, activation='linear'), input_shape=(timestep_num, feat_num))) # sequence labeling
    model.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=['binary_crossentropy', 'accuracy'])
  
    
    return model
