'''
Created on Aug 28, 2016

@author: munichong
'''
import sys, math
import numpy as np
import model_input_gen as mig
from keras.preprocessing import sequence
from sklearn.linear_model import SGDRegressor
from keras_rnn import RNN_keras

BATCH_SIZE = 2000
NUM_EPOCH = 5

num_batch = math.ceil( mig.tts.valid_pv_num / BATCH_SIZE )

# sgdRegressor = SGDRegressor()

print('\nBuild model...')
rnn = RNN_keras(mig.tts.MAX_SEQ_LEN, len(mig.vectorizer.feature_names_))


print("\n****** Iterating over each batch of the training data ******")
for epoch in range(1, NUM_EPOCH+1):
    batch_index = 0
    for X_batch, y_batch in mig.Xy_gen(mig.X_train, mig.y_train, batch_size=BATCH_SIZE):
        batch_index += 1
    #     sgdRegressor.partial_fit(X_batch, y_batch)
#         print(y_batch)
        X_train_pad = sequence.pad_sequences(X_batch, maxlen=mig.tts.MAX_SEQ_LEN, padding='pre', value=-1.0)
        y_train_pad = sequence.pad_sequences(y_batch, maxlen=mig.tts.MAX_SEQ_LEN, padding='pre', value=-1.0)

#         print(X_train_pad)
#         print(y_train_pad)
#         print(X_train_pad.shape)
#         print(y_train_pad.shape)
        
        loss = rnn.train_on_batch(X_train_pad, y_train_pad)
#         print(loss.history)
#         print(rnn.metrics_names)
        print("Epoch", epoch, '/', NUM_EPOCH, ": Batch", batch_index, '/', num_batch, "|",
              rnn.metrics_names[0], "=", loss[0], "| root", rnn.metrics_names[1], "=", np.sqrt(loss[1]))
#         print(rnn.predict_on_batch(X_train_pad).shape)
#         print(rnn.predict_on_batch(X_train_pad))
#         print("******************") 
#         if loss[0] > 10000:
#             sys.exit()
    print()    
         
# The shape of X_train should be (#examples, #values in sequences, dim. of each value)
print()
print("After padding, %d training cases with length %d." %
      (len(mig.X_train), X_train_pad.shape[2]))

