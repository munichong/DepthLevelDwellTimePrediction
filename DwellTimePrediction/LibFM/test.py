'''
Created on Jul 12, 2016

@author: munichong
'''
from gensim.models import Doc2Vec
from sklearn.preprocessing import normalize
from numpy import array


doc2vec_model = Doc2Vec.load('../my_model_20.doc2vec')

d2v_vec = doc2vec_model.infer_vector(['this', 'is'])
print(d2v_vec)
print(type(d2v_vec))
d2v_vec = normalize([d2v_vec])
print(d2v_vec)
print(type(d2v_vec))
d2v_key = ' '.join(str(v) for v in d2v_vec[0])
print(d2v_key)
print(type(d2v_key))