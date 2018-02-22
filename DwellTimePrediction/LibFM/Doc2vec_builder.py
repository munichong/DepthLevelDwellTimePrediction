'''
Created on Jul 1, 2016

@author: munichong
'''
import re, pickle
# from itertools import ifilter
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from pymongo import MongoClient
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords


client = MongoClient()
userlog = client['Forbes_Apr2016']['FreqUserLogPV']
articleInfo = client['Forbes_Apr2016']['ArticleInfo']

'''
porter_stemmer = PorterStemmer()
     '''   
def body_text_generator():
    page_index = 0
    for article_meta in articleInfo.find():
        
        if 'body' in article_meta:
            raw_body = article_meta['body']
            raw_body = re.sub(r'\[.*?\]', ' ', raw_body)
            raw_body = re.sub(r'\[caption.*\>', ' ', raw_body)
            raw_body = re.sub(r'\[\/caption\]', '', raw_body)
            raw_body = re.sub(r'\[entity.*\>', '', raw_body)
            raw_body = re.sub(r'\[\/entity\]', '', raw_body)
            body_text = BeautifulSoup(raw_body).getText()
            
#             if len(body_text) < 100:
#                 body_text = 'unknown'
        else: 
            continue
#             body_text = 'unknown'
                    
      
#tokens = [porter_stemmer.stem(t) for t in 

        tokens = [t for t in 
                    filter(lambda w : w.isalpha() and len(w) > 1 and
                        w not in stopwords.words('english'), 
                            word_tokenize(body_text.lower()))]
        
#         if not tokens:
#             tokens = ['unknown']
        if len(tokens) == 1:
            continue
        
        page_index += 1
        print(page_index)
        
        yield LabeledSentence(tokens, ['PAGE_%s' % page_index])


# class LabeledLineSentence(object):
#     def __init__(self, tokens):
#         self.tokens = tokens
#     def __iter__(self):
#         page_index = 0
#         for t in self.tokens:
#             page_index += 1
#             print(page_index)
#             yield LabeledSentence(t, ['PAGE_%s' % page_index])
            
            
labeled_documents = list(body_text_generator())
# pickle.dump(labeled_documents, open('../labeled_documents.lls', 'wb'))

# all_body_tokens = pickle.load(open('../all_body_tokens.gen', 'rb'))

# labeled_documents = LabeledLineSentence(all_body_tokens)
# pickle.dump(labeled_documents, open('../labeled_documents.lls', 'wb'))

# labeled_documents = pickle.load(open('../labeled_documents.lls', 'rb'))


# documents = []
# documents.append( ['some', 'words', 'is', 'here'] )
# documents.append( ['some', 'people', 'words', 'like'] )
# documents.append( ['people', 'like', 'words'] )

# for doc in labeled_documents:
#     print(doc)


MIN_COUNT = 1

for size in [20, 50, 100, 150]:

    doc2vec_model = Doc2Vec(size=size, window=10, min_alpha=0.025, min_count=MIN_COUNT, workers=4)
#     labeled_documents = LabeledLineSentence(documents)
    doc2vec_model.build_vocab(labeled_documents)

    print("Doc2Vec: Training")
    doc2vec_model.train(labeled_documents)
#     print(doc2vec_model.docvecs['PAGE_1'])

    print("Doc2Vec: Store the model to mmap-able files")
    doc2vec_model.save('../d2v_model_%d_min%d.doc2vec' % (size, MIN_COUNT))
    print("Doc2Vec: Finish size = %d" % size)
    print('\n')


