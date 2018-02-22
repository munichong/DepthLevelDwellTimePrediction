import re, pickle
from itertools import ifilter
from gensim import corpora, models
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from pymongo import MongoClient
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords


client = MongoClient()
userlog = client['Forbes_Dec2015']['FreqUserLogPV']
articleInfo = client['Forbes_Dec2015']['ArticleInfo']


porter_stemmer = PorterStemmer()
        
def body_text_generator():
    page_index = 0
    all_doc_tokens = []
    for article_meta in articleInfo.find():
        
        if 'body' in article_meta:
            body_text = BeautifulSoup(article_meta['body']).getText()
            body_text = re.sub(r'\[.*?\]', ' ', body_text)
#             if len(body_text) < 100:
#                 body_text = 'unknown'
        else: 
            continue
#             body_text = 'unknown'
                    
        tokens = [porter_stemmer.stem(t) for t in 
                          ifilter(lambda w : w.isalpha() and len(w) > 1 and
                                  w not in stopwords.words('english'), 
                                  word_tokenize(body_text.lower()))]
        
#         if not tokens:
#             tokens = ['unknown']
        if len(tokens) < 100:
            continue
        
        page_index += 1
        print(page_index)
        
        all_doc_tokens.append(tokens)
    return all_doc_tokens

tokens = body_text_generator()

""" Build LDA Model (over both training and test pages) """
print("\nBuilding LDA dictionary...")
dictionary = corpora.Dictionary(tokens)
print("Converting doc to bow...")
corpus = [dictionary.doc2bow(d) for d in tokens]
print("Creating the MM file...")
corpora.MmCorpus.serialize('lda.mm', corpus)
mm = corpora.MmCorpus('lda.mm')

dictionary.save('../lda_models/dictionary.dict')

for topic_k in [40]: # 30 may lead to be out of memory 
    print("Building a LDA model for %d topics ..." % topic_k)
    lda = models.ldamodel.LdaModel(corpus=mm, num_topics=topic_k)
    print("A LDA model with %d topics is built.\n" % topic_k)
    lda.save('../lda_models/lda_model_%d.lda' % topic_k)
    print("Finish saving lda_model_%d.lda" % topic_k)
    print('\n')
    
#     doc_lda = lda[dictionary.doc2bow(clean_tokens[0])]
#     pprint(doc_lda)    
    
    
    
    
