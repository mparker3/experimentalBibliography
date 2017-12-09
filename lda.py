#-*- coding: utf-8 -*-
#^^ added to resolve error with special characters in txt files

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import pos_tag
import re
import string
from collections import defaultdict
from gensim import corpora
import gensim
import os
import sys
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def parse(path, doc):
    #loadCleanCorpus("35Corpora/CANON.txt")
    #too lengthy for testing
    
    corpus = []
    counter = 0
    for filename in os.listdir(path):
        print(counter)
        if filename not in [".DS_Store", "CANON.txt"]:
            cleanText = loadCleanNovel(path + filename)
            corpus.append(cleanText)
        counter += 1
    
    cleanDoc = loadCleanNovel(doc)
    preppedDoc = prepDoc(cleanDoc)
    corpusModel = topicModel(corpus, cleanDoc)
    
def prepDoc(doc):
    chunks = []
    for i in range(0, len(doc), 30):
        chunks.append(doc[i:i+30])
    return chunks


def topicModel(corpus, query_text):
    
    newcorpus = corpus
    dictionary = corpora.Dictionary(newcorpus)
    corpus = [dictionary.doc2bow(doc) for doc in corpus]
    lda = gensim.models.ldamodel.LdaModel
    ldamodel = lda(corpus=corpus, num_topics=5, id2word = dictionary,passes=100) 
    
    query_text_bow = dictionary.doc2bow(query_text)
    query_text_lda = ldamodel[query_text_bow]
    sametopic = 0
    topics = 0
    print(ldamodel.print_topic(query_text_lda[0][0], topn=10))
    
    return ldamodel

def loadCleanNovel(filename):
    
    specialchars = ["’", "“", "”"]
    corpus = open(filename, 'r')
    tokenizedCorpus = []
    for line in corpus:
        tokenizedCorpus += word_tokenize(line)
    
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    tokenizedCorpusNoPunctuation = []
    for token in tokenizedCorpus:
        newToken = regex.sub(u'', token)
        if not newToken == u'':
            tokenizedCorpusNoPunctuation.append(newToken)
    noStopwords = []
    stopwordsSet = set(stopwords.words('english'))
    for word in tokenizedCorpusNoPunctuation:
        if word.lower() not in stopwordsSet and word.lower() not in specialchars:
            noStopwords.append(word)
    frequency = defaultdict(int)
    for token in noStopwords:
        frequency[token] += 1
    clean = [token for token in noStopwords if frequency [token] > 1]
#    print(clean)
    pos = pos_tag(clean)
    nouns = []
    for pair in pos:
        if pair[1] in ["NN", "ADJ"] :
            nouns.append(pair[0])
    lesscommon = []
    fdist = FreqDist(word for word in nouns)
    mostCommon = fdist.most_common(25)
    lesscommon = [word for word in nouns if word not in mostCommon]
    return lesscommon

if __name__ == "__main__":
    print("done importing")
    parse(sys.argv[1], sys.argv[2])
