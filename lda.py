#-*- coding: utf-8 -*-
#^^ added to resolve error with special characters in txt files

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
from collections import defaultdict
from gensim import corpora
import gensim
import os
import sys
def parse(path):
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
    topicModel(corpus)

def topicModel(corpus):
    
    #split book into documents for testing
    newcorpus = corpus
    
    dictionary = corpora.Dictionary(newcorpus)
    
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in newcorpus]

    lda = gensim.models.ldamodel.LdaModel

    ldamodel = lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

    print(ldamodel.print_topics(num_topics=3, num_words=3))

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
    print(stopwordsSet)
    for word in tokenizedCorpusNoPunctuation:
        if word.lower() not in stopwordsSet and word.lower() not in specialchars:
            noStopwords.append(word)
        else:
            print(word)
    frequency = defaultdict(int)
    for token in noStopwords:
        frequency[token] += 1
    clean = [token for token in noStopwords if frequency [token] > 1]
#    print(clean)
    return clean

if __name__ == "__main__":
    print("done importing")
    parse(sys.argv[1])
