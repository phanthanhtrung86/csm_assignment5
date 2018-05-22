## -*- coding: utf-8 -*-

import csv
import numpy as np
import lda

def get_hashtagset(collection):
    hashtagSet=set()
    for row in collection:
        for tag in row.split(' '):
            tag = tag.strip()
            if tag !='':
                hashtagSet.add(tag)
    return hashtagSet

def get_data_from_csv(dataSetLabel):
    dataList=list()
    fileName=''
    if dataSetLabel==1:
    	fileName='instagram_csv.csv'
    else:
    	fileName='youtube_csv.csv'
    print(fileName)
    with open(fileName, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            #print(row[0])
            #cell = unicode(row[0], 'utf-8')
            cell =row[0]
            dataList.append(cell)
    return dataList


def load_document(collection, vocab):
    documentSize=len(collection)
    dtm = np.zeros((documentSize, len(vocab)), dtype=np.intc)
    print(dtm.shape)
    postCount=0
    hashtagDict=dict()
    for j,ele in enumerate(vocab):
        hashtagDict[ele]=j
    for row in collection:
        for tag in row.split(' '):
            tag = tag.strip()
            if tag !='':
                if tag in hashtagDict:
                    dtm[postCount,hashtagDict[tag]]+=1
        postCount+=1
    return dtm

def get_input_for_lda(dataSetLabel):
    targetCol=get_data_from_csv(dataSetLabel)
    hashtagSet=get_hashtagset(targetCol)
    hashtagTuple = tuple(hashtagSet)
    vocab = hashtagTuple
    X=load_document(targetCol,vocab)
    return X,vocab


if __name__ == "__main__":
    print('Welcome to using LDA for topic modeling')
    dataSetLabel=input('Choose dataset [1] Instagram OR [2] Youtube: ')
    topicCount = input('Enter number of topic: ')
    topWordCount=input('Enter number of top word in each topic: ')
    number_of_topic=topicCount
    model = lda.LDA(n_topics=number_of_topic, n_iter=1500, random_state=1)
    X,vocab=get_input_for_lda(int(dataSetLabel))
    model.fit(X)
    topic_word = model.topic_word_
    n_top_words = topWordCount

    for i, topic_dist in enumerate(topic_word):
        try:
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print('Topic %d: %s'%(i,topic_words))
        except Exception as e:
            print(e)

