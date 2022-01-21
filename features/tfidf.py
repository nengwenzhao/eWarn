# -*- coding: UTF-8 -*-

from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer 
import jieba
import re
import numpy as np
import pandas as pd


def tfidf_features(train_data, test_data):
    corpus_train = []
    corpus_test = []
    for index, sample in enumerate(train_data):
        alert_token = ""
        ch_alert = []
        for i in sample:
            temp = re.sub('[0-9\!\%\[\]\,\。]', '', i)
            temp = re.sub(r'[^\w\s]', '', temp).replace(' ', '')
            temp = temp.replace('_', '')
            ch_alert.append(temp)

        for i in ch_alert:
            seg_list = jieba.cut(i, cut_all=False)
            seg_list = " ".join(seg_list)
            alert_token += seg_list
        corpus_train.append(alert_token)

    for index, sample in enumerate(test_data):
        alert_token = ""
        ch_alert = []
        for i in sample:
            temp = re.sub('[0-9\!\%\[\]\,\。]', '', i)
            temp = re.sub(r'[^\w\s]', '', temp).replace(' ', '')
            temp = temp.replace('_', '')
            ch_alert.append(temp)

        for i in ch_alert:
            seg_list = jieba.cut(i, cut_all=False)
            seg_list = " ".join(seg_list)
            alert_token += seg_list
        corpus_test.append(alert_token)

    cntVector = CountVectorizer()
    tfidf=TfidfTransformer()
    vocaubulary = cntVector.fit(corpus_train)
    cntTf1 = cntVector.transform(corpus_train)
    cntTf2 = cntVector.transform(corpus_test)
    tfidf_train = tfidf.fit(cntTf1)
    train_feature = tfidf_train.transform(cntTf1)
    test_feature = tfidf_train.transform(cntTf2)
    
    return train_feature, test_feature

def tfidf_multi_instance(train_feature,test_feature,train_label,test_label,multi = True):
    L = 3
    step = 1
    counts = 1
    if multi:
        train_seq = np.array([train_feature[x:(x+L),:].toarray() for x in range(0,train_feature.shape[0]-L+1,step)])
        train_seq_label = np.array([1 if train_label[x:(x+L)].count(1)>=counts else 0 for x in range(0,len(train_label)-L+1,step)])
        test_seq = np.array([test_feature[x:(x+L),:].toarray() for x in range(0,test_feature.shape[0]-L+1,step)])
        test_seq_label = np.array([1 if test_label[x:(x+L)].count(1)>=counts else 0  for x in range(0,len(test_label)-L+1,step)])
        return train_seq,train_seq_label,test_seq,test_seq_label
    else:
        train_seq = np.array([train_feature[x:(x+L),:].toarray() for x in range(0,train_feature.shape[0]-L+1,step)])
        train_seq_label = np.array([train_label[x+L-1] for x in range(0,len(train_label)-L+1,step)])
        test_seq = np.array([test_feature[x:(x+L),:].toarray() for x in range(0,test_feature.shape[0]-L+1,step)])
        test_seq_label = np.array([test_label[x+L-1] for x in range(0,len(test_label)-L+1,step)])
        return train_seq,train_seq_label,test_seq,test_seq_label
