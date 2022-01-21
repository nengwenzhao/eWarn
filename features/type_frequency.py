# coding: utf-8

import numpy as np
import pandas as pd
from utils.write_logs import write_log
import re


def keyword(df, keyword, if_true=True):
    pattern = re.compile('.*' + keyword + '.*')
    if (pattern.match(df["N_SUMMARYCN"]) is not None) and (df['N_CUSTOMERSEVERITY'] == 1):
        return if_true
    else:
        return not if_true

def get_type(df):
    pattern = re.compile('(.*?):.*')
    match = pattern.match(df['N_SUMMARYCN'])
    return 'Others' if not match else match.group(1)
    
def get_type_index(type_dict, x):
    pattern = re.compile('(.*?):.*')
    match = pattern.match(x)
    if not match:
        return type_dict.index('Others')
    else:
        if match.group(1) in type_dict:
            return type_dict.index(match.group(1))
        else:
            return type_dict.index('Others')
        
def freq_vector(type_dict,lst):
    temp = [0 for x in type_dict]
    for x in lst:
        temp[x]+=1
    return temp

def type_frequency(data_name, train_data, test_data, target, app_name):
    df = pd.read_csv('./datas/raw_data/{}'.format(data_name))
    temp = df[(df.apply(keyword, keyword = target, if_true = False, axis = 1)) & (df['N_CUSTOMERSEVERITY'] != 1) & (df['N_APPNAME']==app_name)]
    temp['types'] = temp.apply(get_type,axis = 1)
    type_dict = list(temp.groupby(by = 'types').count().sort_values(by = 'firsttimestamp',ascending = False).index)
    if 'Others' not in type_dict:
        type_dict.append('Others')
    train_feature = list(map(lambda z:freq_vector(type_dict,z),list(map(lambda x:list(map(lambda y:get_type_index(type_dict,y),x)),train_data))))
    test_feature = list(map(lambda z:freq_vector(type_dict,z),list(map(lambda x:list(map(lambda y:get_type_index(type_dict,y),x)),test_data))))
    write_log("\n训练集的特征维度：{}".format(np.array(train_feature).shape))
    return train_feature,test_feature
    
def type_multi_instance(train_feature,test_feature,train_label,test_label):
    L = 3
    step = 1
    counts = 1
    train_seq = np.array([train_feature[x:(x+L)] for x in range(0,len(train_feature)-L+1,step)])
    train_seq_label = np.array([1 if train_label[x:(x+L)].count(1)>=counts else 0 for x in range(0,len(train_label)-L+1,step)])
    test_seq = np.array([test_feature[x:(x+L)] for x in range(0,len(test_feature)-L+1,step)])
    test_seq_label = np.array([1 if test_label[x:(x+L)].count(1)>=counts else 0  for x in range(0,len(test_label)-L+1,step)])
    return train_seq,train_seq_label,test_seq,test_seq_label
    
    
    