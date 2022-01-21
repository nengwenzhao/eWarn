# coding: utf-8

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from utils.write_logs import write_log

def imbalance_sample(train_feature, train_label, method = 'SMOTE'):
    approach = None
    if method == 'SMOTE':
        approach = SMOTE(random_state=0)
    elif method == 'RandomOverSampler':
        approach = RandomOverSampler(random_state=0)
    else:
        return train_feature, train_label
    sample_feature, sample_label = approach.fit_resample(train_feature, train_label)
    print( method + "over sample done ")
    print("train(+): {}".format(np.array(sample_label)[np.array(sample_label)==1].shape[0]))
    print("train(-): {}".format(np.array(sample_label)[np.array(sample_label)==0].shape[0]))
    log = "\n" + method + "over sample method "+"\n"
    write_log(log)
    return sample_feature, sample_label

    
    
    