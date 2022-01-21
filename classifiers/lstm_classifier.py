# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def lstm_classifier(train_seq,train_seq_label,test_seq):

    epochs = 100
    batch_size = 500
    model = Sequential()
    model.add(LSTM(train_seq.shape[2], dropout=0.5, return_sequences=True,recurrent_dropout=0.5,input_shape = (train_seq.shape[1],train_seq.shape[2])))
    model.add(LSTM(200, dropout=0.5))
    model.add(Dense(1,activation='sigmoid'))
    train_label_num_1 = np.sum(np.array(train_seq_label) == 1)
    train_label_num_0 = np.sum(np.array(train_seq_label) == 0)
    cw = {0: train_label_num_1, 1: train_label_num_0}
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    model.fit(np.array(train_seq), np.array(train_seq_label),batch_size=batch_size,epochs=epochs,shuffle=False,class_weight=cw)
    result_proba = model.predict(np.array(test_seq))
    #model.fit(np.array(train_feature), np.array(train_label),batch_size=batch_size,epochs=epochs, shuffle=True,class_weight=cw)
    #result_proba = model.predict(np.array(test_feature))
    #precision, recall, f1 = evaluate_best(list(map(lambda x:(1-x),result_proba)), test_seq_label)
    K.clear_session()
    return list(map(lambda x:(1-x),result_proba))