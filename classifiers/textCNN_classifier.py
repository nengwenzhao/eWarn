# -*- coding: UTF-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras import Input
from keras.models import Sequential
from keras.callbacks import *
from keras.layers import Embedding,Conv1D,MaxPool1D,Flatten,Dropout
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dense,concatenate
from keras.models import Model

from keras import backend as K


EMBEDDING_DIM = 100
CLASS_NUM = 1
batch_size = 32


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


def textCNN_classifier(train_feature, train_label, test_feature, VOCAB_SIZE, MAX_WORDS, epochs=15):
    x_input = Input(shape=(MAX_WORDS,))
    x_emb = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_WORDS)(x_input)
    pool_output = []
    #kernel_sizes = [2, 3, 4, 10, 20, 30, 40]
    # for kernel_size in kernel_sizes:
    #     c = Conv1D(filters=2, kernel_size=kernel_size, strides=1)(x_emb)
    #     p = MaxPool1D(pool_size=int(c.shape[1]))(c)
    #     p = Dropout(0.5)(p)
    #     pool_output.append(p)
    # pool_output = concatenate([p for p in pool_output])
    # x_flatten = Flatten()(pool_output)
    c = Conv1D(filters=2, kernel_size=8, strides=1)(x_emb)
    p = MaxPool1D(pool_size=2)(c)
    p = Dropout(0.5)(p)
    x_flatten = Flatten()(p)
    y = Dense(CLASS_NUM, activation='sigmoid')(x_flatten)
    model = Model([x_input], outputs=[y])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    early_stopping = EarlyStopping(monitor='acc', patience=0,min_delta=0.0001)
    model.fit(train_feature, train_label,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[early_stopping])
    res = model.predict(test_feature)[:, 0]
    K.clear_session()
    return res
