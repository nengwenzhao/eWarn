# -*- coding: UTF-8 -*-

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dense

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


def fasttext_classifier(train_feature, train_label, test_feature, VOCAB_SIZE, MAX_WORDS, epochs=15):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_WORDS))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(CLASS_NUM, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    model.fit(train_feature, train_label,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.04)
    res = model.predict(test_feature)[:, 0]
    K.clear_session()
    return res
