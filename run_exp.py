# -*- coding: utf-8 -*-
import numpy as np

from prepares.prefix import Prefix
from prepares.prefix_extra_features import Prefix_extra_features
from prepares.prefix_fasttext import Prefix_fasttext
from prepares.multi_instance import Multi_instance

from classifiers.classifier import rf_classifier, xgb_classifier, linear_select, rf_regressor
from classifiers.lstm_classifier import lstm_classifier
from classifiers.textCNN_classifier import textCNN_classifier
from classifiers.fasttext_classifier import fasttext_classifier

from features.lda import LDA_model
from features.fasttext_addgram import seq_padding_ML

from features.type_frequency import type_frequency
from features.tfidf import tfidf_features, tfidf_multi_instance

from utils.write_logs import write_log
from utils.evaluate import evaluate_best
from utils.imbalance_sample import imbalance_sample
from sklearn.cluster import AgglomerativeClustering


def lda_rf_test(app_name, target, data_name, alert_level, split_percent, bag_size, pos_size):


    Prefix_Sample = Prefix_extra_features(app_name=app_name, data_name=data_name, target=target,
                                          alert_level=alert_level)
    Prefix_Sample.sample(step=10, window_size=bag_size, react_size=10, positive_range=pos_size, min_log=3)

    train_data, train_label, test_data, test_label, train_extra_features, test_extra_features = Prefix_Sample.split_data(
        split_percent=split_percent)
    lda = LDA_model(train_data, test_data)

    # coherence topic
    # topic = lda.best_topic_coherence()
    topic = 30
    train_lda_features, test_lda_features = lda.lda_feature_extraction(n_topics=topic)
    lda.write_feature_name()



    train_features = np.concatenate((np.array(train_lda_features), np.array(train_extra_features)), axis=1)
    test_features = np.concatenate((np.array(test_lda_features), np.array(test_extra_features)), axis=1)
    train_features, test_features = linear_select(np.array(train_features), np.array(test_features), train_label)

    train_features, train_label_up = imbalance_sample(train_features, train_label, method='SMOTE')

    result_proba = xgb_classifier(train_features, test_features, train_label_up)
    precision, recall, f1 = evaluate_best(result_proba, test_label)


    train_features = np.concatenate((np.array(train_lda_features), np.array(train_extra_features)), axis=1)
    test_features = np.concatenate((np.array(test_lda_features), np.array(test_extra_features)), axis=1)

    train_features, train_label_up2 = imbalance_sample(train_features, train_label, method='SMOTE')
    result_proba = xgb_classifier(train_features, test_features, train_label_up2)
    precision, recall, f1 = evaluate_best(result_proba, test_label)


def mul_ins_test(app_name, target, data_name, alert_level, split_percent,pos_range):
    print(app_name, target)
    import time


    MIL_Sample = Multi_instance(app_name=app_name, data_name=data_name, target=target, alert_level=alert_level)
    MIL_Sample.sample(step=10, instance_size=10, bag_size=60, react_size=10, positive_range=pos_range,
                      min_log=3)
    train_data, train_label, test_data, test_label, train_extra_feature, test_extra_feature = MIL_Sample.split_data(
        split_percent=split_percent)

    lda = LDA_model(np.array(train_data).reshape((-1, 1)), np.array(test_data).reshape((-1, 1)))

    # topic = lda.best_topic_coherence()
    topic = 30
    train_feature, test_feature = lda.lda_feature_extraction(n_topics=topic)
    lda.write_feature_name()
    train_lda_feature, test_lda_feature = np.array(train_feature).reshape(
        (-1, np.array(train_data).shape[1], topic)), np.array(test_feature).reshape(
        (-1, np.array(test_data).shape[1], topic))

    train_bag_feature = np.concatenate((train_extra_feature, train_lda_feature), axis=2)
    test_bag_feature = np.concatenate((test_extra_feature, test_lda_feature), axis=2)

    def is_intersection(a, b):

        tmp = [1 for arr1 in a for arr2 in b if np.sum(abs(arr1 - arr2)) == 0]
        if not tmp:
            return 0
        else:
            return 1

    positive_bags = train_bag_feature[np.array(train_label) == 1]

    def Cluster_weight(train_bag, flag=True):
        clustering = AgglomerativeClustering().fit(train_bag) #n_cluster can be selected based on Silhouette Coefficient
        if flag:
            labels = clustering.labels_

        else:
            labels = clustering.fit_predict(train_bag)
        clusters = clustering.n_clusters_
        weight = []
        for n in range(clusters):
            # cluster_temp = train_bag[(labels == n), :]
            # # print(cluster_temp,cluster_temp.shape)
            # # print(positive_bags.shape)
            # count = sum(list(map(lambda x: is_intersection(cluster_temp, x), positive_bags))) + 1
            # beta = count / positive_bags.shape[0]
            # ck = (beta * np.log(beta) + (1 - beta) * np.log(1 - beta)) / np.log(0.5)
            ck = np.sum(labels==n)/len(labels)
            print(ck)
            weight.append(ck)

        array_weight = list(map(lambda x: weight[x], labels))
        array_weight = np.array(array_weight / sum(array_weight))
        #print(array_weight)
        return np.dot(train_bag.T, array_weight.T)

    train_feature = np.array(list(map(lambda x: Cluster_weight(x), train_bag_feature)))
    test_feature = np.array(list(map(lambda x: Cluster_weight(x,flag=False), test_bag_feature)))

    np.save('feature_data/'+app_name+'train_feature', train_feature)
    np.save('feature_data/'+app_name+'test_feature', test_feature)
    np.save('feature_data/'+app_name+'train_label', train_label)
    np.save('feature_data/'+app_name+'test_label', test_label)

    train_features, test_features = linear_select(np.array(train_feature), np.array(test_feature), train_label)


    result_proba = rf_classifier(train_features, test_features, train_label, weight=None)
    precision, recall, f1 = evaluate_best(result_proba, test_label)

    #train_features, train_label_up2 = imbalance_sample(train_feature, train_label, method='SMOTE')

    result_proba = xgb_classifier(train_feature, test_feature, train_label)
    precision, recall, f1 = evaluate_best(result_proba, test_label)
    del lda
    del MIL_Sample


def fasttext_test(app_name, target, data_name, alert_level, split_percent):

    VOCAB_SIZE = 40000
    Prefix_fasttext_Sample = Prefix_fasttext(app_name=app_name, data_name=data_name, target=target,
                                             alert_level=alert_level,
                                             VOCAB_SIZE=VOCAB_SIZE)
    Prefix_fasttext_Sample.sample(step=10, window_size=60, react_size=10, positive_range=120, min_log=3)
    train_data, train_label, test_data, test_label = Prefix_fasttext_Sample.split_data(split_percent=split_percent)
    # train_feature, test_feature = addgram(2, train_data, test_data, max_features=20000)
    (MAX_WORDS, train_feature, test_feature) = seq_padding_ML(train_data, test_data)
    result_proba = fasttext_classifier(train_feature, train_label, test_feature, VOCAB_SIZE, MAX_WORDS, epochs=50)
    precision, recall, f1 = evaluate_best(result_proba, test_label)


def textCNN_test(app_name, target, data_name, alert_level, split_percent):

    VOCAB_SIZE = 40000
    Prefix_fasttext_Sample = Prefix_fasttext(app_name=app_name, data_name=data_name, target=target,
                                             alert_level=alert_level,
                                             VOCAB_SIZE=VOCAB_SIZE)
    Prefix_fasttext_Sample.sample(step=10, window_size=60, react_size=10, positive_range=120, min_log=3)
    train_data, train_label, test_data, test_label = Prefix_fasttext_Sample.split_data(split_percent=split_percent)
    # train_feature, test_feature = addgram(2, train_data, test_data, max_features=20000)
    (MAX_WORDS, train_feature, test_feature) = seq_padding_ML(train_data, test_data)
    result_proba = textCNN_classifier(train_feature, train_label, test_feature, VOCAB_SIZE, MAX_WORDS, epochs=10)
    precision, recall, f1 = evaluate_best(result_proba, test_label)


def fre_xgb_test(app_name, target, data_name, alert_level, split_percent):

    Prefix_Sample = Prefix(app_name=app_name, data_name=data_name, target=target, alert_level=alert_level)
    Prefix_Sample.sample(step=10, window_size=60, react_size=10, positive_range=120, min_log=3)
    train_data, train_label, test_data, test_label = Prefix_Sample.split_data(split_percent=split_percent)
    train_feature, test_feature = type_frequency(data_name, train_data, test_data, target, app_name)
    result_proba = xgb_classifier(train_feature, test_feature, train_label)
    precision, recall, f1 = evaluate_best(result_proba, test_label)


def tfidf_lstm_test(app_name, target, data_name, alert_level, split_percent):
    Prefix_Sample = Prefix(app_name=app_name, data_name=data_name, target=target, alert_level=alert_level)
    Prefix_Sample.sample(step=10, window_size=60, react_size=10, positive_range=120, min_log=3)
    train_data, train_label, test_data, test_label = Prefix_Sample.split_data(split_percent=split_percent)

    train_feature, test_feature = tfidf_features(train_data, test_data)
    train_seq, train_seq_label, test_seq, test_seq_label = tfidf_multi_instance(train_feature, test_feature,
                                                                                train_label, test_label, False)

    # LSTM Classifier
    result_proba = lstm_classifier(train_seq, train_seq_label, test_seq)
    precision, recall, f1 = evaluate_best(result_proba, test_seq_label)


def lda_tfidf_ensemble(app_name, target, data_name, alert_level, split_percent):

    Prefix_Sample = Prefix(app_name=app_name, data_name=data_name, target=target, alert_level=alert_level)
    Prefix_Sample.sample(step=10, window_size=60, react_size=10, positive_range=120, min_log=3)

    train_data, train_label, test_data, test_label = Prefix_Sample.split_data(split_percent=split_percent)
    lda = LDA_model(train_data, test_data)
    train_feature_LDA, test_feature_LDA = lda.lda_feature_extraction(n_topics=30, doc_topic_prior=50, random_state=0)
    train_feature_TFIDF, test_feature_TFIDF = tfidf_features(train_data, test_data)
    L = 5
    step = 1
    counts = 2
    train_multi_LDA = np.array(
        [train_feature_LDA[x:(x + L)].mean(axis=0) for x in range(0, len(train_feature_LDA) - L + 1, step)])
    train_multi_TFIDF = np.array(
        [train_feature_TFIDF[x:(x + L), :].toarray() for x in range(0, train_feature_TFIDF.shape[0] - L + 1, step)])

    train_multi_label = np.array(
        [1 if train_label[x:(x + L)].count(1) >= counts else 0 for x in range(0, len(train_label) - L + 1, step)])

    # test_multi_LDA = np.array([test_feature_LDA[x+L-1] for x in range(0,len(test_feature_LDA)-L+1,step)])
    test_multi_LDA = np.array(
        [test_feature_LDA[x:(x + L)].mean(axis=0) for x in range(0, len(test_feature_LDA) - L + 1, step)])
    test_multi_TFIDF = np.array(
        [test_feature_TFIDF[x:(x + L), :].toarray() for x in range(0, test_feature_TFIDF.shape[0] - L + 1, step)])

    # test_multi_label = np.array([test_label[x+L-1] for x in range(0,len(test_label)-L+1,step)])
    test_multi_label = np.array(
        [1 if test_label[x:(x + L)].count(1) >= counts else 0 for x in range(0, len(test_label) - L + 1, step)])
    result_proba_LDA = rf_classifier(train_multi_LDA, test_multi_LDA, train_multi_label, weight='balanced')
    result_proba_TFIDF = lstm_classifier(train_multi_TFIDF, train_multi_label, test_multi_TFIDF)
    result_proba_ensemble = list(map(lambda x, y: (x + y) / 2, result_proba_LDA, result_proba_TFIDF))
    evaluate_best(result_proba_ensemble, test_multi_label)


app_name = 'NBANK'  #the name of service systems
target = ''   #the incident you want to predict
data_name = 'data.csv'
mul_ins_test(app_name, target, data_name, 1, 0.6,60)


