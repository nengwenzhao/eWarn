# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')
from features.lda import LDA_model


class LDA_multi_model(LDA_model):
    def __init__(self, train_data, test_data, train_label):

        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label

    def lda_feature_extraction(self, n_topics=30, doc_topic_prior=50, random_state=0):

        train_data = []
        train_label = []
        for i in range(len(self.train_data)):
            if self.train_label[i] == 0:
                train_data += self.train_data[i]
                train_label += [self.train_label[i]] * len(self.train_data[i])
            else:
                temp = []
                for tmp in self.train_data[i]:
                    temp += tmp
                train_data.append(temp)
                train_label.append(1)
        test_data = []
        for i in range(len(self.test_data)):
            test_data += self.test_data[i]

        self.train_feature, self.test_feature = self.lda_features(n_topics, doc_topic_prior, random_state, train_data,
                                                                  test_data)

        print("---LDA feature process done---")
        return self.train_feature, self.test_feature, train_label

    def multi_evaluate(self, test_data, predict_proba, test_label, threshold=0.5):

        predict_label = np.zeros(len(list(predict_proba)))
        predict_label[np.where(predict_proba < 0.5)] = 1

        start = 0
        bag_label = []
        for bag in test_data:
            temp = predict_label[start:(start + len(bag))]
            if 1 in temp:
                bag_label.append(1)
            else:
                bag_label.append(0)
            start += len(bag)

        precision = precision_score(test_label, bag_label)
        recall = recall_score(test_label, bag_label)
        f1 = f1_score(test_label, bag_label)
        print("评估结果：")
        print("precision：{}".format(precision))
        print("recall：{}".format(recall))
        print("f1:{}".format(f1))
        logs = "\nprecision:{:.4f}".format(precision) + \
           "\nrecall:{:.4f}".format(recall) + "\nf1：{:.4f}".format(f1) + "\n--------------\n"
        write_log(logs)
        return precision, recall, f1
