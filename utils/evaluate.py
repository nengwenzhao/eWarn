# coding: utf-8

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.write_logs import write_log


def evaluate(result_proba, test_label, threshold=0.5):
    result_label = np.zeros(len(list(result_proba)))
    result_label[np.where(result_proba < threshold)] = 1
    precision = precision_score(test_label, result_label)
    recall = recall_score(test_label, result_label)
    f1 = f1_score(test_label, result_label)
    print("评估结果：")
    print("预测的正样本数目：{}".format(np.sum(np.array(result_label) == 1)))
    print("预测的负样本数目：{}".format(np.sum(np.array(result_label) == 0)))
    print("测试集的正样本数目：{}".format(np.sum(np.array(test_label) == 1)))
    print("测试集的负样本数目：{}".format(np.sum(np.array(test_label) == 0)))
    print("precision：{:.4f}".format(precision))
    print("recall：{:.4f}".format(recall))
    print("f1:{:.4f}".format(f1))
    logs = "\nprecision:{:.4f}".format(precision) + \
           "\nrecall:{:.4f}".format(recall) + "\nf1：{:.4f}".format(f1) + "\n--------------\n"
    write_log(logs)

    return precision, recall, f1


def evaluate_best(result_proba, test_label):
    threshold = np.arange(0, 1, 0.01)
    result_labels = list(map(lambda x: list(map(lambda y: 1 if y < x else 0, result_proba)), threshold))
    precisions = list(map(lambda x: precision_score(test_label, x), result_labels))
    recalls = list(map(lambda x: recall_score(test_label, x), result_labels))
    f1s = list(map(lambda x: f1_score(test_label, x), result_labels))
    max_index = f1s.index(max(f1s))
    print("最佳评估结果：")
    print("precision：{:.4f}".format(precisions[max_index]))
    print("recall：{:.4f}".format(recalls[max_index]))
    print("f1:{:.4f}".format(f1s[max_index]))
    logs = "\nprecision:{:.4f}".format(precisions[max_index]) + \
           "\nrecall:{:.4f}".format(recalls[max_index]) + "\nf1：{:.4f}".format(f1s[max_index]) + "\n--------------\n"
    write_log(logs)
    return precisions[max_index], recalls[max_index], f1s[max_index]


def evaluate_mul_best(result_proba, test_label):
    def multi_predict(result_proba, threshold, count=1):
        result_label = np.zeros(len(list(result_proba)))
        result_label[np.where(result_proba < threshold)] = 1
        if np.sum(result_label == 1) >= count:
            return 1
        else:
            return 0

    def get_label(result_proba, threshold, count=1):
        return list(map(lambda x: multi_predict(x, threshold=threshold, count=count), result_proba))

    threshold = np.arange(0.1, 1, 0.05)
    result_labels = []
    for count in range(1, np.array(result_proba).shape[1] + 1):
        result_labels += list(map(lambda x: get_label(result_proba, x, count), threshold))
    precisions = list(map(lambda x: precision_score(test_label, x), result_labels))
    recalls = list(map(lambda x: recall_score(test_label, x), result_labels))
    f1s = list(map(lambda x: f1_score(test_label, x), result_labels))
    max_index = f1s.index(max(f1s))
    print("最佳评估结果：")
    print("precision：{:.4f}".format(precisions[max_index]))
    print("recall：{:.4f}".format(recalls[max_index]))
    print("f1:{:.4f}".format(f1s[max_index]))
    logs = "\nprecision:{:.4f}".format(precisions[max_index]) + \
           "\nrecall:{:.4f}".format(recalls[max_index]) + "\nf1：{:.4f}".format(f1s[max_index]) + "\n--------------\n"
    write_log(logs)
    return precisions[max_index], recalls[max_index], f1s[max_index]
