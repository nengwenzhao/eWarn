# coding: utf-8
import numpy as np
from pyspark.mllib.fpm import FPGrowth, PrefixSpan
from pyspark import SparkContext, SparkConf
from sklearn.metrics import precision_score,recall_score,f1_score

def fp_evaluate(train_data, train_label, test_data, test_label, minSupport = 0.1):
    
    positives = np.array(train_data)[np.array(train_label)==1]
    conf = SparkConf().setAppName("app-name-of-your-choice").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    rdd = sc.parallelize(positives, 8)
    model = FPGrowth.train(rdd, minSupport = minSupport, numPartitions = 4)
    return_result = sorted(model.freqItemsets().collect())
    print("-------频繁项集如下----------")
    print(return_result)
    print("---------------------------")
    result_label = []
    for i in test_data:
        flag = False
        for j in return_result:
            if len(list(set(j.items).intersection(set(i))))>0:
                result_label.append(1)
                flag=True
                break
        if not flag:
            result_label.append(0)
    sc.stop()
    print("-------频繁项集的检验结果-----")
    print("precision: {}".format(precision_score(test_label,result_label)))
    print("recall: {}".format(recall_score(test_label,result_label)))
    print("f1: {}".format(f1_score(test_label,result_label)))
    print("---------------------------")
    
    
    