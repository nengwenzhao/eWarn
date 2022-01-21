# coding: utf-8

def split_data(datas, labels, split_percent=0.7):
    train_data = datas[:int(split_percent * len(datas))]
    train_label = labels[:int(split_percent * len(datas))]
    test_data = datas[int(split_percent * len(datas)):]
    test_label = labels[int(split_percent * len(datas)):]
    print("---按照{}比例分训练测试集---".format(split_percent))
    print("训练集正样本数目：{}".format(train_label.count(1)))
    print("训练集负样本数目：{}".format(train_label.count(0)))
    print("测试集正样本数目：{}".format(test_label.count(1)))
    print("测试集负样本数目：{}".format(test_label.count(0)))
    return train_data, train_label, test_data, test_label
