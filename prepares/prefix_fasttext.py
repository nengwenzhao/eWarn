# coding: utf-8


from prepares.prefix import Prefix
import numpy as np
import pandas as pd
import jieba
from utils.split_data import split_data
from utils.write_logs import write_log

class Prefix_fasttext(Prefix):
    def __init__(self, app_name='', data_name='data.csv', target='',alert_level = 1 , VOCAB_SIZE=25000):
        super(Prefix_fasttext, self).__init__(app_name, data_name, target,alert_level)
        self.VOCAB_SIZE = 25000

    def getStopWords(self, datapath):
        stopwords = pd.read_csv(datapath, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
        stopwords = stopwords["stopword"].values
        return stopwords

    def preprocess_text(self, texts, stopwords, vocab_size):
        sentences = []
        bag_dict = {}
        for logs in texts:
            tmp = []
            for text in logs:
                try:
                    segs = jieba.lcut(text)
                    segs = filter(lambda x: len(x) > 1, segs)
                    segs = filter(lambda x: x not in stopwords, segs)
                    for seg in segs:
                        if seg not in bag_dict.keys():
                            bag_dict[seg] = 1
                        else:
                            bag_dict[seg] += 1
                        tmp.append(seg)

                except Exception as e:
                    print(text)
                    continue
            sentences.append(tmp)
        vocab = sorted(bag_dict, key=lambda x: bag_dict[x], reverse=True)[:vocab_size]
        for i, sentence in enumerate(sentences):
            sentences[i] = list(map(lambda x: vocab.index(x) + 1 if x in vocab else 0, sentence))
        return vocab, sentences

    def split_data(self, split_percent=0.7):
        StopWords = self.getStopWords('./datas/pre_data/stopwords.csv')
        vocab, sentences = self.preprocess_text(self.datas, StopWords, self.VOCAB_SIZE)
        
        
        train_data, train_label, test_data, test_label = split_data(sentences, self.labels, split_percent)
        train_label_num_1 = np.sum(np.array(train_label) == 1)
        train_label_num_0 = np.sum(np.array(train_label) == 0)
        test_label_num_1 = np.sum(np.array(test_label) == 1)
        test_label_num_0 = np.sum(np.array(test_label) == 0)
        logs = "\nAPPNAME:{}".format(self.app_name) + \
           "\nalert to predict:{}".format(self.target) + \
           "\nstep_size={}min".format(self.step//60) + \
           "\nwindow_size={}h".format(self.window_size//3600) + \
           "\nreact_size={}min".format(self.react_size//60) + \
           "\npositive_range={}h".format(self.positive_range//3600) + \
           "\nmin_log={}".format(self.min_log) + \
           "\ntrain(+):{}".format(train_label_num_1) + \
           "\ntrain(-):{}".format(train_label_num_0) + \
           "\ntest(+):{}".format(test_label_num_1) + \
           "\ntest(-):{}".format(test_label_num_0)
        write_log(logs)
        return train_data, train_label, test_data, test_label
