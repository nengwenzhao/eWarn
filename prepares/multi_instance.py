# coding: utf-8

import numpy as np
from utils.write_logs import write_log
from prepares.prefix_extra_features import Prefix_extra_features
from utils.split_data import split_data
from datetime import datetime
from functools import reduce



class Multi_instance(Prefix_extra_features):

    def __init__(self, app_name='', data_name='data.csv', target='',alert_level = 1):
        super(Multi_instance, self).__init__(app_name, data_name, target,alert_level)
        self.levels = self.df['N_CUSTOMERSEVERITY'].max()
        
        
    def sample(self, step = 10, instance_size=30, bag_size=60, react_size=10, positive_range=120, min_log=3):
        
        self.step = step * 60
        self.react_size = react_size * 60
        self.instance_size = instance_size * 60
        self.bag_size = bag_size * 60
        self.positive_range = positive_range * 60
        self.min_log = min_log
        # bin_size = 600

        bags = []
        labels = []
        start_stamp = self.df['firsttimestamp'].min()
        end_stamp = self.df['firsttimestamp'].max()
        
        extra_features = []


        
        for now_stamp in range(start_stamp, end_stamp, self.step):
            temp = self.df[(self.df['firsttimestamp'] >= now_stamp) & (
                        self.df['firsttimestamp'] < (now_stamp + self.bag_size))]

            if temp.shape[0] < self.min_log:
                continue
            else:
                if temp[(temp.apply(self.keyword, keyword=self.target, axis=1))].shape[0]:
                    temp = temp[(temp.apply(self.keyword, keyword=self.target, if_true=False, axis=1))]

                bag = []
                extra_feature_bag = []
                
                for inner_stamp in range(now_stamp, now_stamp + self.bag_size - self.instance_size, self.step):
                    instance = temp[(temp['firsttimestamp'] >= inner_stamp) & (temp['firsttimestamp'] < (inner_stamp + self.instance_size))]
                    
                    instance_summary = list(instance['N_SUMMARYCN'].values)
                    instance_summary = " ".join(instance_summary)
                    extra_feature_bag.append(self.get_extra_feature(instance, inner_stamp, self.instance_size))
                    bag.append(instance_summary)
                
                instance = temp[(temp['firsttimestamp'] >= now_stamp + self.bag_size - self.instance_size) & (temp['firsttimestamp'] < (now_stamp + self.bag_size))]
                    
                instance_summary = list(instance['N_SUMMARYCN'].values)
                instance_summary = " ".join(instance_summary)
                extra_feature_bag.append(self.get_extra_feature(instance, now_stamp + self.bag_size - self.instance_size, self.instance_size))
                bag.append(instance_summary)
                
                extra_features.append(extra_feature_bag)
                bags.append(bag)
                
                future = self.df[(self.df['firsttimestamp'] >= (now_stamp + self.bag_size + self.react_size)) & (
                            self.df['firsttimestamp'] <= (
                                now_stamp + self.bag_size + self.react_size + self.positive_range))]
                if future.shape[0]==0:
                    labels.append(0)
                else:
                    if future[future.apply(self.keyword, keyword=self.target, axis=1)].shape[0]:
                        labels.append(1)
                    else:
                        labels.append(0)

        self.datas = bags
        self.labels = labels
        self.extra_features = extra_features
        print("---sample done---")

    def split_data(self, split_percent=0.7):
        
        train_data, train_label, test_data, test_label = split_data(self.datas, self.labels, split_percent)
        train_extra_features, train_label, test_extra_features, test_label = split_data(self.extra_features, self.labels, split_percent)
        train_label_num_1 = np.sum(np.array(train_label) == 1)
        train_label_num_0 = np.sum(np.array(train_label) == 0)
        test_label_num_1 = np.sum(np.array(test_label) == 1)
        test_label_num_0 = np.sum(np.array(test_label) == 0)
        logs = "\n--------------\n" + "APPNAME:{}".format(self.app_name) + \
           "\nalert to predict:{}".format(self.target) + \
           "\ninstance_size={}min".format(self.instance_size//60) + \
           "\nreact_size={}min".format(self.react_size//60) + \
           "\nbag_size={}min".format(self.bag_size//60) + \
           "\npositive_range={}min".format(self.positive_range//60) + \
           "\nmin_log={}".format(self.min_log) + \
           "\ntrain(+): {}".format(train_label_num_1) + \
           "\ntrain(-): {}".format(train_label_num_0) + \
           "\ntest(+): {}".format(test_label_num_1) + \
           "\ntest(-): {}".format(test_label_num_0)
        write_log(logs)
        
        return train_data, train_label, test_data, test_label, train_extra_features, test_extra_features
