# coding: utf-8
import numpy as np
import pandas as pd
from utils.split_data import split_data
from utils.write_logs import write_log
import re


class Prefix:
    
    def __init__(self, app_name='', data_name='data.csv', target='',alert_level = 1):
        df = pd.read_csv(data_name)
        self.app_name = app_name
        self.target = target
        self.alert_level = alert_level
        self.df = df[df['N_APPNAME'] == self.app_name]
        self.datas = []
        self.labels = []

    def keyword(self, df, keyword, if_true=True):
        pattern = re.compile('.*' + keyword + '.*')
        if (pattern.match(df["N_SUMMARYCN"]) is not None) and (df['N_CUSTOMERSEVERITY'] == self.alert_level):
            return if_true
        else:
            return not if_true

    def sample(self, step=10, window_size=60, react_size=10, positive_range=120, min_log=5):
        self.step = step * 60
        self.window_size = window_size * 60
        self.react_size = react_size * 60
        self.positive_range = positive_range * 60
        self.min_log = min_log
        self.data_time = []
        datas = []
        labels = []
        start_stamp = self.df['firsttimestamp'].min()
        end_stamp = self.df['firsttimestamp'].max()
        for i in range(start_stamp, (end_stamp - self.window_size - self.react_size - self.positive_range), self.step):
            temp = self.df[(self.df['firsttimestamp'] >= i) & (self.df['firsttimestamp'] < (i + self.window_size))]
            if temp.shape[0] < self.min_log:
                continue
            else:
                if temp[(temp.apply(self.keyword, keyword=self.target, axis=1))].shape[0]:
                    temp = temp[(temp.apply(self.keyword, keyword=self.target, if_true=False, axis=1))]
                tmp = temp['N_SUMMARYCN'].values
                datas.append(list(tmp))
                future = self.df[(self.df['firsttimestamp'] >= (i + self.window_size + self.react_size)) & (
                            self.df['firsttimestamp'] <= (
                                i + self.window_size + self.react_size + self.positive_range))]
                self.data_time.append(i + self.window_size)
                if future.shape[0]==0:
                    labels.append(0)
                else:
                    if future[future.apply(self.keyword, keyword=self.target, axis=1)].shape[0]:
                        labels.append(1)
                    else:
                        labels.append(0)

        self.datas = datas
        self.labels = labels
        print("---sample done---")

    def split_data(self, split_percent=0.7):
        split_timestamp = self.data_time[int(len(self.data_time) * split_percent)]
        train_df = self.df[self.df['firsttimestamp'] < split_timestamp]
        test_df = self.df[self.df['firsttimestamp'] >= split_timestamp]

        self.train_alert_num = train_df[train_df.apply(self.keyword, keyword=self.target, axis=1)].shape[0]
        self.test_alert_num = test_df[test_df.apply(self.keyword, keyword=self.target, axis=1)].shape[0]
        
        train_data, train_label, test_data, test_label = split_data(self.datas, self.labels, split_percent)
        train_label_num_1 = np.sum(np.array(train_label) == 1)
        train_label_num_0 = np.sum(np.array(train_label) == 0)
        test_label_num_1 = np.sum(np.array(test_label) == 1)
        test_label_num_0 = np.sum(np.array(test_label) == 0)
        
        logs = "\nAPPNAME:{}".format(self.app_name) + \
           "\nalert to predict:{}".format(self.target) + \
           "\ntraining={}".format(self.train_alert_num) + \
           "\ntesting={}".format(self.test_alert_num) + \
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
