# coding: utf-8


import numpy as np
from datetime import datetime
from utils.write_logs import write_log
from prepares.prefix import Prefix
from utils.split_data import split_data
from functools import reduce


class Prefix_extra_features(Prefix):

    def __init__(self, app_name='', data_name='data.csv', target='',alert_level = 1):
        super(Prefix_extra_features, self).__init__(app_name, data_name, target,alert_level)
        self.levels = self.df['N_CUSTOMERSEVERITY'].max()
    
    def get_extra_feature(self, temp, time_index, window_size):
        levels = temp['N_CUSTOMERSEVERITY'].values
        level_features = []  #告警等级个数
        duration_features = [] #有效告警持续时间
        hour_features = []  # 窗口起始点的小时

        weekday_features = [] #窗口起始点处于一个星期的哪一天

        is_weekend_features = [] #是否是周末

        def func(x,length):
            p = np.zeros(length)
            p[x] = 1
            return p
        map_temp = list(map(lambda x :func(x,self.levels+1),levels))
        if len(map_temp)==0:
            zero = np.zeros(self.levels+1)
            zero[0] = 1 
            level_features.append(zero)
        else:
            level_features.append(reduce(lambda x,y: x+y,map_temp))

        if not temp.shape[0]:
            duration_features.append([0])
        else:
            duration = abs(max(temp['firsttimestamp'].tolist()) - min(temp['firsttimestamp'].tolist()))/window_size
            duration_features.append([duration])

        date = datetime.fromtimestamp(time_index)

        hour = date.hour
        day = date.day
        #month = date.month
        week = date.weekday()
        hour_feature = np.zeros(24)
        hour_feature[hour] = 1
        day_feature = np.zeros(31)
        day_feature[day - 1] = 1
        #month_feature = np.zeros(12)
        #month_feature[month-1] = 1
        week_feature = np.zeros(7)
        week_feature[week] = 1
        weekend_feature = np.zeros(2)
        if week>=5:
            weekend_feature[1] = 1
        else:
            weekend_feature[0] = 1
        hour_features.append(hour_feature)
        #monthday_features.append(day_feature)
        weekday_features.append(week_feature)
        #month_features.append(month_feature)
        is_weekend_features.append(weekend_feature)
        extra_features = list(map(lambda x1,x2,x3,x4,x5:np.concatenate((x1,x2,x3,x4,x5)),level_features,duration_features,hour_features,weekday_features,is_weekend_features))
        return extra_features[0]
        
    
    def sample(self, step=10, window_size=120, react_size=10, positive_range=360, min_log=5):
        self.step = step * 60
        self.window_size = window_size * 60
        self.react_size = react_size * 60
        self.positive_range = positive_range * 60
        self.min_log = min_log
        self.data_time = []
        datas = []
        labels = []
        
        extra_features = []
                
        start_stamp = self.df['firsttimestamp'].min()
        end_stamp = self.df['firsttimestamp'].max()
        for i in range(start_stamp, (end_stamp - self.window_size - self.react_size - self.positive_range), self.step):
            temp = self.df[(self.df['firsttimestamp'] >= i) & (self.df['firsttimestamp'] < (i + self.window_size))]
            if temp.shape[0] < self.min_log:
                continue
            else:
                if temp[(temp.apply(self.keyword, keyword=self.target, axis=1))].shape[0]:
                    temp = temp[(temp.apply(self.keyword, keyword=self.target, if_true=False, axis=1))]
                #temp = temp[(temp['N_CUSTOMERSEVERITY'] != 1)]
                extra_features.append(self.get_extra_feature(temp, i, window_size = self.window_size))
                tmp = temp['N_SUMMARYCN'].values
                tmp = list(np.unique(tmp))
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
        self.extra_features = extra_features
        print("---sample done---")

    def split_data(self, split_percent=0.7):
        split_timestamp = self.data_time[int(len(self.data_time) * split_percent)]
        train_df = self.df[self.df['firsttimestamp'] < split_timestamp]
        test_df = self.df[self.df['firsttimestamp'] >= split_timestamp]

        self.train_alert_num = train_df[train_df.apply(self.keyword, keyword=self.target, axis=1)].shape[0]
        self.test_alert_num = test_df[test_df.apply(self.keyword, keyword=self.target, axis=1)].shape[0]
        
        train_data, train_label, test_data, test_label = split_data(self.datas, self.labels, split_percent)
        train_extra_features, train_label, test_extra_features, test_label = split_data(self.extra_features, self.labels, split_percent)
        
        
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
        return train_data, train_label, test_data, test_label, train_extra_features, test_extra_features
        