# -*- coding: UTF-8 -*-

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from classifiers.Adacost import AdaCostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import xgboost as xgb
import numpy as np

from utils.write_logs import write_log

def rf_classifier(train_feature, test_feature, train_label, weight=None):
    rf = RandomForestClassifier(n_estimators=200, class_weight=weight, random_state=0)
    rf.fit(train_feature, train_label)
    
    log = '\n rf_key_features: '+' '.join(list(map(str, rf.feature_importances_.argsort()[::-1][:5])))
    write_log(log)
    print("---Classifier process done---")
    return rf.predict_proba(test_feature)[:, 0]

def rf_regressor(train_feature, test_feature, train_label):
    rf = RandomForestRegressor(n_estimators=200, random_state=0)
    rf.fit(train_feature, train_label)
    
    log = '\n rf_key_features: '+' '.join(list(map(str, rf.feature_importances_.argsort()[::-1][:5])))
    write_log(log)
    print("---Classifier process done---")
    predict = rf.predict(test_feature)
    predict = [(1-(x - min(predict))/(max(predict) - min(predict))) for x in predict]
    return predict



def rf_select_classifier(train_feature, test_feature, train_label, weight="balanced"):
    selector = SelectFromModel(ExtraTreesClassifier(n_estimators = 200, class_weight=weight, random_state=0, bootstrap=True))
    selector.fit(train_feature, train_label)
    train_feature = selector.transform(train_feature)
    test_feature = selector.transform(test_feature)
    print(train_feature.shape)
    print(test_feature.shape)
    rf = RandomForestClassifier(n_estimators=200, class_weight=weight, random_state=0)
    rf.fit(train_feature, train_label)
    print("---Classifier process done---")
    return rf.predict_proba(test_feature)[:, 0]

def linear_select(train_feature, test_feature, train_label):
    train_feature_positive = train_feature[np.array(train_label)==1]
    train_feature_negative = train_feature[np.array(train_label)==0]
    w = np.zeros(train_feature.shape[1])
    for i in range(200):
        index = resample(list(range(train_feature_negative.shape[0])), n_samples=int(0.1*train_feature_negative.shape[0]), random_state=i)
        sample_feature = np.concatenate((train_feature_positive,train_feature_negative[index]),axis = 0)
        sample_label = np.concatenate((np.array(train_label)[np.array(train_label)==1],np.array(train_label)[index]),axis = 0)
        clf = LogisticRegression(random_state=0).fit(sample_feature, sample_label )
        w+=clf.coef_[0]
    w = np.array(list(map(lambda x:abs(x),w)))
    train_feature = train_feature[:,w.argsort()[::-1][:60]]
    test_feature = test_feature[:,w.argsort()[::-1][:60]]
    
    log = '\n select_feature: ' + ' '.join(list(map(str, w.argsort()[::-1][:60])))
    write_log(log)
    
    return train_feature, test_feature

def erf_classifier(train_feature, test_feature, train_label, weight="balanced"):
    rf = ExtraTreesClassifier(n_estimators=200, class_weight=weight, random_state=0, bootstrap=True)
    rf.fit(train_feature, train_label)
    print("---Classifier process done---")
    return rf.predict_proba(test_feature)[:, 0]


def rf_key_topic(lda_topics,train_feature, test_feature, train_label, weight="balanced"):
    rf = RandomForestClassifier(n_estimators=200, class_weight=weight, random_state=0)
    rf.fit(train_feature, train_label)
    topics = '\n'+'\n'.join(list(map(lambda x:lda_topics[x], rf.feature_importances_.argsort()[::-1][:3])))
    write_log(topics)

def rf_mul_classifier(train_feature, test_feature, train_label, weight="balanced"):
    rf = RandomForestClassifier(n_estimators=100, class_weight=weight, random_state=0)
    rf.fit(train_feature, train_label)
    print("---Classifier process done---")
    return list(map(lambda bag: rf.predict_proba(bag)[:, 0],test_feature))

def adacost_classifier(train_feature, test_feature, train_label, beta):
    rf = AdaCostClassifier(n_estimators=100, random_state=0,beta = beta)
    rf.fit(train_feature, train_label)
    print("---Classifier process done---")
    return rf.predict_proba(test_feature)[:, 0]

def rf_classifier_grid(train_feature, test_feature, train_label, test_label):
    n_estimators = list(range(10,300,10)) 
    class_weight=['balanced']
    random_state = [0]
    rf = RandomForestClassifier()
    param_grid = dict(n_estimators = n_estimators, class_weight= class_weight)
    kflod = StratifiedKFold(n_splits=7, shuffle = False,random_state=0)
    grid_search = GridSearchCV(rf,param_grid,scoring = 'roc_auc',n_jobs = -1,cv = kflod)

    grid_result = grid_search.fit(train_feature, train_label)
    rf = RandomForestClassifier(n_estimators = grid_search.best_params_['n_estimators'],class_weight = grid_search.best_params_['class_weight'],random_state= 0)
    rf.fit(train_feature, train_label)
    print("---Classifier process done---")
    return rf.predict_proba(test_feature)[:, 0]

def xgb_classifier(train_feature, test_feature, train_label):
    dtrain = xgb.DMatrix(train_feature, label = train_label)
    dtest = xgb.DMatrix(test_feature)
    params = {'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':10,
        'lambda':2,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':3,
        'eta': 0.025,
        'seed':0,
        'nthread':8,
         'silent':1}

    bst = xgb.train(params,dtrain, num_boost_round=200)
    result_proba = bst.predict(dtest)
    result_proba = np.array(list(map(lambda x:(1-x), result_proba)))
    print("---Classifier process done---")
    return result_proba