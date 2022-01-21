# -*- coding: UTF-8 -*-

import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils.write_logs import write_log
import warnings

warnings.filterwarnings('ignore')


class LDA_model:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def lda_features(self, n_topics, doc_topic_prior, random_state, train_data, test_data):
        # 20 clusters; 30 topic
        lda_value = []
        corpus_train = []
        corpus_test = []
        for index, sample in enumerate(train_data):
            alert_token = ""
            ch_alert = []
            for i in sample:
                temp = re.sub('[0-9\!\%\[\]\,\。]', ' ', i)
                #temp = re.sub(r'[^\w\s]', '', temp).replace(' ', '')
                temp = re.sub(r'[^\w\s]', '', temp)
                temp = temp.replace('_', '')
                ch_alert.append(temp)

            for i in ch_alert:
                seg_list = jieba.cut(i, cut_all=False)
                seg_list = list(filter(lambda x:x!=' ',list(seg_list)))
                seg_list = " ".join(seg_list)
                alert_token += seg_list
            corpus_train.append(alert_token)

        for index, sample in enumerate(test_data):
            alert_token = ""
            ch_alert = []
            for i in sample:
                temp = re.sub('[0-9\!\%\[\]\,\。]', ' ', i)
                #temp = re.sub(r'[^\w\s]', '', temp).replace(' ', '')
                temp = re.sub(r'[^\w\s]', '', temp)
                temp = temp.replace('_', '')
                ch_alert.append(temp)

            for i in ch_alert:
                seg_list = jieba.cut(i, cut_all=False)
                seg_list = list(filter(lambda x:x!=' ',list(seg_list)))
                seg_list = " ".join(seg_list)
                alert_token += seg_list
            corpus_test.append(alert_token)
            
        #stopwords_path = 'datas/pre_data/stopwords.txt'
        #with open(stopwords_path) as fp:
        #    stopword = fp.readlines()
        #stopword = list(map(lambda x:x[:-1].strip(),stopword))
        
        tfidfVector = TfidfVectorizer()
        vocab = tfidfVector.fit(corpus_train)
        stopword = list(map(lambda x:tfidfVector.get_feature_names()[x],tfidfVector.idf_.argsort()))[:10]
        
        cntVector = CountVectorizer()
        #cntVector =  TfidfVectorizer(stop_words = stopword)
        vocaubulary = cntVector.fit(corpus_train)
        cntTf1 = cntVector.transform(corpus_train)
        cntTf2 = cntVector.transform(corpus_test)
        lda = LatentDirichletAllocation(n_components=n_topics, doc_topic_prior=doc_topic_prior, random_state=random_state)
        lda_train = lda.fit(cntTf1)
        train_feature = lda_train.transform(cntTf1)
        test_feature = lda_train.transform(cntTf2)
        self.lda_model = lda_train
        self.cntTf1 = cntTf1
        self.cntTf2 = cntTf2
        self.corpus_train = corpus_train
        self.corpus_test = corpus_test
        self.feature_names = cntVector.get_feature_names()

        return train_feature, test_feature

    def lda_feature_extraction(self, n_topics=30, doc_topic_prior=None, random_state=0):
        self.train_feature, self.test_feature = self.lda_features(n_topics, doc_topic_prior, random_state,
                                                                  self.train_data, self.test_data)
        print("---LDA feature process done---")
        return self.train_feature, self.test_feature
    
    def print_feature_name(self, n_top_words = 10):
        for topic_idx,topic in enumerate(self.lda_model.components_):
            print("Topic {}:".format(topic_idx))
            print(" ".join([self.feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
            
    def best_topic_coherence(self, n_topics = 30):
        train_feature, test_feature = self.lda_features(10, None, 0,self.train_data, self.test_data)
        coherences = []
        for i in range(2, n_topics+2):
            texts = list(map(lambda x:x.split(),self.corpus_train))
            id2word = corpora.Dictionary(texts)
            corpus = [id2word.doc2bow(text) for text in texts]
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus ,
                                               id2word=id2word,
                                               num_topics=i, 
                                               random_state=0,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
            coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            coherences.append(coherence_lda)
        #write_log("\nconherences:{}".format(max(coherences))+",topics_num:{} \n".format(coherences.index(max(coherences))+2))
        return (coherences.index(max(coherences))+2)
    
    def best_topic_perplexity(self, n_topics = 30):
        perplexitys = []
        for i in range(2, n_topics+2):
            lda = LatentDirichletAllocation(n_components=n_topics, doc_topic_prior=None, random_state=0)
            lda.fit(self.cntTf1)
            perplexity = lda.perplexity(self.cntTf1)
            perplexitys.append(perplexity)
        write_log("\nconherences:{}".format(min(perplexitys))+",topics_num:{} \n".format(perplexitys.index(min(perplexitys))+2))
        return (perplexitys.index(min(perplexitys))+2)
    
    
    def get_feature_name(self, n_top_words = 10):
        res = []
        for topic_idx,topic in enumerate(self.lda_model.components_):
            res.append(" ".join([self.feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
        return res
    
    def write_feature_name(self, n_top_words = 10):
        res = []
        for topic_idx,topic in enumerate(self.lda_model.components_):
            res.append("Topic {}: ".format(topic_idx)+" ".join([self.feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
        write_log("\n" + "\n".join(res))

        
        
        
