#!/bin/python
import glob
import pandas as pd
import numpy as np
# import scipy.sparse 
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import re
import pickle
import time
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import warnings
from pandas.core.common import SettingWithCopyWarning

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
        print "vocab_file -- path to the vocabulary file"
        print "file_list -- the list of videos"
        exit(1)
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    path = '../asrs/*.txt'

    filelist = []

    for file in glob.glob(path):
        filelist.append(file)
        
    def concatenate_list_data(list):
        result= ''
        for element in list:
            result += str(element)
        return result        

    text = []
    for i in range(len(filelist)):
    # for i in range(10):
        with open (filelist[i], "r") as myfile:
            data=myfile.readlines()
            data=concatenate_list_data(data)
        text.append(data)
        if i % 1000 == 0:
            print('{}th txt file reading...'.format(i))
        else: pass    

    vect = CountVectorizer(stop_words="english")
    # vect = CountVectorizer()
    bow = vect.fit_transform(text).toarray()
    norm_bow = normalize(bow, norm = 'l1', axis=1)
    norm_data = pd.DataFrame(norm_bow)
    #print norm_data.shape
    norm_data.to_csv('./total_asrfeat.csv', index = False)
    
    norm_data = pd.read_csv('./total_asrfeat.csv')
    norm_data.head(3)
        
    video_name_ind = []
    for i in range(len(filelist)):
        match_front = re.search('asrs/', filelist[i])
        match_end = re.search('.txt', filelist[i])
        video_name_ind.append(filelist[i][match_front.end():match_end.start()])
        video_name = pd.DataFrame({'video': video_name_ind})    
        
    # Making features columns
    k = norm_data.shape[1]
    column_name = ['video']
    for i in range(k):
        column_name.append('feature_{}'.format(i))

    total_data = pd.concat([video_name, norm_data], axis = 1)
    total_data.columns = column_name

    train_ind = pd.read_csv('./list/train', sep = ' ', header = None)
    valid_ind = pd.read_csv('./list/val', sep = ' ', header = None)
    test_ind = pd.read_csv('./list/test.video', sep = ' ', header = None)

    train_ind['Data'] = 'TRAIN'
    valid_ind['Data'] = 'VALID'
    test_ind[1] = 'UNK'
    test_ind['Data'] = 'TEST'

    train_ind.columns = ['video','target','Data']
    valid_ind.columns = ['video','target','Data']
    test_ind.columns = ['video','target','Data']

    data_lable = pd.concat([train_ind, valid_ind, test_ind], axis = 0).reset_index().drop('index', axis = 1)
    # data_lable['target_p001'] = 
    data_lable['target_p001'] = data_lable['target']
    data_lable['target_p002'] = data_lable['target']
    data_lable['target_p003'] = data_lable['target']
    data_lable['target_p001_10'] = 1
    data_lable['target_p002_10'] = 1
    data_lable['target_p003_10'] = 1

    data_lable['target_p001'][data_lable['target'] != 'P001'] = 'Other'
    data_lable['target_p002'][data_lable['target'] != 'P002'] = 'Other'
    data_lable['target_p003'][data_lable['target'] != 'P003'] = 'Other'
    data_lable['target_p001_10'][data_lable['target'] != 'P001'] = 0
    data_lable['target_p002_10'][data_lable['target'] != 'P002'] = 0
    data_lable['target_p003_10'][data_lable['target'] != 'P003'] = 0

    total_mart = total_data.merge(data_lable, how = 'right', on = 'video')
    total_mart = total_mart.fillna(0)
    
    train_mart = total_mart[total_mart['Data'] == 'TRAIN']
    valid_mart = total_mart[total_mart['Data'] == 'VALID']
    test_mart  = total_mart[total_mart['Data'] == 'TEST']
    
    total_mart.to_csv('./datamart_total_asrfeat_{}.csv'.format(k), index=False)
    train_mart.to_csv('./datamart_train_asrfeat_{}.csv'.format(k), index=False)
    valid_mart.to_csv('./datamart_valid_asrfeat_{}.csv'.format(k), index=False)
    test_mart.to_csv('./datamart_test_asrfeat_{}.csv'.format(k), index=False)
    
        
    print "ASR features generated successfully!"
