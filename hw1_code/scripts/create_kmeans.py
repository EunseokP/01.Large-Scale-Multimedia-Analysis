#!/bin/python
import glob
import pandas as pd
import numpy as np
import os
import cPickle
import pickle
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.cluster.k_means_ import KMeans
import sys
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
import re
import time
import warnings
from pandas.core.common import SettingWithCopyWarning

# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    kmeans_model = './'+sys.argv[1]+'.sav'; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    # kmeans_model = './{}.sav'.format(sys.argv[1])
    kmeans = pickle.load(open(kmeans_model,"rb"))
    model = kmeans
    
    # path = './hw1_git/11775-hws/videos/*.mp4'
    path = './mfcc/*.csv'

    filelist = []

    for file in glob.glob(path):
        filelist.append(file)
        
    def get_features(k, model, path_list):
        loaded_model= model
        start_time = time.time()
        features_dict = dict()
        filelist = path_list
        for i in range(len(filelist)):
    #     for i in range(10):        
            if i % 1000 == 0: 
                print('{}th step progressing....'.format(i)) 
            else: 
                pass
            data = pd.read_csv(filelist[i], sep = ';', header = None)
            pred_centers = loaded_model.predict(data)
            num_clusters = k
            bow_preds = np.zeros((1, num_clusters))

            for ind in pred_centers:
                bow_preds[0, ind] += 1
            norm_feat = (1.0 * bow_preds)/np.sum(bow_preds)
            features_dict[i] = pd.DataFrame(norm_feat)

        features_total = features_dict[0].copy()
        for i in range(1, len(features_dict)):
            foo = features_dict[i].copy()
            features_total = pd.concat([features_total, foo], axis = 0)
            features_total = features_total.reset_index().drop('index', axis = 1)

        print("===== The time consuming of getting features : {} seconds =====".format((time.time() - start_time)))
        return features_total  
    
#    total_features = get_features(k = cluster_num, model = model, path_list = filelist)    
#    total_features.to_csv('./total_features_k{}.csv'.format(cluster_num), index=False)
    
    total_features = pd.read_csv('./total_features_k{}.csv'.format(cluster_num))
    total_features.head(3)

    video_name_ind = []
    for i in range(len(filelist)):
        match_front = re.search('mfcc/', filelist[i])
        match_end = re.search('.mfcc.csv', filelist[i])
        video_name_ind.append(filelist[i][match_front.end():match_end.start()])
        video_name = pd.DataFrame({'video': video_name_ind})    
    
    # Making features columns
    k = cluster_num
    column_name = ['video']
    for i in range(k):
        column_name.append('feature_{}'.format(i))

    total_data = pd.concat([video_name, total_features], axis = 1)
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
    
    total_mart.to_csv('./datamart_total_k{}.csv'.format(cluster_num), index=False)
    train_mart.to_csv('./datamart_train_k{}.csv'.format(cluster_num), index=False)
    valid_mart.to_csv('./datamart_valid_k{}.csv'.format(cluster_num), index=False)
    test_mart.to_csv('./datamart_test_k{}.csv'.format(cluster_num), index=False)
    print "K-means features generated successfully!"
