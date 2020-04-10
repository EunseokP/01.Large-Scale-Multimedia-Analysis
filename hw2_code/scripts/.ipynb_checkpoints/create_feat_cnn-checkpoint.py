#!/bin/python
import glob
import pandas as pd
import numpy as np
import os
import torch
# import cPickle
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
from lightgbm import LGBMClassifier
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0]))
        print("cnn_file_list -- path to cnn file: e.g.) ./cnn")
        print("features_num -- number of cluster: e.g.) 1280")

        exit(1)
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    
    cnn_file_list = sys.argv[1]
    n_features = sys.argv[2]

    path = os.path.join(cnn_file_list, '*.pkl')
    
    filelist = []

    for file in glob.glob(path):
        filelist.append(file)

    #####################################################
    ### To reduce time consumping, skipping this step ###        
    #knn_data_list = []
    #start_time = time.time()
    #for i in range(len(filelist)):
    ## for i in range(2500, 20000):
    #    with open(filelist[i], 'rb') as f:
    #        try:
    #            data = pickle.load(f)
    #            if len(data) != 0:
    #                list_foo = []
    #                foo = pickle.load(open(filelist[i],"rb"))
    #                for idx in range(len(foo)):
    ##                 for idx in srs_idx:
    #                    list_foo.append(foo[idx].detach().numpy())
    #                foo_concat = np.concatenate(list_foo, axis = 0)
    #                knn_data_list.append(foo_concat.mean(axis = 0).reshape(1, n_features))
    #            else:
    #                foo_concat = np.zeros((1,1280))
    #                knn_data_list.append(foo_concat)
    #        except EOFError:
    #            pass
    #    if i % 500 == 0:
    #        print('========== {}th step is completed! =========='.format(i))
    #    else:
    #        pass
    #print('========== All complete! It takes {} secs. =========='.format(time.time() - start_time))           
    #####################################################
    #####################################################
    
    #####################################################
    ### To reduce time consumping, skipping this step ###
    #knn_mart = np.concatenate(knn_data_list, axis = 0)
    #data_mart = pd.DataFrame(knn_mart)
    #data_mart.to_csv('./total_cnn_features_avgpool.csv', index=False)
    #####################################################
    #####################################################

    total_features = pd.read_csv('./total_cnn_features_avgpool.csv')
    #total_features.head(3)        
    
    video_name_ind = []
    for i in range(len(filelist)):
        match_front = re.search('cnn/', filelist[i])
        match_end = re.search('.pkl', filelist[i])
        video_name_ind.append(filelist[i][match_front.end():match_end.start()])
        video_name = pd.DataFrame({'video': video_name_ind})          
        
    k = int(n_features)
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

    total_mart.to_csv('./datamart_cnn_total_avgpool.csv', index=False)
    train_mart.to_csv('./datamart_cnn_train_avgpool.csv', index=False)
    valid_mart.to_csv('./datamart_cnn_valid_avgpool.csv', index=False)
    test_mart.to_csv('./datamart_cnn_test_avgpool.csv', index=False)    
    
    print("========== CNN features generated successfully! ==========")