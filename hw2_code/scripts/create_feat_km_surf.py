#!/bin/python
import glob
import pandas as pd
import numpy as np
import os
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

# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0]))
        print("kmeans_model -- path to the kmeans model: e.g.) ./kmeans")
        print("cluster_num -- number of cluster: e.g.) 100")
        print("surf_file_list -- path to surf file: e.g.) ./surf")
        exit(1)
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    
    kmeans_model = sys.argv[1]+'/knn_{}_model.sav'.format(sys.argv[2])
    cluster_num = int(sys.argv[2])
    surf_file_list = sys.argv[3]

    path = os.path.join(surf_file_list, '*.pkl')
    
    filelist = []

    for file in glob.glob(path):
        filelist.append(file)
        
    def get_features(k, model, path_list, n_frames = 500):
        loaded_model = model
        k = k
        start_time = time.time()
        features_list = []
        filelist = path_list
        except_list = ['./surf/HVC5510.pkl', './surf/HVC721.pkl', './surf/HVC3124.pkl']
        for i in range(len(filelist)):
    #     for i in range(100):
            if filelist[i] in except_list:
                array_data = np.zeros((1, 64))
            else:
                with open(filelist[i], 'rb') as f:
                        try:
                            data = pickle.load(f)
                            if len(data) != 0:
                                foo_list = []
                                if len(data) >= 500:                            
                                    rsr_idx = np.random.choice(len(data), n_frames)
                                    for idx in rsr_idx:
                                        if len(data[idx].shape) == 3:
                                            foo_list.append(data[idx][0])
                                        else:
                                            pass
                                else:
                                    for idx in range(len(data)):
                                        if len(data[idx].shape) == 3:
                                            foo_list.append(data[idx][0])
                                        else:
                                            pass
                            else:
                                print('{}th Pickle is Empty!! Skip it!!'.format(i))                    
                            array_data = np.concatenate(foo_list, axis = 0)
                        except EOFError:
                            array_data = np.zeros((1, 64))

            pred_centers = loaded_model.predict(array_data)
            num_clusters = k
            bow_preds = np.zeros((1, num_clusters))

            for ind in pred_centers:
                bow_preds[0, ind] += 1
            norm_feat = (1.0 * bow_preds)/np.sum(bow_preds)
            features_list.append(norm_feat)
            if i % 100 == 0:
                print('{}th step is progressing!!!'.format(i))
            else: pass
        print("===== The time consuming of getting features : {} seconds =====".format((time.time() - start_time)))    
        return features_list

    # load the kmeans model
    # kmeans_model = './{}.sav'.format(sys.argv[1])
    model = pickle.load(open(kmeans_model, "rb"))

    #####################################################
    ### To reduce time consumping, skipping this step ###
    foo = get_features(cluster_num, model, filelist, n_frames = 100)
    data_array = np.concatenate(foo, axis = 0)
    data_mart = pd.DataFrame(data_array)
    data_mart.to_csv('./total_surf_features_k{}.csv'.format(cluster_num))
    #####################################################
    #####################################################

    total_features = pd.read_csv('./total_surf_features_k{}.csv'.format(cluster_num))
    #total_features.head(3)    
    
    path = './surf/*.pkl'
    filelist = []
    for file in glob.glob(path):
        filelist.append(file)    

    video_name_ind = []
    for i in range(len(filelist)):
        match_front = re.search('surf/', filelist[i])
        match_end = re.search('.pkl', filelist[i])
        video_name_ind.append(filelist[i][match_front.end():match_end.start()])
        video_name = pd.DataFrame({'video': video_name_ind})          
        
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

    total_mart.to_csv('./datamart_surf_total_k{}.csv'.format(cluster_num), index=False)
    train_mart.to_csv('./datamart_surf_train_k{}.csv'.format(cluster_num), index=False)
    valid_mart.to_csv('./datamart_surf_valid_k{}.csv'.format(cluster_num), index=False)
    test_mart.to_csv('./datamart_surf_test_k{}.csv'.format(cluster_num), index=False)        
    print("========== K-means features generated successfully! ==========")