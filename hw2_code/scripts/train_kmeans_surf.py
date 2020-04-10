#!/bin/python 

import glob
import pandas as pd
import numpy as np
import os
import re
# import cPickle
import pickle
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.cluster.k_means_ import KMeans
import sys
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
# import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
import time

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0]))
        print("surf_file_path -- path to the surf pkl file: e.g.) ./surf/*.pkl")
        print("cluster_num -- number of cluster: e.g.) 100")
        print("output_file -- path to save the k-means model: e.g.) ./kmeans")
        exit(1)

    surf_file_path = sys.argv[1] 
    cluster_num = int(sys.argv[2])
    output_file = sys.argv[3]
    
    path = os.path.join(surf_file_path, '*.pkl')
    filelist = glob.glob(path)
    #print(filelist)
    np.random.seed(112)
    n = len(filelist)
    idx = np.random.choice(n, np.int(n*0.1))
    
    ### Sampling the datasets to avoid memory issue
    sample_file = []
    for i in idx:
        sample_file.append(filelist[i])

    # sample_dict = dict()
    n_frames = 20
    sample_list = []
    n_datasets = 100
    print("========== Start to create knn datamart by using {} raw datasets ==========".format(n_datasets))
    # for i in range(len(sample_file)):
    for i in range(n_datasets): ### Only use 100 datasets
        with open(sample_file[i], 'rb') as f:
            try:
                data = pickle.load(f)
                if len(data) != 0:
                    foo_list = []
                    ### Random sampling for 20frames from video
                    rsr_idx = np.random.choice(len(data), n_frames) 
                    for idx in rsr_idx:
                        if len(data[idx].shape) == 3:
                            foo_list.append(data[idx][0])
                        else:
                            pass
                else:
                    print('{}th Pickle is Empty!! Skipping it!!'.format(i))                    
                add = np.concatenate(foo_list, axis = 0)
                sample_list.append(add)
            except EOFError:
                pass
        if i % 50 == 0:
            print('========== {}th step is completed! =========='.format(i))
        else:
            pass
    print("========== Ready to do KNN model by using knn datamart ==========")
    knn_mart = np.concatenate(sample_list, axis = 0)        

    print("========== Start to model knn with {} clusters ==========".format(cluster_num))    
    start_time = time.time()
    n_clusters = cluster_num
    n_init = 5
    model = KMeans(n_clusters = n_clusters, random_state = 0, n_init = n_init, n_jobs = -1).fit(knn_mart)
    filename = os.path.join(output_file, 'knn_{}_model.sav'.format(n_clusters))
    pickle.dump(model, open(filename, 'wb'))
    print('knn-{} model: {} time is consumping!!!'.format(n_clusters, time.time()-start_time))
    print('The model is sotred at here: {}'.format(filename))
    
#     print mfcc_csv_file
#     print output_file
#     print cluster_num
    print("========== K-means trained successfully! ==========")
