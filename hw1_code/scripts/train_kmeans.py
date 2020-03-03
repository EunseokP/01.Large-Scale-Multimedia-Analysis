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
# import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
import time

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
        print "mfcc_csv_file -- path to the mfcc csv file"
        print "cluster_num -- number of cluster"
        print "output_file -- path to save the k-means model"
        exit(1)

    mfcc_csv_file = sys.argv[1]; output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])
    
    train_sample = pd.read_csv(mfcc_csv_file, header=None, sep=';')
    start_time = time.time()
    n_clusters = cluster_num
    n_init = 5
    model = KMeans(n_clusters = n_clusters, random_state = 0, n_init = n_init, n_jobs = -1).fit(train_sample)
    
    filename = output_file+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    print "===== The time consuming of Kmeans clustering : {} seconds =====".format((time.time() - start_time))
#     print mfcc_csv_file
#     print output_file
#     print cluster_num
    print "K-means trained successfully!"
