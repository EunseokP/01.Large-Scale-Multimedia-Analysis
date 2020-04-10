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
import pickle
import time
import os
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        print "model_type -- mfcc or asrs"
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    model_type = sys.argv[5]

    
    if model_type == 'mfcc':
        test_mart  = pd.read_csv('./datamart_test_k{}.csv'.format(feat_dim))
        print('test datamart shape is', test_mart.shape)
        print(output_file)
        #filename = 'mfcc_pred/'+model_file+'.model.sav'
        filename = model_file+'.sav'
        model = pickle.load(open(filename, 'rb'))
        #print(model)
        
        X_test = test_mart.iloc[:,1:feat_dim+1]
        scores = model.predict_proba(X_test)
        scores_df = pd.DataFrame(scores)
        scores_df.columns = ['N', 'Y']
        scores_total = pd.concat([test_mart, scores_df], axis = 1)
        #scores_total.head(3)
        
        test_list = pd.read_csv('./list/test.video', header = None)
        test_list.columns = ['video']
        #test_list.head(3)        
        
        final_list = test_list.merge(scores_total, how = 'left', on = 'video')[['video', 'Y']]
        final_list.to_csv(output_file+'.video'+'.csv', index = False)
        final_list['Y'].to_csv(output_file+'.csv', index = False, header = True)
        final_list.to_csv(output_file+'.video'+'.txt', index = False, header = False)
        final_list['Y'].to_csv(output_file+'.txt', index = False, header = False)
        
    elif model_type == 'asrs':
        test_mart  = pd.read_csv('./datamart_test_asrfeat_{}.csv'.format(feat_dim))
        print('test datamart shape is', test_mart.shape)
        #filename = 'asr_pred/'+model_file+'.asrs.model.sav'
        filename = model_file+'.sav'
        model = pickle.load(open(filename, 'rb'))
        #print(model)
        
        X_test = test_mart.iloc[:,1:feat_dim+1]
        scores = model.predict_proba(X_test)
        scores_df = pd.DataFrame(scores)
        scores_df.columns = ['N', 'Y']
        scores_total = pd.concat([test_mart, scores_df], axis = 1)
        #scores_total.head(3)        
        
        test_list = pd.read_csv('./list/test.video', header = None)
        test_list.columns = ['video']
        #test_list.head(3)        

        final_list = test_list.merge(scores_total, how = 'left', on = 'video')[['video', 'Y']]
        final_list.to_csv(output_file+'.video'+'.csv', index = False)
        final_list['Y'].to_csv(output_file+'.csv', index = False, header = True)
        final_list.to_csv(output_file+'.video'+'.txt', index = False, header = False)
        final_list['Y'].to_csv(output_file+'.txt', index = False, header = False)
        
    print 'Model tested successfully for event %s!' % (model_file)