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

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        print "model_type -- mfcc or asrs"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    model_type = sys.argv[5]

    def modeling_ap_SVM(k, train_data, valid_data, target = 'target_p001_10'):
        start_time = time.time()
        k = k
        train_mart = train_data
        valid_mart = valid_data
        target = target

        X_train = train_mart.iloc[:,1:k+1]
        y_train = train_mart[target]
        X_valid = valid_mart.iloc[:,1:k+1]
        y_valid = valid_mart[target]

        model = SVC(kernel=chi2_kernel, probability=True)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_valid)
        y_probs = model.predict_proba(X_valid)
        results = average_precision_score(y_true=y_valid.values, y_score=y_probs[:,1])
        print("===== The time consuming of SVM Modeling : {} seconds =====".format((time.time() - start_time)))   
        print(results)
        return results, y_probs, model

    def modeling_ap_AdaB(k, train_data, valid_data, target = 'target_p001_10'):
        start_time = time.time()
        k = k
        train_mart = train_data
        valid_mart = valid_data
        target = target

        X_train = train_mart.iloc[:,1:k+1]
        y_train = train_mart[target]
        X_valid = valid_mart.iloc[:,1:k+1]
        y_valid = valid_mart[target]

        model = AdaBoostClassifier(n_estimators=200, random_state=0)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_valid)
        y_probs = model.predict_proba(X_valid)
        results = average_precision_score(y_true=y_valid.values, y_score=y_probs[:,1])
        print("===== The time consuming of AdaBoosting Modeling : {} seconds =====".format((time.time() - start_time)))   
        print(results)
        return results, y_probs, model

    def modeling_ap_Boost(k, train_data, valid_data, target = 'target_p001_10'):
        start_time = time.time()
        k = k
        train_mart = train_data
        valid_mart = valid_data
        target = target

        X_train = train_mart.iloc[:,1:k+1]
        y_train = train_mart[target]
        X_valid = valid_mart.iloc[:,1:k+1]
        y_valid = valid_mart[target]

        model = GradientBoostingClassifier(n_estimators=200, random_state=0)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_valid)
        y_probs = model.predict_proba(X_valid)
        results = average_precision_score(y_true=y_valid.values, y_score=y_probs[:,1])
        print("===== The time consuming of Boosting Modeling : {} seconds =====".format((time.time() - start_time)))   
        print(results)
        return results, y_probs, model

    def modeling_ap_xgb(k, train_data, valid_data, target = 'target_p001_10'):
        start_time = time.time()
        k = k
        train_mart = train_data
        valid_mart = valid_data
        target = target

        X_train = train_mart.iloc[:,1:k+1]
        y_train = train_mart[target]
        X_valid = valid_mart.iloc[:,1:k+1]
        y_valid = valid_mart[target]

        model = XGBClassifier()
        model.fit(X_train, y_train)
        y_preds = model.predict(X_valid)
        y_probs = model.predict_proba(X_valid)
        results = average_precision_score(y_true=y_valid.values, y_score=y_probs[:,1])
        print("===== The time consuming of XgBoosting Modeling : {} seconds =====".format((time.time() - start_time)))   
        print(results)
        return results, y_probs, model

    def modeling_ap_lgbm(k, train_data, valid_data, target = 'target_p001_10'):
        start_time = time.time()
        k = k
        train_mart = train_data
        valid_mart = valid_data
        target = target

        X_train = train_mart.iloc[:,1:k+1]
        y_train = train_mart[target]
        X_valid = valid_mart.iloc[:,1:k+1]
        y_valid = valid_mart[target]

        model = LGBMClassifier(random_state=0, n_jobs=-1)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_valid)
        y_probs = model.predict_proba(X_valid)
        results = average_precision_score(y_true=y_valid.values, y_score=y_probs[:,1])
        print("===== The time consuming of XgBoosting Modeling : {} seconds =====".format((time.time() - start_time)))   
        print(results)
        return results, y_probs, model    
    
    if model_type == 'mfcc':
        train_mart = pd.read_csv('./datamart_train_k{}.csv'.format(feat_dim))
        valid_mart = pd.read_csv('./datamart_valid_k{}.csv'.format(feat_dim))
        test_mart  = pd.read_csv('./datamart_test_k{}.csv'.format(feat_dim))
        
        print(train_mart.shape, valid_mart.shape, test_mart.shape)
        
        if event_name == 'P001':
            precision, y_probs, model = modeling_ap_lgbm(k=feat_dim, train_data = train_mart, valid_data = valid_mart, target = 'target_p001_10')
            foo = pd.Series({'Validation-AP':precision})
            foo_y = pd.DataFrame(y_probs)
            foo_y.columns = ['N', 'Y']
            foo_y2 = pd.concat([valid_mart, foo_y], axis = 1)[['video', 'N', 'Y']]
            foo.to_csv('./mfcc_pred/{}_k{}_model_Val_AP.csv'.format(event_name, feat_dim), index = False)
            foo_y2.to_csv('./mfcc_pred/{}_k{}_model_Val_Scores.csv'.format(event_name, feat_dim), index = False)
            # save the model to disk
            filename = output_file+'.sav'
            pickle.dump(model, open(filename, 'wb'))
        elif event_name == 'P002':
            precision, y_probs, model = modeling_ap_xgb(k=feat_dim, train_data = train_mart, valid_data = valid_mart, target = 'target_p002_10')
            foo = pd.Series({'Validation-AP':precision})
            foo_y = pd.DataFrame(y_probs)
            foo_y.columns = ['N', 'Y']
            foo_y2 = pd.concat([valid_mart, foo_y], axis = 1)[['video', 'N', 'Y']]
            foo.to_csv('./mfcc_pred/{}_k{}_model_Val_AP.csv'.format(event_name, feat_dim), index = False)
            foo_y2.to_csv('./mfcc_pred/{}_k{}_model_Val_Scores.csv'.format(event_name, feat_dim), index = False)
            # save the model to disk
            filename = output_file+'.sav'
            pickle.dump(model, open(filename, 'wb'))            
        elif event_name == 'P003':
            precision, y_probs, model = modeling_ap_xgb(k=feat_dim, train_data = train_mart, valid_data = valid_mart, target = 'target_p003_10')       
            foo = pd.Series({'Validation-AP':precision})
            foo_y = pd.DataFrame(y_probs)
            foo_y.columns = ['N', 'Y']
            foo_y2 = pd.concat([valid_mart, foo_y], axis = 1)[['video', 'N', 'Y']]
            foo.to_csv('./mfcc_pred/{}_k{}_model_Val_AP.csv'.format(event_name, feat_dim), index = False)
            foo_y2.to_csv('./mfcc_pred/{}_k{}_model_Val_Scores.csv'.format(event_name, feat_dim), index = False)
            # save the model to disk
            filename = output_file+'.sav'
            pickle.dump(model, open(filename, 'wb'))            


            
    elif model_type == 'asrs':
        train_mart = pd.read_csv('./datamart_train_asrfeat_{}.csv'.format(feat_dim))
        valid_mart = pd.read_csv('./datamart_valid_asrfeat_{}.csv'.format(feat_dim))
        test_mart  = pd.read_csv('./datamart_test_asrfeat_{}.csv'.format(feat_dim))
        
        print(train_mart.shape, valid_mart.shape, test_mart.shape)
        
        if event_name == 'P001':
            precision, y_probs, model = modeling_ap_Boost(k=feat_dim, train_data = train_mart, valid_data = valid_mart, target = 'target_p001_10')
            foo = pd.Series({'Validation-AP':precision})
            foo_y = pd.DataFrame(y_probs)
            foo_y.columns = ['N', 'Y']
            foo_y2 = pd.concat([valid_mart, foo_y], axis = 1)[['video', 'N', 'Y']]
            foo.to_csv('./asr_pred/{}_{}_model_Val_AP.csv'.format(event_name, feat_dim), index = False)
            foo_y2.to_csv('./asr_pred/{}_{}_model_Val_Scores.csv'.format(event_name, feat_dim), index = False)
            # save the model to disk
            filename = output_file+'.sav'
            pickle.dump(model, open(filename, 'wb'))                
        elif event_name == 'P002':
            precision, y_probs, model = modeling_ap_AdaB(k=feat_dim, train_data = train_mart, valid_data = valid_mart, target = 'target_p002_10')
            foo = pd.Series({'Validation-AP':precision})
            foo_y = pd.DataFrame(y_probs)
            foo_y.columns = ['N', 'Y']
            foo_y2 = pd.concat([valid_mart, foo_y], axis = 1)[['video', 'N', 'Y']]
            foo.to_csv('./asr_pred/{}_{}_model_Val_AP.csv'.format(event_name, feat_dim), index = False)
            foo_y2.to_csv('./asr_pred/{}_{}_model_Val_Scores.csv'.format(event_name, feat_dim), index = False)
            # save the model to disk
            filename = output_file+'.sav'
            pickle.dump(model, open(filename, 'wb'))                            
        elif event_name == 'P003':
            precision, y_probs, model = modeling_ap_SVM(k=feat_dim, train_data = train_mart, valid_data = valid_mart, target = 'target_p003_10') 
            foo = pd.Series({'Validation-AP':precision})
            foo_y = pd.DataFrame(y_probs)
            foo_y.columns = ['N', 'Y']
            foo_y2 = pd.concat([valid_mart, foo_y], axis = 1)[['video', 'N', 'Y']]
            foo.to_csv('./asr_pred/{}_{}_model_Val_AP.csv'.format(event_name, feat_dim), index = False)
            foo_y2.to_csv('./asr_pred/{}_{}_model_Val_Scores.csv'.format(event_name, feat_dim), index = False)
            # save the model to disk
            filename = output_file+'.sav'
            pickle.dump(model, open(filename, 'wb'))                

    print 'Model trained successfully for event %s!' % (event_name)
