# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:52:10 2018

@author: Falcon4
"""
import math
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from neupy.algorithms import PNN

from scipy import io
import os

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from load_benchmark import load_datasetSL
from load_benchmark import load_datasetML
from RBF_ISCC_complete import RBF_ISCC
from RBF_ISCC_complete import RBF_ISCC_update_once
from RBF_ISCC_complete import RBF_ISCC_ML
from RBF_ISCC_complete import Normalize_col
from measure_score import performance_measure
from measure_score import performance_measureML
from measure_score import performance_Fmeasure
from measure_score import SClass_measure
        
if __name__ == "__main__":
#%%
    
    # Data set
    dataname = "yeast"
    dataset = load_datasetSL(dataname)
    data = dataset.data
    data = scale(data, axis=0)
    target = dataset.target
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0, stratify=target)
    train_labels = RBF_ISCC.target2matrix(Y_train, len(dataset.label_dict), pos_note=1, neg_note=0)
    true_label = RBF_ISCC.target2matrix(Y_test, len(dataset.label_dict), neg_note=0)
    for t in range(len(dataset.label_dict)):
        if t not in Y_train:
            print("Class {t} is not in train data".format(t=t))
#%%
#    # Parameter
#    threshold = 0.4
#    sigma_init = 0.8
#    alpha = math.sqrt(16)
#    beta = 4
#    learning_rate = 0.01
#    lambda_i = 0
#    normal = False
#    
#    # Algorithm method
#    starttime = time.time()
#    rbf_iscc = RBF_ISCC_update_once(threshold=threshold, sigma_init=sigma_init, 
#                                  alpha=alpha, beta=beta, lambda_i=lambda_i, 
#                                  learning_rate=learning_rate, mode='v4', norm=normal)
#    rbf_iscc.train(X_train, Y_train)
#    mean = rbf_iscc.node_mean
#    sigma = rbf_iscc.node_sigma
#    groups = rbf_iscc.node_num
#    weights = rbf_iscc.weights
#    
#    pred_label_tr = rbf_iscc.transform(X_train)
#    tra_label = RBF_SCC.target2matrix(Y_train, len(dataset.label_dict), neg_note=0)
#    score_tr = performance_measure(pred_label_tr, tra_label)
#    mark_tr = SClass_measure(pred_label_tr, tra_label)
#    error_train = [1-score_tr["AACC"]]
#    error_train_all = [score_tr]
#    
#    pred_label = rbf_iscc.transform(X_test)
#    score = performance_measure(pred_label, true_label)
#    mark = SClass_measure(pred_label, true_label)
#    print(score)
#    error_test = [1 - score["AACC"]]
#    error_test_all = [score]
#
#    flag = 0
#    end_flag = 1
#    while flag < end_flag:
#        rbf_iscc.update_once()
#        weights = rbf_iscc.weights
#        pred_label = rbf_iscc.transform(X_test, mode='v4')
#        score = performance_measure(pred_label, true_label)
#        mark = SClass_measure(pred_label, true_label)
#        error_test.append(1 - score["AACC"])
#        error_test_all.append(score)
#        
#        pred_label_tr = rbf_iscc.transform(X_train)
#        score_tr = performance_measure(pred_label_tr, tra_label)
#        mark_tr = SClass_measure(pred_label_tr, tra_label)
#        error_train.append(1 - score_tr["AACC"])
#        error_train_all.append(score_tr)
##        score = performance_measure(pred_label, true_label)
##        print(score)
#        
##        if abs(rbf_iscc.error[-2] - rbf_iscc.error[-1])/rbf_iscc.error[-1] < 0.02:
##            flag = end_flag
#        if abs(error_train[-2]-error_train[-1])/error_train[-1] < 0.00001:
#            flag = end_flag
#        elif rbf_iscc.epoch == rbf_iscc.epoch_max:
#            flag = end_flag
#            print("Epochs reach Maximun times.")
#        rbf_iscc.epoch += 1
#
#    plt.plot(error_train, marker='o')
#    plt.plot(error_test, marker='^')
#    plt.legend(['Training', 'Testing'], loc='upper right')
#    plt.show()
#    
#    plt.plot(rbf_iscc.error, marker='o')
#    plt.legend(['Class output error'], loc='upper right')
#    plt.show()
#    
#    rbfiscc_score = performance_measure(pred_label, true_label)
#    print("RBF-ISCC:{}".format(rbfiscc_score))
#    endtime = time.time()
#    print("Time:{} s".format(endtime-starttime))
    
#%%  
    # Parameter
    threshold = 0.1
    sigma_init = 0.8
    alpha = math.sqrt(16)
    beta = 2
    learning_rate = 0.1
    lambda_i = 0
    normal = False
    
    starttime=time.time()
    rbf_iscc = RBF_ISCC(threshold=threshold, sigma_init=sigma_init, alpha=alpha, 
                      beta=beta, lambda_i=lambda_i, learning_rate=learning_rate, 
                      mode='v4', norm=normal)
    rbf_iscc.train(X_train, Y_train)
    endtime=time.time()
    pred_label = rbf_iscc.transform(X_test, mode='v4')
    rbfiscc_score = performance_measure(pred_label, true_label)
    print("RBF-ISCC:{}".format(rbfiscc_score))
    print("Time:{} s".format(endtime-starttime))
    mean = rbf_iscc.node_mean
    sigma = rbf_iscc.node_sigma
    groups = rbf_iscc.node_num
    epoch = rbf_iscc.epoch
#%%    
    starttime=time.time()
    classifier = OneVsRestClassifier(LinearSVC(random_state=42))
    classifier.fit(X_train, train_labels)
    svc_predictions = classifier.predict(X_test)
    endtime=time.time()
    svc_score = performance_measure(svc_predictions, true_label)
    print("SVC:{score}".format(score=svc_score))
    print("Time:{} s".format(endtime-starttime))    
#%%    
    starttime=time.time()
    for i in range(100):
        svm_clf = SVC(kernel='rbf').fit(X_train, Y_train)
    endtime=time.time()
    svm_clf_pred = svm_clf.predict(X_test)
    svm_predictions = RBF_ISCC.target2matrix(svm_clf_pred, len(dataset.label_dict), pos_note=1, neg_note=0)
    svm_score = performance_measure(svm_predictions, true_label)
    print("SVM:{score}".format(score=svm_score))
    print("Time:{}s".format(endtime-starttime))
#%%
    starttime=time.time()
    for i in range(100):
        svr_clf = SVR().fit(X_train, Y_train)
        svr_clf_pred = np.round(svr_clf.predict(X_test)).astype(int)
    endtime=time.time()
    svr_predictions = RBF_ISCC.target2matrix(svr_clf_pred, len(dataset.label_dict), pos_note=1, neg_note=0)
    svr_score = performance_measure(svr_predictions, true_label)
    print("SVR:{score}".format(score=svr_score))
    print("Time:{}s".format(endtime-starttime))
    
#%%
    starttime=time.time()
    nnclf = MLPClassifier(solver="lbfgs", activation='tanh' , hidden_layer_sizes=(groups.sum(), ), random_state=1)
    nnclf.fit(X_train, Y_train)
    endtime=time.time()
    nn_pred = nnclf.predict(X_test)
    nn_predictions = RBF_ISCC.target2matrix(nn_pred, len(dataset.label_dict), pos_note=1, neg_note=0)
    nn_score = performance_measure(nn_predictions, true_label)
    print("NN:{score}".format(score=nn_score))
    print("Time:{}s".format(endtime-starttime))
#%%
    starttime=time.time()
    rbfclf = PNN(std=0.5, batch_size="all")
    rbfclf.train(X_train, Y_train)
    endtime=time.time()
    rbf_pred = rbfclf.predict(X_test)
    rbf_predictions = RBF_ISCC.target2matrix(rbf_pred, len(dataset.label_dict), pos_note=1, neg_note=0)
    rbf_score = performance_measure(rbf_predictions, true_label)
    print("RBF:{score}".format(score=rbf_score))
    print("Time:{}s".format(endtime-starttime))
#%%
#    starttime=time.time()
#    krr_clf = KernelRidge().fit(X_train, Y_train)
#    endtime=time.time()
#    krr_clf_pred = krr_clf.predict(X_test)
##    krr_clf_pred = np.round(krr_clf.predict(X_test)).astype(int)
##    krr_predictions = RBF_ISCC.target2matrix(krr_clf_pred, len(dataset.label_dict), pos_note=1, neg_note=0)
#    krr_score = performance_measureML(krr_predictions, true_label)
#    print("KRR:{score}".format(score=krr_score))
#    print("Time:{}s".format(endtime-starttime))
    
#%%
    """
    # Data set
    dataname = "flags"
    dataset = load_datasetML(dataname)
    data = dataset.data
    data = scale(data, axis=0)
    target = dataset.target
    n_train = 129
    X_train = data[:n_train]
    Y_train = target[:n_train]
    X_test = data[n_train:]
    Y_test = target[n_train:]
    true_label = Y_test
    for t in range(len(dataset.label_dict)):
        if t not in Y_train:
            print("Class {t} is not in train data".format(t=t))
    
#%%  
    # Parameter
    threshold = 0.4
    sigma_init = 0.8
    alpha = math.sqrt(8)
    beta = 2
    learning_rate = 0.1
    lambda_i = 10
    normal = False
    
    starttime=time.time()
    rbf_iscc = RBF_ISCC_ML(threshold=threshold, sigma_init=sigma_init, alpha=alpha, 
                      beta=beta, lambda_i=lambda_i, learning_rate=learning_rate, 
                      mode='v4', norm=normal)
    rbf_iscc.train(X_train, Y_train)
    endtime=time.time()
    pred_label = rbf_iscc.transform(X_test, mode='v4')
#    rbfiscc_score = performance_measureML(pred_label, true_label)
    rbfiscc_score = performance_Fmeasure(pred_label, true_label)
    print("RBF-ISCC:{}".format(rbfiscc_score))
    print("Time:{} s".format(endtime-starttime))
    mean = rbf_iscc.node_mean
    sigma = rbf_iscc.node_sigma
    groups = rbf_iscc.node_num
    epoch = rbf_iscc.epoch
#%%    
    starttime=time.time()
    classifier = OneVsRestClassifier(LinearSVC(random_state=42))
    classifier.fit(X_train, Y_train)
    svc_predictions = classifier.predict(X_test)
    endtime=time.time()
#    svc_score = performance_measureML(svc_predictions, true_label)
    svc_score = performance_Fmeasure(svc_predictions, true_label)
    print("SVC:{score}".format(score=svc_score))
    print("Time:{} s".format(endtime-starttime))

#%%
    starttime=time.time()
    nnclf = MLPClassifier(solver="lbfgs", activation='tanh' , hidden_layer_sizes=(groups.sum(), ), random_state=1)
    nnclf.fit(X_train, Y_train)
    endtime=time.time()
    nn_pred = nnclf.predict(X_test)
#    nn_predictions = RBF_ISCC.target2matrix(nn_pred, len(dataset.label_dict), pos_note=1, neg_note=0)
#    nn_score = performance_measureML(nn_predictions, true_label)
#    nn_score = performance_measureML(nn_pred, true_label)
    nn_score = performance_Fmeasure(nn_pred, true_label)
    print("NN:{score}".format(score=nn_score))
    print("Time:{}s".format(endtime-starttime))
    
#%%
    starttime=time.time()
    knnclf = KNeighborsClassifier(n_neighbors=groups.sum())
    knnclf.fit(X_train, Y_train)
    endtime=time.time()
    knn_pred = knnclf.predict(X_test)
#    nn_predictions = RBF_ISCC.target2matrix(nn_pred, len(dataset.label_dict), pos_note=1, neg_note=0)
#    nn_score = performance_measureML(nn_predictions, true_label)
#    knn_score = performance_measureML(knn_pred, true_label)
    knn_score = performance_Fmeasure(knn_pred, true_label)
    print("kNN:{score}".format(score=knn_score))
    print("Time:{}s".format(endtime-starttime))
    """
#%%    
    """ 
    data_name = dataname
    if not os.path.exists(data_name):
        os.mkdir(data_name)
    filename_data = "{n}/Data.mat".format(n=data_name)
    filename_target = "{n}/Target.mat".format(n=data_name)
    io.savemat(filename_data, {'X_train': X_train, 'X_test': X_test})
    io.savemat(filename_target, {'Y_train': Y_train, 'Y_test': Y_test})
    filename_cluster = "{n}/cluster.txt".format(n=data_name)
    with open(filename_cluster, mode='a+') as f1:
        f1.write("{n_c}, ".format(n_c=num_groups.sum()))
    """