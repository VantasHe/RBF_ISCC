#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:28:52 2017

@author: Vick, He
@data: 2018/07/24
@version: 1.1.0
@description: Classifier implementing RBF-ISCC.

"""

import numpy as np
import math

from numpy.linalg import lstsq
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from SCCI import SCCI
from SCCI import SCCI_enhance
from measure_score import performance_measure


class Normalize_col:
    def __init__(self, train_data):
        self.col = np.size(train_data, 1)
        self.col_min = np.min(train_data, axis=0)
        col_max = np.max(train_data, axis=0)

        self.delta_array = col_max - self.col_min

        for idx, delta in enumerate(self.delta_array):
            if delta == 0:
                self.delta_array[idx] = 1

    def norm(self, data):
        nor_data = (data - self.col_min) / self.delta_array
        return nor_data


class RBF_ISCC:
    """
    Radius basis function network with Iterative self-construct cluster algorithm.

    Parameter:
    ----------
        node_mean : array_like, shape(num_hidden_nodes, dim)
            Centroids of groups after clustering.
        node_sigam :    array_like, shape(num_hidden_nodes, dim)
            Deviations of groups after clustering.
        node_num :  list
        weights :


    """

    def __init__(self, threshold=0.55, sigma_init=0.2, alpha=1, beta=1, lambda_i=0,
                 learning_rate=0.1, mode='v4', norm=False):
    """
    Initialize the parameters of RBF-ISCC.

    Parameter:
    ----------
        threshold :
        sigma_init :
        alpha :
        beta :
        lambda_i :
        learning_rate :
        mode :
        norm :
    """

        self.threshold = threshold
        self.sigma_init = sigma_init
        self.alpha = alpha
        self.beta = beta
        self.lambda_i = lambda_i
        self.learning_rate = learning_rate
        self.mode = mode
        self.norm = norm
        self.normalizer = None
        self.error = []
        self.error_train = []
        self.epoch_max = 100
        self.epoch = 1

    def train(self, train_data, train_label):
        data = self.preprocess_data(train_data, train_label)
        tar_matrix = self.target2matrix(train_label, self.num_classes)

        self.get_hidden_nodes(data, train_label)
        SCCI_enhance.set_alpha(self.alpha)
        node_output = SCCI_enhance.get_MD(
            data, self.node_mean, self.node_sigma)
        node_output = np.append(node_output, np.ones(
            (np.size(node_output, 0), 1), dtype=float), axis=1)

        if self.mode == 'v1':
            pass
        elif self.mode == 'v2':
            self.get_network_weights(node_output, tar_matrix)
            class_output = np.matmul(node_output, self.weights)
            self.error.append(self.error_estimate(class_output, tar_matrix))
        elif self.mode == 'v4':
            self.get_network_weights(node_output, tar_matrix)
            class_output = np.matmul(node_output, self.weights)
            self.error.append(self.error_estimate(class_output, tar_matrix))

            # Check train result
            boundary = 0.0
            pred_table = self.matrix2Binary(class_output, boundary)
            train_target = self.target2matrix(
                train_label, self.num_classes, neg_note=0)
            self.error_train.append(1-accuracy_score(pred_table, train_target))

            flag = 0
            end_flag = 1
            count = 0
            while flag < end_flag:
                self.update(data, tar_matrix, node_output, class_output)
                node_output = SCCI_enhance.get_MD(
                    data, self.node_mean, self.node_sigma)
                node_output = np.append(node_output, np.ones(
                    (np.size(node_output, 0), 1), dtype=float), axis=1)
                self.get_network_weights(node_output, tar_matrix)
                class_output = np.matmul(node_output, self.weights)
                pred_table = self.matrix2Binary(class_output, boundary)

                self.error_train.append(
                    1-accuracy_score(pred_table, train_target))
                self.error.append(self.error_estimate(
                    class_output, tar_matrix))

#                if abs(self.error[-2] - self.error[-1])/self.error[-1] < 0.01:
#                    flag = end_flag
                if self.error_train[-1] == 0.0:
                    flag = end_flag
                    print("Error is equal to ZERO!")
                elif ((self.error_train[-2]-self.error_train[-1])/self.error_train[-1]) < 0.01:
                    if count > 3:
                        flag = end_flag
                    else:
                        count += 1
                elif self.epoch == self.epoch_max:
                    flag = end_flag
                    print("Epochs reach Maximun times.")
                else:
                    count = 0
                self.epoch += 1

    def transform(self, test_data, mode=None):
        if self.norm == True:
            t_data = self.normalizer.norm(test_data)
        else:
            t_data = test_data

        if mode == None:
            mode = self.mode

        if mode == 'v1':
            SCCI_enhance.set_alpha(self.alpha)
            similarity_result = SCCI_enhance.get_MD(
                t_data, self.node_mean, self.node_sigma)
            pred_table = np.zeros([len(test_data), self.num_classes])
            for i, sim_result in enumerate(similarity_result):
                pred_cls = sim_result.argmax()
                pred = self.node_to_class_dict[pred_cls]
                pred_table[i][pred] = 1
        else:
            boundary = 0.0
            SCCI_enhance.set_alpha(self.alpha)
            node_output = SCCI_enhance.get_MD(
                t_data, self.node_mean, self.node_sigma)
            node_output = np.append(node_output, np.ones(
                (np.size(node_output, 0), 1)), axis=1)
            target_output = np.matmul(node_output, self.weights)
            pred_table = self.matrix2Binary(target_output, boundary)

        return pred_table

    def transform2(self, test_data, mode=None):
        if self.norm == True:
            t_data = self.normalizer.norm(test_data)
        else:
            t_data = test_data

        if mode == None:
            mode = self.mode

        if mode == 'v1':
            similarity_result = SCCI.get_MD(
                t_data, self.node_mean, self.node_sigma)
            pred_table = np.zeros([len(test_data), self.num_classes])
            for i, sim_result in enumerate(similarity_result):
                pred_cls = sim_result.argmax()
                pred = self.node_to_class_dict[pred_cls]
                pred_table[i][pred] = 1
        else:
            boundary = 0.0
            SCCI_enhance.set_alpha(self.alpha)
            node_output = SCCI_enhance.get_MD(
                t_data, self.node_mean, self.node_sigma)
            node_output = np.append(node_output, np.ones(
                (np.size(node_output, 0), 1)), axis=1)
            target_output = np.matmul(node_output, self.weights)
            pred_table = self.matrix2Binary(target_output, boundary)
            ylabel = []
            for i in pred_table:
                ylabel.append(i.argmax())
            ylabel = np.array(ylabel)

        return ylabel

    def preprocess_data(self, train_data, train_labels):
        if self.norm:
            self.normalizer = Normalize_col(train_data)
            data = self.normalizer.norm(train_data)
        else:
            data = train_data
        self.num_classes = len(set(list(train_labels)))
        return data

    def get_hidden_nodes(self, train_data, train_labels):
        dataID_splited_by_class = [[] for i in range(self.num_classes)]
        for label_id, label in enumerate(train_labels):
            dataID_splited_by_class[label].append(label_id)

        trained_model = []
        for dataID_by_class in dataID_splited_by_class:
            scci = SCCI(threshold=self.threshold, sigma_init=self.sigma_init)
#            scci = SCCI_enhance(threshold=self.threshold,
#                    sigma_init=self.sigma_init, deviation_alpha=self.alpha)
            if len(dataID_by_class) >= 1:
                if len(dataID_by_class) == 1:
                    training_set = train_data[dataID_by_class].reshape(
                        1, len(train_data[0]))
                else:
                    training_set = train_data[dataID_by_class]
                scci.scci(training_set)
                trained_model.append(scci)

        # Concatenate all centroid of SCC to a total Mean and Sigma.
        temp_node_mean = None
        for model in trained_model:
            if temp_node_mean is None:
                temp_node_mean = model.mean
#                temp_node_sigma = model.get_no_init_sigma()
                temp_node_sigma = model.sigma
            else:
                temp_node_mean = np.concatenate(
                    (temp_node_mean, model.mean), axis=0)
                temp_node_sigma = np.concatenate(
                    #                    (temp_node_sigma, model.get_no_init_sigma()), axis=0)
                    (temp_node_sigma, model.sigma), axis=0)
            print("Iterative times:{}".format(model.iter_times))

        self.node_mean = temp_node_mean
        self.node_sigma = temp_node_sigma
        self.node_num = np.array(
            [model.num_clusters for model in trained_model])

        self.node_to_class_dict = dict.fromkeys(range(self.node_num.sum()))
        temp_count = 0
        for count, num in enumerate(self.node_num):
            for n in range(num):
                self.node_to_class_dict[temp_count+n] = count
            temp_count += num

    def get_network_weights(self, node_output, target_matrix):
        # Least-square to find the weights
        out_identity = np.append(node_output, math.sqrt(
            self.lambda_i)*np.identity(np.size(node_output, axis=1)), axis=0)
        tar_identity = np.append(target_matrix, np.zeros(
            (np.size(node_output, axis=1), np.size(target_matrix, axis=1))), axis=0)
        self.weights = lstsq(out_identity, tar_identity, rcond=None)[0]
#        self.weights = lstsq(node_output, target_matrix)[0]

    def update(self, train_data, tar_mat, node_out, cls_out):
        trans_cls_out = self.basis_func(cls_out)

        update_mean = np.zeros(self.node_mean.shape, dtype=float)
        update_sigma = np.zeros(self.node_sigma.shape, dtype=float)
        for mean_id, mean in enumerate(self.node_mean):
            delta_mean = []
            delta_sigma = []
            for data_id, data in enumerate(train_data):
                # x-c_j
                temp_A_1 = data - mean
                # (alpha*v(j))^2
                temp_A_2 = pow(self.alpha*self.node_sigma[mean_id], 2)
                # (A_1/A_2)*hidden_out(j)
                temp_A = (temp_A_1/temp_A_2)*node_out[data_id][mean_id]

                # target-clsOut
                temp_B_1 = tar_mat[data_id] - trans_cls_out[data_id]
                # beta*(1-clsOut^2)
                temp_B_2 = self.beta*(1-pow(trans_cls_out[data_id], 2))
                # weight
                temp_B_3 = self.weights[mean_id]
                temp_B = (temp_B_1*temp_B_2*temp_B_3).sum()

                # (x-c(j))^2
                temp_C_1 = pow((data - mean), 2)
                # alpha^2*v(j)^3
                temp_C_2 = pow(self.alpha, 2)*pow(self.node_sigma[mean_id], 3)
                temp_C = (temp_C_1/temp_C_2)*node_out[data_id][mean_id]

                delta_mean.append(temp_A*temp_B)
                delta_sigma.append(temp_C*temp_B)

            delta_mean = (np.array(delta_mean).sum(axis=0))/len(train_data)
            delta_sigma = (np.array(delta_sigma).sum(axis=0))/len(train_data)
            update_mean[mean_id] = 4*self.learning_rate*delta_mean
            update_sigma[mean_id] = 4*self.learning_rate*delta_sigma

        self.node_mean += update_mean
        self.node_sigma += update_sigma

    def basis_func(self, out_ar):
        # (exp(beta*sf) - exp(-beta*sf))/(exp(beta*sf)+exp(-beta*sf))
        posi_exp = np.exp(self.beta*out_ar)
        nega_exp = np.exp((-1.0)*self.beta*out_ar)
        activation = (posi_exp-nega_exp)/(posi_exp+nega_exp)
        return activation

    def error_estimate(self, out_ar, tar_matrix):
        output = self.basis_func(out_ar)
        error_temp = tar_matrix - output
        sqr_error = (error_temp*error_temp).sum(axis=0)/len(out_ar)
        return sqr_error.sum()

    @classmethod
    def matrix2Binary(cls, source_matrix, boundary=0):
        output_table = source_matrix > boundary
        pred_table = np.zeros(
            (np.size(source_matrix, 0), np.size(source_matrix, 1)))
        for i, sim_result in enumerate(output_table):
            for j, result in enumerate(sim_result):
                if result:
                    pred_table[i][j] = 1
                else:
                    pred_table[i][j] = 0
        return pred_table

    @classmethod
    def target2matrix(cls, target_label, num_classes, pos_note=1.0, neg_note=-1.0):
        target_matrix = neg_note * \
            np.ones([len(target_label), num_classes], dtype=float)
        for serial, target in enumerate(target_label):
            target_matrix[serial][target] = pos_note
        return target_matrix


class RBF_ISCC_update_once(RBF_ISCC):
    def __init__(self, threshold=0.55, sigma_init=0.2, alpha=1, beta=1, learning_rate=0.1, lambda_i=1, mode='v4', norm=True):
        super().__init__(threshold, sigma_init, alpha,
                         beta, lambda_i, learning_rate, mode, norm)

    def train(self, train_data, train_label):
        self.data = self.preprocess_data(train_data, train_label)
        self.tar_matrix = self.target2matrix(train_label, self.num_classes)

        self.get_hidden_nodes(self.data, train_label)
        SCCI_enhance.set_alpha(self.alpha)
        node_output = SCCI_enhance.get_MD(
            self.data, self.node_mean, self.node_sigma)
        self.node_output = np.append(node_output, np.ones(
            (np.size(node_output, 0), 1), dtype=float), axis=1)

        if self.mode == 'v1':
            pass
        elif self.mode == 'v2':
            self.get_network_weights(self.node_output, self.tar_matrix)
            class_output = np.matmul(self.node_output, self.weights)
            self.error.append(self.error_estimate(
                class_output, self.tar_matrix))
        elif self.mode == 'v4':
            self.get_network_weights(self.node_output, self.tar_matrix)
            class_output = np.matmul(self.node_output, self.weights)
            self.error.append(self.error_estimate(
                class_output, self.tar_matrix))

            # Check train result
            pred_table = self.matrix2Binary(class_output)
            self.train_target = self.target2matrix(
                train_label, self.num_classes, neg_note=0)
            self.error_train.append(
                1 - accuracy_score(pred_table, self.train_target))

    def update_once(self):
        class_output = np.matmul(self.node_output, self.weights)
        self.update(self.data, self.tar_matrix, self.node_output, class_output)
        node_output = SCCI_enhance.get_MD(
            self.data, self.node_mean, self.node_sigma)
        self.node_output = np.append(node_output, np.ones(
            (np.size(node_output, 0), 1), dtype=float), axis=1)
        self.get_network_weights(self.node_output, self.tar_matrix)
        class_output = np.matmul(self.node_output, self.weights)
        pred_table = self.matrix2Binary(class_output)

        self.error_train.append(
            1 - accuracy_score(pred_table, self.train_target))
        self.error.append(self.error_estimate(class_output, self.tar_matrix))


class RBF_ISCC_ML(RBF_ISCC):
    def __init__(self, threshold=0.55, sigma_init=0.2, alpha=1, beta=1, learning_rate=0.1, lambda_i=1, mode='v4', norm=False):
        super().__init__(threshold, sigma_init, alpha,
                         beta, lambda_i, learning_rate, mode, norm)

    def train(self, train_data, train_label):
        self.num_classes = len(train_label[0])
        data = train_data
        tar_matrix = np.zeros(train_label.shape)
        for i, row in enumerate(train_label):
            for j, col in enumerate(row):
                if col == 1:
                    tar_matrix[i][j] = 1.0
                else:
                    tar_matrix[i][j] = -1.0

        self.get_hidden_nodes(data, train_label)
        SCCI_enhance.set_alpha(self.alpha)
        node_output = SCCI_enhance.get_MD(
            data, self.node_mean, self.node_sigma)
        node_output = np.append(node_output, np.ones(
            (np.size(node_output, 0), 1), dtype=float), axis=1)

        if self.mode == 'v1':
            pass
        elif self.mode == 'v2':
            self.get_network_weights(node_output, tar_matrix)
            class_output = np.matmul(node_output, self.weights)
            self.error.append(self.error_estimate(class_output, tar_matrix))
        elif self.mode == 'v4':
            self.get_network_weights(node_output, tar_matrix)
            class_output = np.matmul(node_output, self.weights)
            self.error.append(self.error_estimate(class_output, tar_matrix))

            # Check train result
            boundary = 0.0
            pred_table = self.matrix2Binary(class_output, boundary)
            train_target = train_label
            self.error_train.append(1-accuracy_score(pred_table, train_target))

            flag = 0
            end_flag = 1
            count = 0
            while flag < end_flag:
                self.update(data, tar_matrix, node_output, class_output)
                node_output = SCCI_enhance.get_MD(
                    data, self.node_mean, self.node_sigma)
                node_output = np.append(node_output, np.ones(
                    (np.size(node_output, 0), 1), dtype=float), axis=1)
                self.get_network_weights(node_output, tar_matrix)
                class_output = np.matmul(node_output, self.weights)
                pred_table = self.matrix2Binary(class_output, boundary)

                self.error_train.append(
                    1-accuracy_score(pred_table, train_target))
                self.error.append(self.error_estimate(
                    class_output, tar_matrix))

                # Early stopping
                if self.error_train[-1] == 0.0:
                    flag = end_flag
                    print("Error is equal to ZERO!")
                elif ((self.error_train[-2]-self.error_train[-1])/self.error_train[-1]) < 0.005:
                    if count > 2:
                        flag = end_flag
                    else:
                        count += 1
                elif self.epoch == self.epoch_max:
                    flag = end_flag
                    print("Epochs reach Maximun times.")
                self.epoch += 1

    def get_hidden_nodes(self, train_data, train_labels):
        dataID_splited_by_class = [[] for i in range(self.num_classes)]
        for data_id, data_label in enumerate(train_labels):
            for tar_id, target in enumerate(data_label):
                if target == 1:
                    dataID_splited_by_class[tar_id].append(data_id)

        trained_model = []
        for dataID_by_class in dataID_splited_by_class:
            scci = SCCI(threshold=self.threshold, sigma_init=self.sigma_init)
#            scci = SCCI_enhance(threshold=self.threshold,
#                    sigma_init=self.sigma_init, deviation_alpha=self.alpha)
            if len(dataID_by_class) >= 1:
                if len(dataID_by_class) == 1:
                    training_set = train_data[dataID_by_class].reshape(
                        1, len(train_data[0]))
                else:
                    training_set = train_data[dataID_by_class]
                scci.scci(training_set)
                trained_model.append(scci)

        # Concatenate all centroid of SCC to a total Mean and Sigma.
        temp_node_mean = None
        for model in trained_model:
            if temp_node_mean is None:
                temp_node_mean = model.mean
#                temp_node_sigma = model.get_no_init_sigma()
                temp_node_sigma = model.sigma
            else:
                temp_node_mean = np.concatenate(
                    (temp_node_mean, model.mean), axis=0)
                temp_node_sigma = np.concatenate(
                    #                    (temp_node_sigma, model.get_no_init_sigma()), axis=0)
                    (temp_node_sigma, model.sigma), axis=0)
#            print("Iterative times:{}".format(model.iter_times))

        self.node_mean = temp_node_mean
        self.node_sigma = temp_node_sigma
        self.node_num = np.array(
            [model.num_clusters for model in trained_model])

        self.node_to_class_dict = dict.fromkeys(range(self.node_num.sum()))
        temp_count = 0
        for count, num in enumerate(self.node_num):
            for n in range(num):
                self.node_to_class_dict[temp_count+n] = count
            temp_count += num


if __name__ == "__main__":
    X_train = np.array([
        [0.30, 0.60],
        [0.70, 0.35],
        [0.50, 0.52],
        [0.35, 0.38],
        [0.19, 0.89],
        [0.78, 0.20],
        [0.62, 0.25],
        [0.24, 0.81],
        [0.29, 0.89],
        [0.40, 0.65],
        [0.28, 0.48],
        [0.24, 0.89]
    ])
    Y_train = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0])

    threshold = 0.55
    sigma_init = 0.2
    alpha = math.sqrt(2)
    beta = 2
    learning_rate = 1
    lambda_i = 0

    tra_label = RBF_ISCC.target2matrix(Y_train, 2, neg_note=0)

    rbf_iscc = RBF_ISCC_update_once(threshold=threshold, sigma_init=sigma_init,
                                    alpha=alpha, beta=beta, lambda_i=lambda_i,
                                    learning_rate=learning_rate, mode='v4', norm=False)
    rbf_iscc.train(X_train, Y_train)

    w = rbf_iscc.weights
    SCCI_enhance.set_alpha(alpha)
    node_output = SCCI_enhance.get_MD(
        X_train, rbf_iscc.node_mean, rbf_iscc.node_sigma)
    node_output = np.append(node_output, np.ones(
        (np.size(node_output, 0), 1), dtype=float), axis=1)
    cls_output = np.matmul(node_output, rbf_iscc.weights)

    pred_label_tr = rbf_iscc.transform(X_train)
    score_tr = performance_measure(pred_label_tr, tra_label)
    error_tr = [1 - score_tr["AACC"]]

    flag = 0
    end_flag = 1
    if error_tr[-1] == 0:
        flag = end_flag
    while flag < end_flag:
        rbf_iscc.update_once()
        pred_label_tr = rbf_iscc.transform(X_train)
        score_tr = performance_measure(pred_label_tr, tra_label)
        error_tr.append([1 - score_tr["AACC"]])

        if error_tr[-1] == 0:
            flag = end_flag
        elif abs(error_tr[-2] - error_tr[-1])/error_tr[-1] < 0.01:
            flag = end_flag
        elif rbf_iscc.epoch == rbf_iscc.epoch_max:
            flag = end_flag
            print("Epochs reach Maximun times.")
        rbf_iscc.epoch += 1

    test_data = np.array([[0.8, 0.35]])
    result = rbf_iscc.transform(test_data)
    node_output = SCCI_enhance.get_MD(
        test_data, rbf_iscc.node_mean, rbf_iscc.node_sigma)
    node_output = np.append(node_output, np.ones(
        (np.size(node_output, 0), 1), dtype=float), axis=1)
    cls_output = np.matmul(node_output, rbf_iscc.weights)
    pred_1 = rbf_iscc.transform(test_data)
    score_1 = performance_measure(pred_1, np.array([[1, 0]]))

    test_data2 = np.array([[0.5, 0.5]])
    result2 = rbf_iscc.transform(test_data2)
    node_output = SCCI_enhance.get_MD(
        test_data2, rbf_iscc.node_mean, rbf_iscc.node_sigma)
    node_output2 = np.append(node_output, np.ones(
        (np.size(node_output, 0), 1), dtype=float), axis=1)
    cls_output2 = np.matmul(node_output2, rbf_iscc.weights)
    pred_2 = rbf_iscc.transform(test_data2)
    score_2 = performance_measure(pred_2, np.array([[0, 1]]))

    test_data3 = np.array([[0.3, 0.95]])
    result3 = rbf_iscc.transform(test_data3)
    node_output = SCCI_enhance.get_MD(
        test_data3, rbf_iscc.node_mean, rbf_iscc.node_sigma)
    node_output3 = np.append(node_output, np.ones(
        (np.size(node_output, 0), 1), dtype=float), axis=1)
    cls_output3 = np.matmul(node_output3, rbf_iscc.weights)
    pred_3 = rbf_iscc.transform(test_data3)
    score_3 = performance_measure(pred_3, np.array([[1, 0]]))
#    #if len(sys.argv) > 1:
#    #    dataname = sys.argv[1]
#    dataname = "iris"
#    #accuarcy = []
#    n_fold = 5
#    iteration = 0
#    score = []
#    for serial in range(5):
#        score.append(run_benchmark(dataname, serial+1, epoch=iteration))
#    SCC_score = np.array([scc[0] for scc in score]).sum(axis=0)
#    SVM_score = np.array([svm[1] for svm in score]).sum(axis=0)
#    #scores = np.array(accuarcy).sum(axis=0) / 5
#    print("Dataset: {name}".format(name=dataname))
#    print("LVQ-SCC accuracy: {scc_acc:.4f}, precision: {scc_pre:.4f}, recall: {scc_rec:.4f}, f1_score: {scc_f1:.4f}".format(
#        scc_acc=SCC_score[0]/n_fold, scc_pre=SCC_score[1]/n_fold, scc_rec=SCC_score[2]/n_fold, scc_f1=SCC_score[3]/n_fold))
#    #print("KMeans accuracy = {kmean}".format(kmean=k_Mean_score/n_fold))
#    print("SVM \taccuracy: {svm_acc:.4f}, precision: {svm_pre:.4f}, recall: {svm_rec:.4f}, f1_score: {svm_f1:.4f}".format(
#        svm_acc=SVM_score[0]/n_fold, svm_pre=SVM_score[1]/n_fold, svm_rec=SVM_score[2]/n_fold, svm_f1=SVM_score[3]/n_fold))
