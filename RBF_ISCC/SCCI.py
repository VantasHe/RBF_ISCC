#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:28:52 2017

@author: Vick, He
@date: 2018/06/21
@version: 1.0.3
@update_date: 2018/07/05
@description: SCC_I implement

"""
import numpy as np
import math


class SCC:
    """
    Original Self-Constructing Clustering algorithm.

    parameter:
    ----------
        mean :  array_like, shape(num_cluster, dim)
            Centroids of groups after clustering.
        sigma : array_like, shape(num_cluster, dim)
            Deviations of groups after clustering.
        sigma_init : array, shape(dim)
            Inital deviation for SCC first time.
        mu_threshold :  float
            The threshold is calulated by natural log.
        group_of_data : array, shape(num_data)
            Store the group for each data.
        num_in_group :  list
            Store the numbers of data in each group.
        num_clusters :  int
            Store number of total clusters
        dataID_in_group :   list
            Store the serial number of data in each group.

    """

    def __init__(self, threshold, sigma_init=0.1):
        """
        Initialize the parameters of SCC.

        Parameter:
        ----------
            threshold : float
                The threshold to decide whether adding point to group or not.
            sigma_init : float
                The initial value of deviation.

        """

        # Parameter
        self.threshold = threshold
        self.sigma_initial = sigma_init

    def scc(self, traindata):
        """
        Original Self-Constructing Clustering algorithm.

        Parameter:
        ----------
            traindata : array_like, shape(n_data, n_dimentions)
                Training data.
        """

        # Initialize all parameters.
        self.num_data, self.dim = traindata.shape
        # Initialize all sigma for each dimention.
        self.sigma_init = np.full(self.dim, self.sigma_initial)
        # ln(sigma^n_dim) = n_dim*sigma
        self.mu_threshold = math.log(self.threshold)*self.dim

        # Initialize group information.
        self.num_in_group = [0]
        self.num_clusters = 0
        self.group_of_data = np.full(self.num_data, -1, dtype=int)  # '-1' represents that data belongs no group.
        self.dataID_in_group = []

        # Initialize mean and sigma.
        # Only one mean for beginning
        self.mean = np.zeros((1, self.dim), dtype=float)
        self.sigma = np.zeros((1, self.dim), dtype=float)

        # Begin for the first points.
        self.mean[0] = traindata[0]  # The first mean is the first data.
        self.sigma[0] = self.sigma_init
        self.num_in_group[0] += 1
        self.num_clusters += 1
        # Add the first point to the first group.
        self.dataID_in_group.append([0])
        self.group_of_data[0] = 0   # Assign groupID to first data.

        # Begin SCC
        if self.num_data > 1:
            # Compare data with all means by Z_distances.
            z_dist = []
            # Beginning from the 2nd data.
            for data_id, data in enumerate(traindata[1:]):
                # Compare the data to current existing means.
                z_dist = [-(1.0)*self.Z_distance(data, mean, self.sigma[mean_id])
                            for mean_id, mean in enumerate(self.mean)]
                dist = np.array(z_dist)
                if dist.max() >= self.mu_threshold:
                    # Begining from 2nd data, so the data_id+1.
                    self.add_point(dist.argmax(), data_id+1, data)
                else:
                    self.add_cluster(data_id+1, data)
                z_dist = []     # Initialize z_distance for each round.

        else:
            print("Warning: There are only one data in the training set.")

    def add_point(self, group_id, data_id, data):
        """
        Add the current point to the current group.
        Updating the means and deviations of group.

        Parameter:
        ----------
            group_id :  int
                The serial number of the current group.
            data_id :   int
                The serial number of the current data.
            data :  array, shape(dim)
                The input data.
        """

        # (S-1)*([sigma]-[sigma_init])^2
        temp_A_1 = (self.num_in_group[group_id]-1) * \
            pow((self.sigma[group_id]-self.sigma_init), 2)
        # S*[mean]^2 + [data]^2
        temp_A_2 = self.num_in_group[group_id] * \
            pow(self.mean[group_id], 2)+pow(data, 2)
        # sum(above)/S
        temp_A = (temp_A_1 + temp_A_2)/self.num_in_group[group_id]
        # (S+1)/S
        temp_B = (self.num_in_group[group_id]+1)/self.num_in_group[group_id]
        # (S*[mean]+[data])/(S+1)
        temp_Mean = (
            self.num_in_group[group_id]*self.mean[group_id]+data)/(self.num_in_group[group_id]+1)

        self.sigma[group_id] = np.sqrt(np.abs(
            temp_A-(temp_B*pow(temp_Mean, 2)))) + self.sigma_init     # Updata deviation
        self.mean[group_id] = temp_Mean     # Update Mean
        self.num_in_group[group_id] += 1
        self.group_of_data[data_id] = group_id
        self.dataID_in_group[group_id].append(data_id)

    def add_cluster(self, data_id, data):
        """
        Add new cluster. The centroid of group is the data.

        Parameter:
        ----------
            data_id :   int
                The serial number of the current data.
            data :  array, shape(dim)
                The input data.
        """

        self.mean = np.concatenate(
            (self.mean, data.reshape(1, self.dim)), axis=0)
        self.sigma = np.concatenate(
            (self.sigma, self.sigma_init.reshape(1, self.dim)), axis=0)
        self.num_in_group.append(1)
        self.num_clusters += 1
        self.group_of_data[data_id] = len(self.mean)-1
        self.dataID_in_group.append([data_id])

    @classmethod
    def Z_distance(cls, data, mean, sigma):
        """
        Sum all dimensions calcuated by following function. "[]" means a vector.
        Z(i,j) = ([data(i)]-[mean(j)]/[sigma(j)])^2.

        Parameter:
        ----------
            data :  array, shape(dim)
            mean :  array, shape(dim)
            sigma : array, shape(dim)

        Return:
        -------
            sum_all_dim :   float
                Sum all dimensions.

        """
        temp = (data - mean) / sigma
        square_temp = temp*temp
        sum_all_dim = square_temp.sum()

        return sum_all_dim

    @classmethod
    def get_MD(cls, data, mean, sigma):
        """
        Get Membership degree of input data to all centroids.

        Parameter:
        ----------
            data :  array_like, shape(num_data, dim)
            mean :  array, shape(dim)
            sigma : array, shape(dim)

        Return:
        -------
            total_dist : array_like, shape(num_data, num_clusters)
        """

        total_dist = []
        for d in data:
            z_dist = [-(1.0)*cls.Z_distance(d, mean, sigma[mean_id])
                      for mean_id, mean in enumerate(mean)]
            total_dist.append(z_dist)
            
        MD = np.exp(total_dist)
        return MD


# Inherit from SCC
class SCCI(SCC):
    """
    Iterative Self-Constructing Clustering algorithm.

    parameter:
    ----------
        mean :  array_like, shape(num_cluster, dim)
            Centroids of groups after clustering.
        sigma : array_like, shape(num_cluster, dim)
            Deviations of groups after clustering.
        sigma_init : array, shape(dim)
            Inital deviation for SCC first time.
        mu_threshold :  float
            The threshold is calulated by natural log.
        group_of_data : array, shape(num_data)
            Store the group for each data.
        num_in_group :  list
            Store the numbers of data in each group.
        dataID_in_group :   list
            Store the serial number of data in each group.
        num_clusters :  int
            Store number of total clusters
        iter_times: int
            The times of iteraion after SCC_I.
    """

    def __init__(self, threshold, sigma_init=0.1, max_iter=10, limited_iteration=None):
        """
        Inherit from SCC and initialize the parameters of SCC_I.

        Parameter:
        ----------
            threshold : float
                The threshold to decide whether adding point to group or not.
            sigma_init :    float
                The initial value of deviation.
            max_iter :  float
                The maximun iterative times.
            limited_iteraion : int
                User_defined iterative times. "None" for auto-stops by SCC_I.

        """

        super().__init__(threshold, sigma_init)
        self.iteration = limited_iteration
        self.iter_times = 0
        self.max_iter = max_iter

    def scci(self, traindata):
        """
        Iterative Self-Constructing Clustering algorithm.

        Parameter:
        ----------
            traindata : array_like, shape(n_data, n_dimentions)
                Training data.

        """
        flag = 0
        # Do SCC algorithm for the first time.
        self.scc(traindata)

        # Check the result. If the result is the same as previous result, SCC_I stops.
        current_result = np.copy(self.group_of_data)
        if self.iteration == None:
            mode = 0    # SCC_I stops by the same result.
            continue_flag = 1
        else:
            mode = 1    # SCC_I stops by limited iterative times.
            continue_flag = self.iteration
        
        # The situation that only one data in training set.
        if self.num_data == 1:
            flag = continue_flag

        # Begin SCC_I
        while flag < continue_flag:
            z_dist = []
            for data_id, data in enumerate(traindata):
                # Remove the current point.
                current_group_id = self.group_of_data[data_id]
                if self.num_in_group[current_group_id] > 2:
                    self.remove_point(current_group_id, data_id, data)
                elif self.num_in_group[current_group_id] == 2:
                    # Get the remained data in the group.
                    self.dataID_in_group[current_group_id].remove(data_id)
                    remain_data_id = self.dataID_in_group[current_group_id][0]
                    remain_data = traindata[remain_data_id]
                    self.just_one_point(
                        current_group_id, remain_data_id, remain_data)
                else:
                    self.remove_cluster(current_group_id)

                # Redistribute the current point.
                z_dist = [-(1.0)*self.Z_distance(data, mean, self.sigma[mean_id])
                               for mean_id, mean in enumerate(self.mean)]
                dist = np.array(z_dist)
                if dist.max() >= self.mu_threshold:
                    self.add_point(dist.argmax(), data_id, data)
                else:
                    self.add_cluster(data_id, data)
                z_dist = []

            self.iter_times += 1
            # Check whether end of while.
            if mode == 0:
                if np.array_equal(current_result, self.group_of_data):
                    flag = continue_flag
                elif self.iter_times == self.max_iter:
                    flag = continue_flag
                    print("Reach Maximun iteration:{}".format(self.max_iter))
                else:
                    current_result = np.copy(self.group_of_data)
                    flag = 0
            elif mode == 1:
                flag += 1

    def remove_point(self, group_id, data_id, data):
        """
        Remove the current point from the current group.
        Update means and deviations of the current group.

        Parameter:
        ----------
            group_id :  int
                The serial number of the current group.
            data_id :   int
                The serial number of the current data.
            data :  array, shape(dim)
                The input data.        
        """

        # (S-1)*([sigma]-[sigma_init])^2
        temp_A_1 = (self.num_in_group[group_id]-1) * \
            pow((self.sigma[group_id]-self.sigma_init), 2)
        # S*[mean]^2-[data]^2
        temp_A_2 = self.num_in_group[group_id] * \
            pow(self.mean[group_id], 2)-pow(data, 2)
        # sum(above)/S-2
        temp_A = (temp_A_1 + temp_A_2)/(self.num_in_group[group_id]-2)
        # (S-1)/(S-2)
        temp_B = (self.num_in_group[group_id]-1) / \
            (self.num_in_group[group_id]-2)
        # (S*[mean]-[data])/(S-1)
        temp_Mean = (
            self.num_in_group[group_id]*self.mean[group_id]-data)/(self.num_in_group[group_id]-1)

        self.sigma[group_id] = np.sqrt(
            np.abs(temp_A-(temp_B*pow(temp_Mean, 2)))) + self.sigma_init
        self.mean[group_id] = temp_Mean
        self.num_in_group[group_id] -= 1
        self.dataID_in_group[group_id].remove(data_id)

    def remove_cluster(self, group_id):
        """
        Remove the current group.
        The deleted group is covered by the last group and remove the last one.
        If the current group is the last group, just remove last one.

        Parameter:
        ----------
            group_id :  int
                The serial number of the current group.     
        """

        if group_id < (self.num_clusters-1):
            self.mean[group_id] = self.mean[-1]
            self.mean = np.delete(self.mean, -1, axis=0)
            self.sigma[group_id] = self.sigma[-1]
            self.sigma = np.delete(self.sigma, -1, axis=0)
            self.num_in_group[group_id] = self.num_in_group[-1]
            self.num_in_group.pop(-1)
            self.dataID_in_group[group_id] = self.dataID_in_group[-1]
            self.dataID_in_group.pop(-1)
            for did in self.dataID_in_group[group_id]:
                self.group_of_data[did] = group_id
        else:
            self.mean = np.delete(self.mean, -1, axis=0)
            self.sigma = np.delete(self.sigma, -1, axis=0)
            self.num_in_group.pop(-1)
            self.dataID_in_group.pop(-1)

        self.num_clusters -= 1

    def just_one_point(self, group_id, remain_data_id, remain_data):
        """
        There is only one point in group after removing. The centroid of group is the remained data.

        Parameter:
        ----------
            group_id :  int
                The serial number of the current group.
            remain_data_id :   int
                The serial number of the remained data.
            remain_data :  array, shape(dim)
                The remained data.
        """
        self.mean[group_id] = remain_data
        self.sigma[group_id] = self.sigma_init
        self.num_in_group[group_id] = 1


class SCCI_enhance(SCCI):
    alpha = 1
    def __init__(self, threshold, sigma_init=0.1, deviation_alpha=1 , limited_iteration=None, max_iter=10):
        super().__init__(threshold, sigma_init, max_iter, limited_iteration)
        self.deviation_alpha = deviation_alpha
        self.set_alpha(deviation_alpha)

    # def add_point(self, group_id, data_id, data):
    #     """
    #     Add the current point to the current group.
    #     Updating the means and deviations of group.

    #     Parameter:
    #     ----------
    #         group_id :  int
    #             The serial number of the current group.
    #         data_id :   int
    #             The serial number of the current data.
    #         data :  array, shape(dim)
    #             The input data.
    #     """
        
    #     if self.num_in_group[group_id] == 1:
    #         exist_data = np.copy(self.mean[group_id])    # The centroid is the point itself. 
    #         self.mean[group_id] = (exist_data+data)/2   # Update mean
    #         temp_exist_point = pow((self.mean[group_id] - exist_data), 2)
    #         temp_input_point = pow((self.mean[group_id] - data), 2)
    #         self.sigma[group_id] = np.sqrt((temp_exist_point+temp_input_point))   # Update deviation
    #     else:
    #         # (S-1)*[sigma]^2
    #         temp_A_1 = (self.num_in_group[group_id]-1) * \
    #             pow(self.sigma[group_id], 2)
    #         # S*[mean]^2 + [data]^2
    #         temp_A_2 = self.num_in_group[group_id] * \
    #             pow(self.mean[group_id], 2)+pow(data, 2)
    #         # sum(above)/S
    #         temp_A = (temp_A_1 + temp_A_2)/self.num_in_group[group_id]
    #         # (S+1)/S
    #         temp_B = (self.num_in_group[group_id]+1)/self.num_in_group[group_id]
    #         # (S*[mean]+[data])/(S+1)
    #         temp_Mean = (
    #             self.num_in_group[group_id]*self.mean[group_id]+data) / \
    #                 (self.num_in_group[group_id]+1)
    
    #         self.sigma[group_id] = np.sqrt(np.abs(
    #             temp_A-(temp_B*pow(temp_Mean, 2))))     # Updata deviation
    #         self.mean[group_id] = temp_Mean     # Update mean
            
    #     self.num_in_group[group_id] += 1
    #     self.group_of_data[data_id] = group_id
    #     self.dataID_in_group[group_id].append(data_id)

    # def remove_point(self, group_id, data_id, data):
    #     """
    #     Remove the current point from the current group.
    #     Update means and deviations of the current group.

    #     Parameter:
    #     ----------
    #         group_id :  int
    #             The serial number of the current group.
    #         data_id :   int
    #             The serial number of the current data.
    #         data :  array, shape(dim)
    #             The input data.        
    #     """

    #     # (S-1)*[sigma]^2
    #     temp_A_1 = (self.num_in_group[group_id]-1) * \
    #         pow(self.sigma[group_id], 2)
    #     # S*[mean]^2-[data]^2
    #     temp_A_2 = self.num_in_group[group_id] * \
    #         pow(self.mean[group_id], 2)-pow(data, 2)
    #     # sum(above)/S-2
    #     temp_A = (temp_A_1 + temp_A_2)/(self.num_in_group[group_id]-2)
    #     # (S-1)/(S-2)
    #     temp_B = (self.num_in_group[group_id]-1) / \
    #         (self.num_in_group[group_id]-2)
    #     # (S*[mean]-[data])/(S-1)
    #     temp_Mean = (
    #         self.num_in_group[group_id]*self.mean[group_id]-data)/(self.num_in_group[group_id]-1)

    #     self.sigma[group_id] = np.sqrt(
    #         np.abs(temp_A-(temp_B*pow(temp_Mean, 2))))  # Update sigma
    #     self.mean[group_id] = temp_Mean     # Update mean
    #     self.num_in_group[group_id] -= 1
    #     self.dataID_in_group[group_id].remove(data_id)

    @classmethod
    def set_alpha(cls, alpha):
        cls.alpha = alpha

    @classmethod
    def Z_distance(cls, data, mean, sigma, deviation_alpha=None):
        """
        Sum all dimensions calcuated by following function. "[]" means a vector.
        Z(i,j) = ([data(i)]-[mean(j)]/[sigma])^2.

        Parameter:
        ----------
            data :  array, shape(dim)
            mean :  array, shape(dim)
            sigma : array, shape(dim)

        Return:
        -------
            sum_all_dim :   float
                Sum all dimensions.

        """
        if deviation_alpha == None:
            alpha = cls.alpha
        else:
            alpha = deviation_alpha
        temp = (data - mean) / (alpha*sigma)
        square_temp = temp*temp
        sum_all_dim = square_temp.sum()
        #print(cls.__name__)

        return sum_all_dim

    def get_no_init_sigma(self):
        temp = self.sigma - self.sigma_init

        arg_zero = np.argwhere(temp == 0.0)
        if len(arg_zero) != 0:
            for indx in arg_zero:
                temp[indx[0]][indx[1]] = self.sigma_initial
        
        return temp

        
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
    alpha = 1

    Y_label_num = len(set(list(Y_train)))
    Y_train_class = []  # Store the same label IDs.
    for i in range(Y_label_num):
        Y_train_class.append(
            [c for c, data in enumerate(Y_train) if data == i])

    # SCC_I method
    X_train_class = []  # Store SCC result for each label.
    for i in range(Y_label_num):
        if len(X_train[Y_train_class[i]]) != 0:
            cluster_method = SCCI_enhance(threshold=threshold, sigma_init=sigma_init, deviation_alpha=alpha)
            cluster_method.scci(X_train[Y_train_class[i]])
            X_train_class.append(cluster_method)

    # Concatenate all centroid of SCC to a total Mean and Sigma.
    c_t_mean = None
    for scc_i in X_train_class:
        if c_t_mean is None:
            c_t_mean = scc_i.mean
            c_t_sigma = scc_i.sigma
        else:
            c_t_mean = np.concatenate((c_t_mean, scc_i.mean), axis=0)
            c_t_sigma = np.concatenate((c_t_sigma, scc_i.sigma), axis=0)
