import re
import numpy as np
from sklearn import datasets
from scipy.io import arff
import pandas as pd
import xml.etree.ElementTree as ET

def bin2matrix(label):
    label_matrix = np.zeros(label.shape)
    for i, row in enumerate(label):
        for j, element in enumerate(row):
            if element == b'1':
                label_matrix[i][j] = 1
            elif element == b'0':
                label_matrix[i][j] = 0
    return label_matrix

def target2matrix(target_label, pos_note=1.0, neg_note=0.0):
    classes = list(set(target_label))
    num_classes = len(classes)
    label_dict = dict.fromkeys(classes)
    for idx, lbl in enumerate(classes):
        label_dict[lbl] = idx
        
    target_matrix = neg_note * \
        np.ones([len(target_label), num_classes], dtype=float)
    for serial, target in enumerate(target_label):
        target_matrix[serial][label_dict[target]] = pos_note
    return target_matrix

class load_datasetSL:
    """
    Load dataset in path './datasets/'.
    The dataset is downloaded from UCI dataset.

    Parameter:
    ----------
        data_name: string
            Name of file in path './datasets/'. Such as 'iris', 'yeast' etc. Check the datasets

    Member:
    -------
        data: {M, M, ..., M} ndarray, shape(n_samples, n_features)
            Matrix for all data features. 'M' for feature as array.
        target: (i, i, ..., i) array_like, shape(n_samples)
            The Labels of data. 'i' for label as integer.
        label_dict: dict
            You can check the original label from target.
     
    """

    def __init__(self, data_name):

        data = []
        target = []
        data_separate_by_comma = ['iris', 'libras', 'sonar', 'soybeanS', 'lung-cancer', 'bupa', 'ionosphere', 'LSVT', 'musk']
        data_separate_by_space = ['ecoli', 'yeast', 'heart']
        data_from_datasets = ['wine', 'breast']

        if data_name == "glass":
            with open('datasets/glass.data') as f:
                for line in f:
                    temp = line.split(',')
                    temp.pop(0)
                    target.append(temp.pop())
                    data.append([float(num) for num in temp])

        elif data_name in data_separate_by_comma:
            with open('datasets/'+data_name+'.data') as f:
                for line in f:
                    line = re.sub('\n', '', line)
                    temp = line.split(',')
                    if len(temp) > 2:
                        if re.search(r'[A-Za-z]+', temp[0]) is not None:
                            temp.pop(0)
                        if data_name == 'lung-cancer':
                            target.append(temp.pop(0))
                        elif data_name == 'musk':
                            temp.pop(0)
                            target.append(temp.pop())
                        else:
                            target.append(temp.pop())
                        data.append([float(num) for num in temp])

        elif data_name in data_separate_by_space:
            with open('datasets/'+data_name+'.data') as f:
                for line in f:
                    line = re.sub('\n', '', line)
                    temp = line.split()
                    if len(temp) > 2:
                        if re.search(r'[A-Za-z]+', temp[0]) is not None:
                            temp.pop(0)
                        target.append(temp.pop())
                        data.append([float(num) for num in temp])

        elif data_name in data_from_datasets:
            if data_name == 'breast':
                dataset = datasets.load_breast_cancer()
            elif data_name == 'wine':
                dataset = datasets.load_wine()
            data = dataset.data
            target = dataset.target

        else:
            print("Not ready for {name} datasets.".format(name=data_name))
            self.data = None
            self.target = None
            self.label_dict = None

        if data_name not in data_from_datasets:
            label_set = list(set(target))
            label_dict = dict.fromkeys(target)
            for idx, lbl in enumerate(label_set):
                label_dict[lbl] = idx

            self.data = np.array(data)
            self.target = np.array([label_dict[lb] for lb in target])
            self.label_dict = label_dict
            
        elif data_name in data_from_datasets:
            self.data = data
            self.target = target
            self.label_dict = dataset.target_names


class load_datasetML:
    """
    Load dataset in path './datasets/'.
    The dataset is downloaded from UCI dataset.

    Parameter:
    ----------
        data_name: string
            Name of file in path './datasets/'. Such as 'iris', 'yeast' etc. Check the datasets

    Member:
    -------
        data: {M, M, ..., M} ndarray, shape(n_samples, n_features)
            Matrix for all data features. 'M' for feature as array.
        target: (i, i, ..., i) array_like, shape(n_samples)
            The Labels of data. 'i' for label as integer.
        label_dict: dict
            You can check the original label from target.
     
    """

    def __init__(self, data_name):

        filename = ['birds', 'scene', 'emotions', 'flags', 'yeastML']
        data_with_nominal = ['birds', 'flags']

        flpath = "./datasets/{dn}/".format(dn=data_name)
        data_path = flpath+data_name+".arff"
        
        if data_name in filename:
            with open(data_path) as f:
                temp = arff.loadarff(f)
                pd_data = pd.DataFrame(temp[0])
            
            xmlfile = ET.parse(flpath+data_name+".xml")
            root = xmlfile.getroot()
            self.n_category = len(root)
            
            if data_name not in data_with_nominal:
                data = pd_data.iloc[:,:-(self.n_category)].values
                target = pd_data.iloc[:,-(self.n_category):]
            else:
                if data_name == 'birds':
                    nominal = ["hasSegments", "location"]
                elif data_name == 'flags':
                    nominal = ["landmass", "zone", "language", "religion", 
                                "crescent", "triangle", "icon", "animate", "text"]
                nominal_data = pd_data[nominal].T.values
                nominal_sets = []
                for attrib in nominal_data:
                    if len(nominal_sets) == 0:
                        nominal_sets = target2matrix(attrib)
                    else:
                        nominal_sets = np.concatenate((nominal_sets, target2matrix(attrib)), axis=1)
                        
                data = pd_data.drop(labels=nominal, axis=1).iloc[:,:-(self.n_category)].values
                data = np.concatenate((data, nominal_sets), axis=1)
                target = pd_data.iloc[:,-(self.n_category):]
                    
            label_dict = dict.fromkeys(list(target))
            for idx, lbl in enumerate(list(target)):
                label_dict[lbl] = idx

            self.data = data
            self.target = bin2matrix(target.values)
            self.label_dict = label_dict


if __name__ == "__main__":
    data_name = 'birds'
    dataset = load_datasetML(data_name)
    data = dataset.data
    target = dataset.target
            
                
            