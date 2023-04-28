
import scipy
import scipy.io
from .utils import *
import torch
import numpy as np

def medical_data(data_name,normal_class, root='DATA/',
                 contamination_rate=0.0):
    root = root + data_name+'/'
    trainset = torch.load(root + 'trainset_2048.pt')
    train_data, train_targets = trainset
    testset = torch.load(root + 'testset_2048.pt')
    test_data, test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets == normal_class] = 0

    train_clean = train_data[np.where(train_targets == normal_class)]
    num_clean = train_clean.shape[0]

    #### Other classes as anomalies ####
    train_contamination = train_data[np.where(train_targets !=normal_class)]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)
    # rng = np.random.RandomState(123)
    idx_contamination = np.random.choice(np.arange(train_contamination.shape[0]),num_contamination,replace=False)
    train_contamination = train_contamination[idx_contamination]
    train_data = torch.cat((train_clean, train_contamination), 0)

    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:] = 1
    # rng = np.random.RandomState(123)
    idx_permute = np.random.permutation(np.arange(train_data.shape[0]))
    train_data = train_data[idx_permute]
    train_labels = train_labels[idx_permute]
    return train_data, train_labels, test_data, test_labels

def tabular_data(name_of_file,contamination_rate):
    name_of_file = "DATA/" + name_of_file
    dataset = scipy.io.loadmat(name_of_file)
    X = dataset['X']
    classes = dataset['y']
    dim = X.shape[1]
    jointXY = np.concatenate((X, classes,), 1)
    normals=jointXY[jointXY[:,-1]==0]
    anomalies=jointXY[jointXY[:,-1]==1]
    normals = normals[np.random.permutation(normals.shape[0])]

    train_norm = normals[:int(normals.shape[0] / 2) + 1]
    test_norm = normals[int(normals.shape[0] / 2) + 1:]

    num_clean = train_norm.shape[0]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)
    train_abnorm = anomalies[:num_contamination]
    test_abnorm = anomalies[num_contamination:]

    train = np.concatenate((train_norm, train_abnorm),0)
    train_labels = train[:, -1]
    train = train[:, :dim]

    test = np.concatenate((test_norm, test_abnorm),0)
    test_labels = test[:, -1]
    test = test[:, :dim]

    idx_permute = np.random.permutation(np.arange(train.shape[0]))
    train = train[idx_permute]
    train_labels = train_labels[idx_permute]

    train = torch.tensor(train)
    test = torch.tensor(test)
    return train, train_labels,test, test_labels

def CIFAR10_feat(normal_class, root='DATA/',
                 contamination_rate=0.0):
    trainset = torch.load(root + 'trainset_2048.pt')
    train_data, train_targets = trainset
    testset = torch.load(root + 'testset_2048.pt')
    test_data, test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets == normal_class] = 0

    train_clean = train_data[np.where(train_targets == normal_class)]
    num_clean = train_clean.shape[0]

    #### Other classes as anomalies ####
    train_contamination = train_data[np.where(train_targets !=normal_class)]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)
    # rng = np.random.RandomState(123)
    idx_contamination = np.random.choice(np.arange(train_contamination.shape[0]),num_contamination,replace=False)
    train_contamination = train_contamination[idx_contamination]
    train_data = torch.cat((train_clean, train_contamination), 0)

    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:] = 1
    # rng = np.random.RandomState(123)
    idx_permute = np.random.permutation(np.arange(train_data.shape[0]))
    train_data = train_data[idx_permute]
    train_labels = train_labels[idx_permute]
    return train_data, train_labels, test_data, test_labels

def FMNIST_feat(normal_class, root='DATA/',
                 contamination_rate=0.0):
    trainset = torch.load(root + 'trainset_2048.pt')
    train_data, train_targets = trainset
    testset = torch.load(root + 'testset_2048.pt')
    test_data, test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets == normal_class] = 0

    train_clean = train_data[np.where(train_targets == normal_class)]
    num_clean = train_clean.shape[0]

    #### Other classes as anomalies ####
    train_contamination = train_data[np.where(train_targets !=normal_class)]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)
    # rng = np.random.RandomState(123)
    idx_contamination = np.random.choice(np.arange(train_contamination.shape[0]),num_contamination,replace=False)
    train_contamination = train_contamination[idx_contamination]
    train_data = torch.cat((train_clean, train_contamination), 0)

    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:] = 1
    # rng = np.random.RandomState(123)
    idx_permute = np.random.permutation(np.arange(train_data.shape[0]))
    train_data = train_data[idx_permute]
    train_labels = train_labels[idx_permute]
    return train_data, train_labels, test_data, test_labels


def load_data(data_name,cls,contamination_rate=0.0):

    tabular_dataset = ['pima','breastw','ionosphere','satellite']
    medical_dataset = ['blood','organa','organs','organc','path','derma','oct','pneumonia', 'tissue']
    ## normal data with label 0, anomalies with label 1
    if data_name == 'cifar10_feat':
        train, train_label, test, test_label = CIFAR10_feat(cls,contamination_rate=contamination_rate)
    if data_name == 'fmnist_feat':
        train, train_label, test, test_label = FMNIST_feat(cls,contamination_rate=contamination_rate)
    elif data_name in tabular_dataset:
        train, train_label, test, test_label = tabular_data(data_name,contamination_rate=contamination_rate)
    elif data_name in medical_dataset:
        train, train_label, test, test_label = medical_data(data_name,cls,contamination_rate=contamination_rate)

    trainset = CustomDataset(train,train_label)
    testset = CustomDataset(test,test_label)
    return trainset,testset,testset






