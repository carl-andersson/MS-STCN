import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import fnmatch
import re
from lxml import etree
from pathlib import Path

from data.base import SeqDataset, Statistics

def plot_scatter_example(X, y=None, equal=True, show=False, save=False,
                         save_name="tmp.png"):

    rgba_colors = np.zeros((len(X), 4))
    normed = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # for red the first column needs to be one
    rgba_colors[:, 0] = normed[:, 0]

    # for blue last color column needs to be one
    rgba_colors[:, 2] = np.abs(1 - normed[:, 0])

    # the fourth column needs to be alphas
    rgba_colors[:, 3] = np.ones((len(X),)) * .4 + .4 * normed[:, 0]

    if len(X[0]) == 3:
        plt.scatter(X[:, 1], X[:, 2], color=rgba_colors)
    elif len(X[0]) == 2:
        plt.scatter(X[:, 0], X[:, 1], color=rgba_colors)

    if y is not None:
        plt.title(y)

    if equal:
        plt.axis('equal')

    if show:
        if save is True:
            raise ValueError("save cannot be True if show is True!")
        plt.show()
    elif save:
        plt.savefig(save_name)


def plot_lines_example(X, y=None, equal=True, show=False, save=False,
                               save_name="tmp.png", diff=False):
    if diff:
        X[:, 1:] = np.cumsum(X[:, 1:], 0)
        X = np.concatenate([np.array([[0, 0, 0]]), X], 0)

    penup_index = np.where(X[:, 0] == 1)[0] + 1

    prev_penup = 0
    for penup in penup_index:
        plt.plot(X[prev_penup:penup, 1], X[prev_penup:penup, 2], color='k')
        prev_penup = penup


    if y is not None:
        plt.title(y)

    if equal:
        plt.axis('equal')

    if show:
        if save is True:
            raise ValueError("save cannot be True if show is True!")
        plt.show()
    elif save:
        plt.savefig(save_name)


def getData(path, training_validation_split_ratio=0.95):
    train = path.joinpath("deepwriting_training.npz")
    test = path.joinpath("deepwriting_validation.npz")

    with np.load(train, allow_pickle=True) as data:
        train_val = data["samples"]
        train_val = [np.concatenate([d[:, 2:], d[:, :1], -d[:, 1:2]], 1) for d in train_val]

    with np.load(test, allow_pickle=True) as data:
        test = data["samples"]
        test = [np.concatenate([d[:, 2:], d[:, :1], -d[:, 1:2]], 1) for d in test]

    rng = np.random.RandomState(1)
    rng.shuffle(train_val)

    train = train_val[:int(len(train_val)*training_validation_split_ratio)]
    valid = train_val[int(len(train_val)*training_validation_split_ratio):]


    return (train, valid, test,)


def calcStatistics(data):
    cat_data = np.concatenate(data, 0)
    mean = np.mean(cat_data, 0, keepdims=True)
    std = np.std(cat_data, 0, ddof=1, keepdims=True)

    # Change from TxC to CxT
    mean = np.transpose(mean)
    std = np.transpose(std)

    return Statistics(mean, std)

def normalize_scale(all_data, stats):
    res = []
    for data in all_data:
        res.append(np.concatenate([data[:, :1], (data[:, 1:])/stats.std[1:, 0]], 1))
    return res


def addMeasNoise(all_data, measure_noise):
    res = []
    for data in all_data:
        noise = (np.random.uniform(size=data[:, 1:].shape) - 0.5) * measure_noise
        res.append(np.concatenate([data[:, :1], data[:, 1:] + noise], 1))
    return res


def getDatasets(dataset_basepath, data_path, seq_len_train, seq_len_valid, seq_len_test, min_padding_ratio):
    # Testset not implemented yet, neither are outputs
    data_path = Path(dataset_basepath).joinpath(data_path).expanduser()
    train_x, valid_x, test_x = getData(data_path)


    # Calculate statistics for normalization
    true_stats = calcStatistics(train_x)

    # Add measurement noise to avoid overfitting. The numbers below are the resolution
    measure_noise = np.array([1.37478113e-04, 1.10030182e-04])
    train_x = addMeasNoise(train_x, measure_noise)

    # We do not normalize the mean here since this can be done in the model instead.
    # This still gives the same estimates of the log likelihood
    train_x = normalize_scale(train_x, true_stats)
    valid_x = normalize_scale(valid_x, true_stats)
    test_x = normalize_scale(test_x, true_stats)
    model_stats = calcStatistics(train_x)




    trainset = SeqDataset(train_x, seq_len_train, min_padding_ratio=min_padding_ratio)

    validset = SeqDataset(valid_x, seq_len_valid, min_padding_ratio=min_padding_ratio)
    trainset_eval = SeqDataset(train_x[:len(valid_x)], seq_len_valid)

    testset = SeqDataset(test_x, seq_len_test, min_padding_ratio=min_padding_ratio)

    return trainset, trainset_eval, validset, testset, model_stats



if __name__ == "__main__":
    dataset_basepath = "/home/calle/datasets"

    data_path = Path(dataset_basepath).joinpath("deepwriting").expanduser()
    train_x, valid_x, test_x = getData(data_path)

    trainset, trainset_eval, validset, testset, model_stats = getDatasets(dataset_basepath,
                                                                         "deepwriting", 200, 200, 200, 0.9)

    stats = calcStatistics(train_x)

    measure_noise = np.array([1.37478113e-04, 1.10030182e-04])
    train_x = addMeasNoise(train_x, measure_noise)

    #train_x = normalize_scale(train_x, stats)

    print(train_x[0])
    



