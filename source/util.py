import sys

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import rpy2.robjects as robjects


def compute_class_weights(labels):
    # create a 2 by M array with weights for 0's and 1's
    weights = np.zeros((2, labels.shape[1]))
    # calculates weights for each label in a for loop
    for i in range(labels.shape[1]):
        weights_n, weights_p = (labels.shape[0] / (2 * (labels[:, i] == 0).sum())), (
                labels.shape[0] / (2 * (labels[:, i] == 1).sum()))
        weights[1, i], weights[0, i] = weights_p, weights_n
    return weights


def get_data(base_path, series_type, period):
    readRDS = robjects.r['readRDS']
    if period == 'train':
        df_series_rds = readRDS(base_path + 'val/train/' + 'series_' + series_type + '.rds')
    elif period == 'test':
        df_series_rds = readRDS(base_path + 'test/' + 'series_test_' + series_type + '.rds')
    else:
        sys.exit('Invalid period: choose between "train" or "test" periods')
    series_list = list(map(lambda df: {key: np.asarray(df.rx2(key)) for key in df.names}, df_series_rds))
    ts_train = list(map(lambda x: x['ts_train'], series_list))
    ts_test = list(map(lambda x: x['ts_test'], series_list))
    mat_forecast = list(map(lambda x: x['mat_forecast'], series_list))
    if period == 'train':
        div_label = list(map(lambda x: x['div_label'], series_list))
    elif period == 'test':
        div_label = []
    else:
        sys.exit('Invalid period: choose between "train" or "test" periods')
    return ts_train, ts_test, mat_forecast, div_label


def get_data_LargeST(base_path, series_type, period):
    readRDS = robjects.r['readRDS']
    if period == 'train':
        df_series_rds = readRDS(base_path + 'LargeST/' + 'series_' + series_type + '.rds')
    elif period == 'test':
        df_series_rds = readRDS(base_path + 'LargeST/' + 'series_test_' + series_type + '.rds')
    else:
        sys.exit('Invalid period: choose between "train" or "test" periods')
    series_list = list(map(lambda df: {key: np.asarray(df.rx2(key)) for key in df.names}, df_series_rds))
    ts_train = list(map(lambda x: x['ts_train'], series_list))
    ts_test = list(map(lambda x: x['ts_test'], series_list))
    mat_forecast = list(map(lambda x: x['mat_forecast'], series_list))
    if period == 'train':
        div_label = list(map(lambda x: x['div_label'], series_list))
    elif period == 'test':
        div_label = []
    else:
        sys.exit('Invalid period: choose between "train" or "test" periods')
    return ts_train, ts_test, mat_forecast, div_label


def plot_len_hist(len_list, series_type):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.hist(len_list, bins=100, edgecolor='b')
    ax.axvline(np.percentile(len_list, 50), color='k', linestyle='-', linewidth=1)
    ax.axvline(np.percentile(len_list, 90), color='r', linestyle='--', linewidth=1)
    plt.xticks(np.arange(0, max(len_list) + 1, 64))
    plt.xlim([0, 512])
    plt.title('Length distribution: ' + series_type)
    plt.show()


def standardize_series(x, x_mean=None, x_std=None):
    if x_mean is None and x_std is None:
        x_mean = np.mean(x)
        x_std = np.std(x)
    x_norm = (np.copy(x) - x_mean) / x_std
    return x_norm.tolist(), x_mean, x_std