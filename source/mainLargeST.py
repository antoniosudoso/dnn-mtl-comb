import os

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
from util import standardize_series, get_data_LargeST, compute_class_weights
from networks import build_baseline, build_network
import argparse

BASE_PATH = '/home/antonio/Dropbox/Metalearning/'

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-SERIES_TYPE", type=str, help="Time series type", required=True)
    argParser.add_argument("-MAX_LEN", type=int, help="Time series length", required=True)
    argParser.add_argument("-LAMBDA", type=float, help="Lambda hyperparameter", required=True)
    argParser.add_argument("-SCALE_LOSS", type=int, help="Scale forecast combination loss", required=True)

    args = argParser.parse_args()
    print(args)

    SERIES_TYPE = args.SERIES_TYPE
    MAX_LEN = args.MAX_LEN
    LAMBDA = args.LAMBDA
    SCALE_LOSS = bool(args.SCALE_LOSS)

    print('Command line arguments:')
    print('\tSeries type:', SERIES_TYPE)
    print('\tSequence length:', MAX_LEN)
    print('\tLambda hyperparameter:', LAMBDA)
    print('\tScale loss:', SCALE_LOSS)

    print('Reading ' + SERIES_TYPE + ' time series...')
    ts_train_list, ts_test_list, mat_forecast_list, div_label_list = get_data_LargeST(BASE_PATH, SERIES_TYPE, period='train')
    print('\tDone!')

    # len_list = np.array(list(map(lambda x: len(x), ts_train_list)))
    # plot_len_hist(len_list)

    print('Processing ' + SERIES_TYPE + ' time series...')
    train_std_data = tuple(map(lambda x: standardize_series(x), ts_train_list))
    ts_train_list_std = list(map(lambda x: x[0], train_std_data))
    print('\tstd train series:', len(ts_train_list_std))
    ts_test_list_std = np.array(ts_test_list)
    print('\tstd test series:', len(ts_train_list_std))
    ts_train = tf.keras.preprocessing.sequence.pad_sequences(ts_train_list_std, maxlen=MAX_LEN, dtype='float64', padding="pre", truncating="pre", value=0.0)
    ts_train = ts_train.reshape((ts_train.shape[0], ts_train.shape[1], 1))
    ts_test = ts_test_list_std.copy()
    ts_mat_forecast = np.array(mat_forecast_list)
    ts_div_label = np.array(div_label_list)

    cls_weights = compute_class_weights(ts_div_label)
    print(cls_weights)

    print('\ttrain series shape:', ts_train.shape)
    print('\ttest series shape:', ts_test.shape)
    print('\tforecast matrix shape:', ts_mat_forecast.shape)
    print('\tdiversity label shape:', ts_div_label.shape)

    F_METHODS = ts_mat_forecast.shape[1]
    print('Methods:', F_METHODS)
    F_HORIZON = ts_mat_forecast.shape[2]
    print('Horizon:', F_HORIZON)

    f_model, w_model, reg_model, cls_model = build_network(MAX_LEN, F_METHODS, F_HORIZON, LAMBDA, 0.0, scale_loss=SCALE_LOSS, class_weights=cls_weights)
    f_model.summary()
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    # # This callback will stop the training when there is no improvement in the loss for consecutive epochs.
    history = f_model.fit([ts_train, ts_mat_forecast, ts_div_label], ts_test, epochs=500, shuffle=True, batch_size=32, validation_split=0.2, callbacks=[es_callback])

    if SCALE_LOSS:
        f_model.save('./saved_model_LargeST/' + SERIES_TYPE + '/main/f_model_' + str(MAX_LEN) + '_' + str(LAMBDA))
        w_model.save('./saved_model_LargeST/' + SERIES_TYPE + '/main/w_model_' + str(MAX_LEN) + '_' + str(LAMBDA))
        reg_model.save('./saved_model_LargeST/' + SERIES_TYPE + '/main/reg_model_' + str(MAX_LEN) + '_' + str(LAMBDA))
        cls_model.save('./saved_model_LargeST/' + SERIES_TYPE + '/main/cls_model_' + str(MAX_LEN) + '_' + str(LAMBDA))
        history_df = pd.DataFrame(history.history)
        history_df.to_csv('./saved_model_LargeST/' + SERIES_TYPE + '/main/history_' + str(MAX_LEN) + '_' + str(LAMBDA) + '.csv', index=False)
    else:
        f_model.save('./saved_model_LargeST/' + SERIES_TYPE + '/main_no_scale/f_model_' + str(MAX_LEN) + '_' + str(LAMBDA))
        w_model.save('./saved_model_LargeST/' + SERIES_TYPE + '/main_no_scale/w_model_' + str(MAX_LEN) + '_' + str(LAMBDA))
        reg_model.save('./saved_model_LargeST/' + SERIES_TYPE + '/main_no_scale/reg_model_' + str(MAX_LEN) + '_' + str(LAMBDA))
        cls_model.save('./saved_model_LargeST/' + SERIES_TYPE + '/main_no_scale/cls_model_' + str(MAX_LEN) + '_' + str(LAMBDA))
        history_df = pd.DataFrame(history.history)
        history_df.to_csv('./saved_model_LargeST/' + SERIES_TYPE + '/main_no_scale/history_' + str(MAX_LEN) + '_' + str(LAMBDA) + '.csv', index=False)

    print('Saved!')