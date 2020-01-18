#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization

import keras
from keras.models import Sequential
from keras import optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model

import time
from keras.callbacks import TensorBoard


def select_group(group_Num, seq_length):
    GJ_2009 = pd.read_csv("./data/final_group_2009.csv")
    GJ_2010 = pd.read_csv("./data/final_group_2010.csv")
    GJ_2011 = pd.read_csv("./data/final_group_2011.csv")
    GJ_2012 = pd.read_csv("./data/final_group_2012.csv")
    GJ_2013 = pd.read_csv("./data/final_group_2013.csv")
    GJ_2014 = pd.read_csv("./data/final_group_2014.csv")
    GJ_2015 = pd.read_csv("./data/final_group_2015.csv")
    GJ_2016 = pd.read_csv("./data/final_group_2016.csv")
    GJ_2017 = pd.read_csv("./data/final_group_2017.csv")

    # 2016년도는 알코올 변수가 없어서 제거

    del GJ_2009['alcohol_2009']
    del GJ_2010['alcohol_2010']
    del GJ_2011['alcohol_2011']
    del GJ_2012['alcohol_2012']
    del GJ_2013['alcohol_2013']
    del GJ_2014['alcohol_2014']
    del GJ_2015['alcohol_2015']
    # del GJ_2016['alcohol_2016']
    del GJ_2017['alcohol_2017']

    group_01_Y = pd.concat([pd.Series(GJ_2009.loc[group_Num, 'target']), pd.Series(GJ_2010.loc[group_Num, 'target']),
                            pd.Series(GJ_2011.loc[group_Num, 'target']), pd.Series(GJ_2012.loc[group_Num, 'target']),
                            pd.Series(GJ_2013.loc[group_Num, 'target']), pd.Series(GJ_2014.loc[group_Num, 'target']),
                            pd.Series(GJ_2015.loc[group_Num, 'target']), pd.Series(GJ_2016.loc[group_Num, 'target']),
                            pd.Series(GJ_2017.loc[group_Num, 'target'])])

    del GJ_2009['target']
    del GJ_2010['target']
    del GJ_2011['target']
    del GJ_2012['target']
    del GJ_2013['target']
    del GJ_2014['target']
    del GJ_2015['target']
    del GJ_2016['target']
    del GJ_2017['target']

    GJ_01 = pd.concat([GJ_2009.iloc[group_Num, :], GJ_2010.iloc[group_Num, :], GJ_2011.iloc[group_Num, :]
                          , GJ_2012.iloc[group_Num, :], GJ_2013.iloc[group_Num, :], GJ_2014.iloc[group_Num, :]
                          , GJ_2015.iloc[group_Num, :], GJ_2016.iloc[group_Num, :], GJ_2017.iloc[group_Num, :]], axis=0)

    group_01_X = pd.DataFrame(np.array(GJ_01).reshape(9, 29))
    #     group_01_Y = group_01_Y.loc[group_Num]
    group_01_Y = np.array(group_01_Y)

    #     return group_01_Y

    dataX = []
    dataY = []

    for i in range(0, int(len(group_01_Y) - seq_length)):
        _x = group_01_X[i: i + seq_length]
        _x = np.array(_x)
        _y = group_01_Y[i + seq_length]
        _y = np.array(_y)
        dataX.append(_x)  # dataX 리스트에 추가 / add to dataX's list
        dataY.append(_y)  # dataY 리스트에 추가 / add to dataY's list

    # train/test split
    train_size = int(len(dataY) * 0.9)
    test_size = len(dataY) - train_size

    # 데이터를 잘라 학습용 데이터 생성
    trainX = np.array(dataX[0:train_size])
    trainY = np.array(dataY[0:train_size])

    # 데이터를 잘라 테스트용 데이터 생성
    testX = np.array(dataX[train_size:len(dataX)])
    testY = np.array(dataY[train_size:len(dataY)])

    return trainX, trainY, testX, testY, group_01_X


def model_train(trainX, trainY, testX, testY, epochs=100):
    ## LSTM 모델
    seq_length = 3
    input_columns = 29

    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_length, input_columns), return_sequences=True, stateful=False))
    model.add(LSTM(128, return_sequences=False, stateful=False))
    model.add(Dense(1))
    model.add(Activation('linear'))
    # , dropout=0.2

    #     model.summary()
    # 모델 학습 설정 및 진행
    keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    print("=" * 50)
    hist = model.fit(trainX, trainY, epochs=epochs, batch_size=30, verbose=1, validation_data=(testX, testY))

    return model, hist

def y_pred(testX, testY, model):
    y_pred = model.predict(testX, batch_size=1, verbose=1)
    a = np.array(y_pred).reshape(testY.shape[0])
    b = np.array(testY).reshape(testY.shape[0])
    result = pd.DataFrame([a,b], index=['y_pred', 'real_y'])
    RMSE = np.sqrt(mean_squared_error(testY, y_pred))
    MAE = mean_absolute_error(testY, y_pred)
    MAPE = np.mean(np.abs((testY - y_pred) / testY)) * 100
    return result, RMSE, MAE, MAPE


def make_testset(group_01_X):
    # 2018년 Y예측
    predict_next_X = list()

    temp = group_01_X[-3:]
    temp = np.array(temp)
    predict_next_X.append(temp)

    predict_next_X = np.array(predict_next_X)

    return predict_next_X

