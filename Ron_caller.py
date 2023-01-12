# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:15:40 2020

@author: Chance De Smet

This method is used as a calling function to the synthetoc PPDM generator
RON GAUSS, giving MAOGAN and MAOGAN RL an additional framework for comparison
"""
from ron_gauss import RONGauss
import pandas as pd
import numpy as np
from data_generation import import_data
from sklearn import preprocessing

target_variable = 'sex'

target_epsilon = 8
target_delta = 1e-5

X_orig = import_data(0,250,"heart.csv")
X_test = import_data(0,50,"heart.csv")


string_orig = "norm_heart.csv"
string_test = "test_heart.csv"

get_axis = pd.read_csv("heart.csv")
axis = get_axis.columns
print("axis is", axis)

X_orig = pd.DataFrame(X_orig,columns=axis)
X_test = pd.DataFrame(X_test,columns=axis)


X_orig.to_csv(string_orig,index=False)
X_test.to_csv(string_test,index=False)

train = pd.read_csv("norm_heart.csv")
test = pd.read_csv("test_heart.csv")
print("Train is")
print(train)


#train.set_axis(axis, axis='columns', inplace=True)
#test.set_axis(axis, axis='columns', inplace=True)

X_train = np.nan_to_num(train.drop([target_variable], axis=1).values)
#print(X_train)
y_train = np.nan_to_num(train[target_variable].values)
X_test = np.nan_to_num(test.drop([target_variable], axis=1).values)
y_test = np.nan_to_num(test[target_variable].values)


input_dim = X_train.shape[1]
z_dim = int(input_dim / 4 + 1) if input_dim % 4 == 0 else int(input_dim / 4)
model = RONGauss(z_dim, target_epsilon, target_delta, True)


X_syn, y_syn, dp_mean_dict = model.generate(X_train, y=y_train)
for label in np.unique(y_test):
    idx = np.where(y_test == label)
    x_class = X_test[idx]
    x_norm = preprocessing.normalize(x_class)
    x_bar = x_norm - dp_mean_dict[label]
    x_bar = preprocessing.normalize(x_bar)
    X_test[idx] = x_bar