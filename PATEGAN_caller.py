# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:54:14 2020

@author: Chance De Smet
"""
#from data_generation import import_data
import numpy as np
import pandas as pd
from pate_gan import PATE_GAN
import collections
from scipy.special import expit
'''
parser_pate_gan = subparsers.add_parser('pate-gan', parents=[privacy_parser])
parser_pate_gan.add_argument('--lap-scale', type=float,
                             default=0.0001, help='Inverse laplace noise scale multiplier. A larger lap_scale will '
                                                  'reduce the noise that is added per iteration of training.')
parser_pate_gan.add_argument('--batch-size', type=int, default=64)
parser_pate_gan.add_argument('--num-teachers', type=int, default=10, help="Number of teacher disciminators in the pate-gan model")
parser_pate_gan.add_argument('--teacher-iters', type=int, default=5, help="Teacher iterations during training per generator iteration")
parser_pate_gan.add_argument('--student-iters', type=int, default=5, help="Student iterations during training per generator iteration")
parser_pate_gan.add_argument('--num-moments', type=int, default=100, help="Number of higher moments to use for epsilon calculation for pate-gan")

privacy_parser.add_argument('--target-epsilon', type=float, default=8, help='Epsilon differential privacy parameter')
privacy_parser.add_argument('--target-delta', type=float, default=1e-5, help='Delta differential privacy parameter')
privacy_parser.add_argument('--save-synthetic', action='store_true', help='Save the synthetic data into csv')

'''


def import_data(start, stop, dat_ref):
    df = pd.read_csv(dat_ref)
    #normalized_df=(df-df.mean())/df.std()
    df = df.apply(pd.to_numeric)
    #print(df.min(), df.max())
    normalized_df = (df-df.min())/(df.max()-df.min())
    #print(normalized_df)
    out = normalized_df.sample(n=stop-start, replace=True).to_numpy()
    return(out)

def train(dat_name,data_loc,var):
    target_epsilon = 8
    target_delta = 1e-5
    save_synthetic = True  
    
    lap_scale = 0.0001
    batch_size = 50
    num_teachers = 5
    teacher_iters = 5
    student_iters = 5
    num_moments = 100
    target_variable = var #the variable to remove
    #train = import_data(0,250,"heart.csv")
    #test = import_data(0,50,"heart.csv")
    #heart_norm.csv
    X_orig = import_data(0,250,data_loc)
    X_test = import_data(0,50,data_loc)
    
    
    string_orig = data_loc
    string_test = data_loc
    
    get_axis = pd.read_csv(data_loc)
    axis = get_axis.columns
    print("axis is", axis, axis.shape)
    
    X_orig = pd.DataFrame(X_orig,columns=axis)
    X_test = pd.DataFrame(X_test,columns=axis)
    
    
    X_orig.to_csv(string_orig,index=False)
    X_test.to_csv(string_test,index=False)
    
    train = pd.read_csv(data_loc)
    test = pd.read_csv(data_loc)
    #train = import_data(0,250,"heart_norm.csv")
    #test = import_data(0,50,"heart_norm.csv")
    
    
    print("Train is")
    print(train.shape)
    
    
    #train.set_axis(axis, axis='columns', inplace=True)
    #test.set_axis(axis, axis='columns', inplace=True)
    
    X_train = np.nan_to_num(train.drop([target_variable], axis=1).values)
    #print(X_train)
    y_train = np.nan_to_num(train[target_variable].values)
    X_test = np.nan_to_num(test.drop([target_variable], axis=1).values)
    y_test = np.nan_to_num(test[target_variable].values)
    
    
    input_dim = X_train.shape[1]
    
    
    #what if z_dim is one???
    z_dim = int(input_dim / 4 + 1) if input_dim % 4 == 0 else int(input_dim / 4)
    #z_dim=1
    
    class_ratios = train[target_variable].sort_values().groupby(train[target_variable]).size().values/train.shape[0]
    
    # Training the generative model
    Hyperparams = collections.namedtuple(
        'Hyperarams',
        'batch_size num_teacher_iters num_student_iters num_moments lap_scale class_ratios lr')
    Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)
    print("Dims are", input_dim, z_dim)
    model = PATE_GAN(input_dim, z_dim, num_teachers, target_epsilon, target_delta, 30000, folder_name=dat_name,  conditional=True)
    model.train(X_train, y_train, Hyperparams(batch_size=batch_size, num_teacher_iters=teacher_iters,num_student_iters=student_iters, num_moments=num_moments,lap_scale=lap_scale, class_ratios=class_ratios, lr=1e-4))

train('heart','./../HydraGAN/Datasets/heart.csv', 'age')
train('cervical','./../HydraGAN/Datasets/cervical_cancer.csv', 'Age')
train('grid', './../HydraGAN/Datasets/electric_grid.csv', )
train('insurance', './../HydraGAN/Datasets/health_insurance.csv', 'secret')
train('iris', './../HydraGAN/Datasets/iris.csv', 'Petal Length')
train('smarthome', './../HydraGAN/Datasets/smarthome_wsu.csv', 'age')