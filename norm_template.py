# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:06:07 2021

@author: sl1mc
"""
import pandas as pd
#import seaborn as sn
#import matplotlib.pyplot as plt
import numpy as np


def norm_temp():
    grid = pd.read_csv('heart.csv')
    
    grid -= grid.min()
    
    grid /= grid.max()
    
    grid.to_csv('./heart_norm.csv')
    
def corr_matrix(df_path, sens):  
    df = pd.read_csv(df_path)
    corrMatrix = df.corr()
    cared_corr = corrMatrix.iloc[sens]
    #sn.heatmap(cared_corr, annot=True)
    #plt.show()
    return(cared_corr.to_numpy())

norm_temp()
'''
path = './last_test/out/grid168.csv'
sens=13
out_hydra = corr_matrix(path, sens)[None,:]

path = './testing/ppgan_grid.csv'
out_pp = corr_matrix(path, sens)[None,:]

path = './testing/pate_grid.csv'
out_pate = corr_matrix(path, sens)[None,:]

path = './testing/orig_grid.csv'
out_orig = corr_matrix(path, sens)[None,:]

comb = np.concatenate((out_orig, out_hydra, out_pp, out_pate), axis=0)

#print(out_orig.shape, out_hydra.shape, out_pp.shape, out_pate.shape)


comb_df = pd.DataFrame(comb)
comb_df.index = ['Original', 'Hydra', 'PPGAN', 'PATEGAN']
sn.heatmap(comb_df, annot=False)
plt.show()
'''
