# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:03:22 2020

@author: Chance DeSmet
"""
import numpy as np

'''
This function takes in the sensitive features, and moves those columns to the front
of the data so that it may be cropped off for the reidentification gent
'''
def move_sensitive(data,sens):
    i = 0
    newperms = []
    for items in sens:
        data[:,[i, sens[i]]] = data[:,[sens[i], i]]
        newperms.append(i)
        i += 1
    return(data,newperms)
