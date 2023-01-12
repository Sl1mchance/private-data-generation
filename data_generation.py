# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:33:49 2019

@author: Chance DeSmet

I am starting everything out at zero, but this is EXTREMELY subject to change
"""
import numpy as np
import csv
#from scipy.stats import wasserstein_distance
from network_support import move_sensitive
import pandas as pd

def init_data(batchsize):
    X = np.random.normal(loc = 0, scale = .5, size = (batchsize))
    #X = np.zeros((batchsize))
    n_samples = X.shape[0]
    y = np.zeros((n_samples, 2))
    y[:, 1] = 1
    return(X,y)

def normalize(v):
        norm = max(v)
        #print("norm is", norm)
        if norm == 0: 
           return v
        return v / norm
    
def import_data(start,stop,csv_input):
        tolerance = 0
        with open(csv_input, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            i=0
            vectors = []
            for row in spamreader:
                temp = []
                if(i>0):
                    for item in row:
                        temp.append(float(item))
                    #print(temp)
                #temp = np.array(temp)
                #temp = temp.astype('float')
                #print(temp)
                vectors.append(temp)
                i += 1
        #vectors = np.array(vectors)
        vectors = np.array(vectors)
        vectors = vectors[1:len(vectors)]
        '''
        if(self.adver == 1):
            noise = np.random.normal(loc=0.0, scale=(tolerance/2.0),size=vectors.shape)
            vectors = np.add(vectors,noise)
        '''
        vec = np.zeros(shape=(len(vectors),len(vectors[0])))
        features = len(vectors[0])
        
        i = 0
        for items in vectors:
            vec[i] = vectors[i]
            i += 1
        
        #print(vectors)
        #print("Vec is:")
        #print(vec)
        i = 0
        '''
        Normalizing again
        '''
        for items in vec[1,:]:  
            vec[:,i] = normalize(vec[:,i])
            i += 1
        if(len(vec)<stop):
            stop = len(vec) - 1
        
        '''
        Commented out the initial privacy preservation aspect    
        noise = np.random.normal(loc=0.0, scale=(tolerance/2.0),size=vec.shape)
        vec = np.add(vec,noise)
        '''
        
        
        #print("Vec is:")
        #print(vec)
        np.random.shuffle (vec)
        return(vec[start:stop:1])


'''
This function take in data, a reid discriminator,
and some sensitive permutations and uses them to
find which ones are identified correctly, (using 
built in rounding), and then marks them as if they
were binary reidentifiable or not using these guesses

Assuming these are all generated from a specific permuation, 
we are then seeing if the reid got it right, and then 
marking a sample as reidentifiable if it did
'''
def make_indices(num,sens):
    index = []
    i = 0
    for items in range(num):
        if(i not in sens):
            index.append(int(i))
        i += 1
    return(index)
        



def filter_sensitive(data, sensitive):
    i = 0
    newdata = data.copy()
    #print("before shape is", newdata.shape)
    for items in sensitive:
        newdata = np.delete(newdata,sensitive[i],1)
        i += 1
    #print("after shape is", newdata.shape)
    return(newdata)

  
def get_matching_perm(data,sensdict,perm_length,sens):
    i = 0
    flag = 0
    while i < perm_length:
        q = sensdict.get(i)
        j = 0
        flag = 1
        for things in q:
            #print(q[j], data[[sens[j]]], j,q,len(sens),sens)
            if(q[j] != data[[sens[j]]]):
                flag = 0
            j += 1
        if(flag == 1):
            return(i)
        i += 1
    print("ERROR ERROR NO MATCHING CANDIDATE")
    return(-1)

def classify(data,sens,perms,sensdict):
    i = 0
    sens_list = []
    guess_list = np.zeros(perms)
    guess = []
    for rows in data:
        j = 0
        row = data[i]
        for items in sens:
            sens_list.append(row[sens[j]])
            j += 1
        perm = get_matching_perm(row,sensdict,perms,sens)
        #print("perm is", perm)
        sens_row = np.zeros(perms)
        sens_row[perm] = 1
        guess.append(sens_row)
        i += 1
    #print(guess)
    guess = np.array(guess)
    #print(guess.shape)
    return(guess)
        
            
        

def validate_security(reid,data,sens,mydict,perm_amount):
    '''
    Just like determine security, but we will be using baseline
    data for this, so we won't be given a value that the data
    is generated from
    
    However, they are always reidentifiable, andwe can split them to
    have the correct output
    '''
    y = []
    guess = []
    y_comp = np.zeros(2)
    y_comp[1] == 1
    
    i = 0
    for items in data:
        y.append(y_comp)
        #easy append that this data is reidentifiable and we don't
        #want that
        row = data[i]
        i += 1
    y = np.array(y)
    guess_comp = classify(data,sens,perm_amount,mydict)
    guess = np.array(guess_comp)
    return(y,guess)

def determine_security(reid, data, sens,vals,perm):
    #perm is the position we expext to find the value in
      
    #print("predicted shapes are", y.shape, guess.shape)
    newy = []
    #print("input values are", data,sens,vals)
    stripped_list,filt_list = filter_data(data, sens,vals)
    '''
    filter data takes in the data and returns a 
    1-D array that returns whether the row matched
    the sensitive values
    '''
    #ndata = filter_sensitive(data, sens)
    both = reid.predict(data)
    guess = both
    y = np.zeros((guess.shape[0],2))
    '''
    if perm = -1 then we have been given original data
    then, we need to make guess as the actual sensitive
    catagorization of the data sample
    '''
    z = np.zeros(guess.shape[1])
    #print("Z SHAPE IS", z.shape)
    z[perm] = 1
    i = 0 
    newguess = []
    #print("filt list shape is", filt_list.shape)
    '''
    If the guess is correct, then we are happy about that portion
    but the data is not secure enough.  If the guess is incorrect
    then we are not worried about the security of the data, but must
    fix reliability
    
    resetting the y value, but I beleive we want the data that we want
    to get to be set as 1 == 1, and so, if the guess is the same as the
    value, that is not what we want
    '''
    for items in filt_list:
        if(filt_list[i] == round(guess[i][perm])):
            #print("True; Filt list is", filt_list[i])
            #print("Guess is", guess[i][perm], round(guess[i][perm]))
            y[i][0] = 1
            y[i][1] = 0
        else:
            #print("False; Filt list is", filt_list[i])
            #print("Guess is", guess[i][perm], round(guess[i][perm]))
            y[i][0] = 0
            y[i][1] = 1    
        #print("appending", y[i],filt_list[i],guess[i][perm])
        newy.append((y[i],z))
        newguess.append(z)
        i += 1
        #print("filt list i is", i)
    newy = np.array(newy)
    newguess = np.array(newguess)
    #print("returned shape is", newy)
    return(y,newguess)
        
    '''
    just use filter data to see if the values given 
    are withing the given generators domain
    '''

  
'''
filter data takes in a set of data, the sensitive attribute(s) we want
and the positions of these attributs, and otputs the members of this data 
that possess them
'''    
def filter_data(data,sensitive,vals):
    i = 0    
    flags = []
    empty_list = []
    for rows in data:
        row = data[i]
        j = 0
        flag = 1
        #rounding to integer precision
        for items in sensitive:
            #print(round(row[sensitive[j]]), vals[j],flag)
            if(round(row[sensitive[j]]) != vals[j]):
                flag = 0          
            j+=1
        if(flag == 1):
            empty_list.append(row)
        flags.append(flag)
        i += 1
        #print("i in loop is:", i)
    empty_list = np.array(empty_list)
    '''
    returning the list of data we want as well as the 
    scores for the data as a whole
    '''
    flags = np.array(flags)
    #print(flags)
    return(empty_list,flags)
        
    
def find_perc_dif(data,sensitive,i):
    '''
    finding the difference between generated and original
    
    easy to get difference between real and generated, now just need a way to 
    express the difference in reidentifiability between generated and original
    
    maybe have each gen spit some out, reidentify them all, and take a percent?
    '''
    init = data.shape[0]
    data2,no = filter_data(data,sensitive,i)
    after = data2.shape[0]
    per = (after - init)/(init)
    return(per)
    
def compare_dist(X,file,truesensitive):
    i = 0
    real_dist = import_data(0,X.shape[0],file)
    real_dist,unused = move_sensitive(real_dist, truesensitive)
    priv_dist = X
    #print("shape is", X.shape)
    priv = []
    real = []
    while i < X.shape[0]:
        gen = X[:][i]
        p = priv_dist[:][i]
        r = real_dist[:][i]
        d_p = wasserstein_distance(gen, p)
        d_r = wasserstein_distance(gen, r)
        real.append(d_r)
        priv.append(d_p)
        i += 1
    priv = np.array(priv)
    real = np.array(real)
    m_priv = np.mean(priv)
    m_real = np.mean(real)
    return(m_priv,m_real,(priv-real))
    
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
'''
k = import_data(0,300,"Groups.csv")
j = pd.DataFrame(k)
j.to_csv("norm_Groups.csv")
'''