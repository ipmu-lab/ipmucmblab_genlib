import numpy as np
import pylab as py
import sys
import os

pi = np.pi

# FILE I/O related
def read_txt2f(filename):
    import fileinput
    arr1 = []
    arr2 = []
    filelines = fileinput.input(filename)
    i = 0
    for line in filelines:
        if i>0: 
            ar = line.split()
            arr1.append(float(ar[0]))
            arr2.append(float(ar[1]))
        i=+1
    return np.array(arr1), np.array(arr2)

def sp_lowpass(f,tau):
    a = 2.*pi*f*tau
    b = 1.+pow(a,2)
    tf = complex(1./b,-a/b)
    return tf

def sp_highpass(f,tau):
    a = 2.*pi*f*tau
    b = 1.+pow(a,2)
    tf = complex(1./b,-a/b)
    return tf

def median_filter(array,n):
    nb = len(array)
    array_out = zeros(nb-n)
    for i in range(0,nb-n):
        idx = range(i,i+n)
        array_out[i] = median(array[idx])
    return array_out


# DEFINE THE FUNCTIONS
def model_exp_spin_down(x,a,b,c):
    model = a*np.exp(-(x-b)*c)
    return model

def model_explin_spin_down_t0(x,a,b,c,d):
    model = a*np.exp(-(x-b)*c)+d
    if ( (c<0) | ((-2.*pi*d*c)<0) | ((a+d)<0) ): return 1.e10*np.ones(len(x))
    return model

def model_explin_spin_down(x,a,b,c):
    model = a*np.exp(-x*b)+c
#    if ( (b<0) | ((-2.*pi*c*b)<0) | ((a+c)<0) ): return 1.e10*np.ones(len(x))
    return model

def model_lin(x,a,b):
    model = a*x+b
    return model

def model_fracspeedvar(x,a,b):
    model = np.sqrt(1.+a/x**2)-1.+b
    return model

