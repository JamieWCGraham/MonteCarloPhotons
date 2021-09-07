#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:54:03 2019

@author: jamiegraham
"""

# This is a code that evalutes an integral using the regular mean value method 
# as well as the importance sampling method. The integral is evaluated using 
# 10,000 sample points for each method. Output consists of histograms of the 
# resulting integral values for both methods using 10 bins from 0.8 to 0.88. 


from random import random, uniform, seed
import numpy as np
from scipy.constants import c, pi, k, g
from scipy import tan, cos, arctan, exp, inf, sin, log
import matplotlib.pyplot as plt
from numpy import copy
import scipy as scipy
from scipy import optimize


# PSEUDOCODE 
# 1. Define integrand, evaluate integral using the regular mean value method, 
#    using 10000 sample points. Repeat 100 times. 
# 3. Define transformation for non-uniform sampling of region.
# 2. Make histogram of the resulting integral vaues using 10 bins from 0.8 to 0.88 
# 4. Evaluate integral using the importance sampling method, using 10000 sample points.
#    Repeat 100 times. 
# 5. Make histogram of the resulting integral vaues using 10 bins from 0.8 to 0.88 



# 1. Define integrand, evaluate integral using the regular mean value method, 
#    using 10000 sample points. Repeat 100 times. 


def f(x): 
    return     x**(-0.5)/(1 + exp(x))

N = 10000            # number of sample points
repetitions = 100    # number of repetitions of integral evaluation 
I = np.zeros(repetitions)    # initialize integral value array for mean value method
I_2= I                       # same, but for importance sampling method
b = 1
a = 0 


def mean_val_method(N,b,a,f):    # implementation of mean value method Monte Carlo for integrals
    
    x = np.zeros(N)              # initializing x array 
    for i in range(0,N):         # iterating over sample points 
        x[i] = random()          # uniformly random sampling on the domain of the integral 
    
    f_avg = 1/N * sum(f(x))      # Monte Carlo mean value method definition
    I = f_avg*(b-a)              
    
    return I 


for i in range(repetitions):     # performing the integral evaluation method 100 times and storing in I array. 
    I[i] = mean_val_method(N,b,a,f)


# 2. Plot histogram of the resulting integral vaues using 10 bins from 0.8 to 0.88 
    
    
plt.hist(I,bins = 10,range = [0.8,0.88])
plt.title('Distribution of Integral Values for 100 iterations, Mean Value Method')
plt.xlabel('Integral values')
plt.ylabel('Count')
plt.savefig('meanval.png')
plt.show()




# 3. Define transformation for non-uniform sampling of region.
    
def x_trans(z): # transformation was derived via integrating p(x)dx = p(z)dz for p(z) = 1 from 0 to 1. 

    return z**2



# 4. Evaluate integral using the importance sampling method, using 10000 sample points.
#    Repeat 100 times. 
    

def w(x):                        # weighting function that removes the singularity in the integrand
    return x**(-1/2)

def imp_sample_method(N,b,a,f):  # definition of importance sampling function 
    
    x = np.zeros(N)              # initializing x array 
    for i in range(0,N):         # iterating over sample points 
        z = random()
        x[i] = x_trans(z)             # non-uniformly random sampling on the domain of the integral 
    
    f_over_w_avg = 1/N * sum(f(x)/w(x))      # Monte Carlo importance sampling definition
    I_2 = f_over_w_avg*2                       # the factor of 2 comes from the mathematical definition, integrating w(x) = x**(-1/2) from 0 to 1 yields the value of 2.             
    
    return I_2
    
      
for i in range(repetitions):     # performing the integral evaluation method 100 times and storing in I array. 
    I_2[i] = imp_sample_method(N,b,a,f)
    
    
    
# 5. Plot histogram of the resulting integral vaues using 10 bins from 0.8 to 0.88 
    
plt.hist(I_2,bins = 10,range = [0.8,0.88])
plt.title('Distribution of Integral Values for 100 iterations, Importance Sampling Method')
plt.xlabel('Integral values')
plt.ylabel('Count')
plt.savefig('impsamp.png')
plt.show()







