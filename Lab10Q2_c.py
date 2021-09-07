#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:39:32 2019

@author: jamiegraham

"""

# C) This is the same code just for T_max = 0.0001.

# This is a code that performs a Monte Carlo simulation in order to compute 
# the angular distribution of photons emitted from the core of a star 
# all the way to the photosphere, as well as the angular dependency of the 
# specific intensity. Input consists of various max optical depths,
# T_max = 10 and T_max = 0.0001, and output consists of histograms of the photon
# distribution for various angles, as well as plots illustrating the angular 
# dependency of the specific intensity for both T_max inputs. 


from random import random, uniform, seed
import numpy as np
from scipy.constants import c, pi, k, g
from scipy import tan, cos, arctan, exp, inf, sin, log
import matplotlib.pyplot as plt
from numpy import copy
import scipy as scipy
from scipy import optimize

# PSEUDOCODE 

# 1. Define photonwalk functions, take from the lab manual, and write your own implementation 
#    for the actual photonwalk.
# 2. Test photonwalk function, perform photonwalk for 10**5 photons, store final mu values
# 3. Plot histogram of the final mu values
# 4. Calculate specific intensity proportional to (N(u)/u_midpoints) by performing a linear regression 
#    on the data. 
# 5. Plot the linear regression with the data, as well as the theoretical expected specific intensity 




# A) 

# The following functions are copied from the lab manual. 


# 1. Define photonwalk functions, take from the lab manual, and write your own implementation 
#    for the actual photonwalk.

def get_tau_step():
    
    """calculate how far a photon travels before it gets scattered .
    Input : tau - optical depth of the atmosphere
    Output: optical depth traveled"""
    
    delta_tau = -np.log(np.random.random())
    return delta_tau

def emit_photon( tau_max ) :
    
    """Emit a photon from the stellar core.
    Input : tau max - max optical depth
    Output :
    tau: optical depth at which the photon is created
    mu: directional cosine of the photon emitted """
    
    tau = tau_max
    delta_tau = get_tau_step()
    mu = np.random.random()
    
    return tau-delta_tau*mu, mu

def scatter_photon(tau):
    
    
    """Scatter a photon .
    Input : tau âĹŠ optical depth of the atmosphere
    Output :
    tau: new optical depth
    """
    
    delta_tau = get_tau_step()
    # sample mu uniformly from -1 to 1
    mu = 2*np.random.random()-1
    tau = tau - delta_tau*mu
    return tau, mu

# This is a function that computes the random walk of the photon as it traverses
# from the core to the photosphere.

def photonwalk(T_max):    
    T, mu = emit_photon(T_max)      # initialize photonwalk, get first T, mu values
    counter = 0                     # number of scatters
    while T>0:                      # iterate until photon reaches T=0
        counter += 1                # increment scatters
        T, mu = scatter_photon(T)   # scatter the photon 
        if T >= T_max:              # if photon scatters into the core, emit new photon
            T, mu = emit_photon(T_max)
            counter = 0             # re-initialize scattering counter
        else:
            continue                # otherwise, keep iterating until T = 0
    return mu, counter              # returning the cos(theta) of the last angle, scattering counter



# 2. Test photonwalk function, perform photonwalk for 10**5 photons, store final mu values
# B)


d = photonwalk(10)                  # testing the photonwalk function 

print(d[0], ': final scattering cos(theta)')
print(d[1], ': scatters for photon')


u_bucket = np.zeros(10**5)       # initializing final mu value array for 10**5 photons
counter_bucket = np.zeros(10**5) # initializing a scattering counter bucket
for i in range(10**5):           # iterating 10**5 photonwalks
    u, counter = photonwalk(0.0001) 
    u_bucket[i] = u              # populating the array with final mu values
    counter_bucket[i] = counter

  
# 3. Plot histogram of the final mu values


hist = np.histogram(u_bucket,bins = 20)
hist = np.histogram(u_bucket,bins=20,weights = u_bucket/hist[0][19])

n = plt.hist(u_bucket,bins=20);
plt.clf()
n_new = n[0]/n[0][19]
bins = n[1]
vals = n_new

plt.figure()
plt.fill_between(bins,np.concatenate(([0],vals)), step="pre")
plt.xlabel('u values (rad)')
plt.ylabel('N(u)/N(1)')
plt.title("Distribution of Photons Emitted for Various u Values")
plt.savefig("distributionT=0001.png")
plt.show()



# 4. Calculate specific intensity proportional to (N(u)/u_midpoints) by performing a linear regression 
#    on the data. Define I_exp, I_theoretical for plotting. 



u_midpoints = []                                  # calculating midpoints of the histogram bins
I = np.zeros(20)
for i in range(1,40,2):
    u_midpoints.append(round(0.025*i,3))

def I_exp():                                      # calculating specific I(u)/I_1 experimentally
    I_1 = hist[0][19]/1                           # computing I_1 for later
    u_midpoints = []                              # calculating midpoints of the histogram bins
    I = np.zeros(20)                              # initializing specific intensity array
    for i in range(1,40,2):                       # same midpoints calculation as above, internal
        u_midpoints.append(round(0.025*i,3))
    for i in range(len(u_midpoints)):             # iterating over u_midpoints, 
        I[i] = hist[0][i]/(u_midpoints[i])        # calculation of experimental specific I 
    return I/I_1

I_exp = I_exp()                                   # storing experiemental intensity array 

def I_theoretical(u_midpoints):                   # using equation 1 in lab manual to compute specific I/I_1 
    I_1 = hist[0][19]/1                           # computing I_1 for later
    I_theor = np.zeros(len(u_midpoints))          # intializing theoretical specific intensity array
    for i in range(len(u_midpoints)):
        I_theor[i] = (0.4 + 0.6*u_midpoints[i])   # iterating over u_midpoints, computing specific intensity values
    return I_theor


I_theor = I_theoretical(u_midpoints)              # storing theoretical specific intensity


def lin(u_midpoints,m,b):                         # general linear regression function 
    I = np.zeros(len(u_midpoints))                # initialize specific intensity
    for i in range(len(u_midpoints)):             # iterating through u_midpoints to calculate I values 
        I[i] = u_midpoints[i]*m + b               # general linear function for fitting 
    return I

ans, cov = scipy.optimize.curve_fit(lin,u_midpoints,I_exp)   # linear regression implementation
fit_c = ans                                                  # coefficients 
print('m = :', fit_c[0])
print('b = :', fit_c[1])

I_lin = lin(u_midpoints,fit_c[0],fit_c[1])        # storing linear regression array



# 5. Plot the linear regression with the data, as well as the theoretical expected specific intensity 


plt.figure()
plt.plot(u_midpoints,I_lin,label = 'Linear Regression of data, coefficients:' + " " + str(round(fit_c[0],2)) + "," + str(round(fit_c[1],2)))
plt.plot(u_midpoints,I_exp, label = 'I_exp data')
#plt.plot(u_midpoints,I_theor, label = 'Theoretical (I/I_1 = 0.4 + 0.6u)')
plt.title("Angular Dependence of Specific Intensity")
plt.xlabel("u (cosine of theta, rad)")
plt.ylabel("I(u)/I_1 (W m^−2 sr^−1 Hz−1) ")
plt.legend()
plt.savefig('T = 0001.png')













