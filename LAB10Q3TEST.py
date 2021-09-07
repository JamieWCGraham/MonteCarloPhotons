#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:01:05 2019

@author: jamiegraham
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:38:39 2018
@author: student
"""
#packages for use
import numpy as np
import numpy.random as ra
from random import random
import matplotlib.pyplot as plt

a = 0#lower bound of integral
b = 1#Higher bound of integral
N = 10000#random total goal

# define the target function
def fx(x):
    return (x**(-1/2))/(np.exp(x)+1)

#define the weight function
def w(x):
    return x**(-1/2)

#define the mean value method
def generate (N):
    total = 0
    for i in range(N):
        x = random()
        total += fx(x)
    return total/N

#prepare the set for containing result
result = np.zeros(100)

#combine the results into one array
for i in range(100):
    result[i]=generate(N)
    
#output the figure of the mean value method
plt.figure(1)  
plt.hist(result,10,range=[0.8,0.88])
plt.title('mean value method')
plt.show()

#define the importance sampling method
def generate_1(N):
    total = 0
    for i in range(N):
        x = random()
        total += 2*fx(x**2)/w(x**2)
    return total/N

#prepare the set for containing result
result_1 = np.zeros(100)

#combine the results into one array
for i in range(100):
    result_1[i] = generate_1(N)

#output the figure of the importance sampling method
plt.figure(2)  
plt.hist(result_1,10,range=[0.8,0.88])
plt.title('importance sampling method')
plt.show()

a = 0#lower bound of integral
b = 10#Higher bound of integral
N = 10000#random total goal

# define the target function
def f(x):
    return np.exp(-2*abs(x-5))

#define the weight function
def wi(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-(x-5)**2/2)

#define the mean value method
def generate_2(N):
    total = 0
    for i in range(N):
        x = ra.random(1)*10
        total += f(x)
    return (b-a)*total/N

#prepare the set for containing result
result = np.zeros(100)

#combine the results into one array
for i in range(100):
    result[i]=generate_2(N)
    
#output the figure of the mean value method
plt.figure(3)  
plt.hist(result,10,range=[0.96,1.04])
plt.title('mean value method')
plt.show()

#define the importance sampling method
def generate_3(N):
    total = 0
    for i in range(N):
        x = ra.normal(5,1)
        total += f(x)/wi(x)
    return total/N

#prepare the set for containing result
result_1 = np.zeros(100)

#combine the results into one array
for i in range(100):
    result_1[i] = generate_3(N)

#output the figure of the importance sampling method
plt.figure(4)  
plt.hist(result_1,10,range=[0.96,1.04])
plt.title('importance sampling method')
plt.show()