# -*- coding: utf-8 -*-
#---------Connor Braase----------------------------------------
#---------ECE 351 Section 1------------------------------------
#---------Labs 3-----------------------------------------
#---------Due: 9/15/2020---------------------------------------




"""
Created on Tue Sep 15 19:25:43 2020

@author: Connor
"""

import numpy as np
import scipy.signal as sig 
import time
import matplotlib.pyplot as plt





#-------Ramp Function---------------------
def ramp(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y
       
        
#-------Step Function-------
def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y



def f1(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
       y = step(t-2) - step(t-9)
    return y


def f2(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
       y = np.exp(-t)*step(t)
    return y

def f3(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
       y = (ramp(t-2)*(step(t-2)-step(t-3)))+(ramp(4-t)*(step(t-3)-step(t-4)))
    return y


def convolve(a,b):
  
    la = len(a)
    lb = len(b)
    # set the lengths of each function to be the sum of the original lengths
    aext= np.append(a, np.zeros((1,lb-1)))
    bext= np.append(b, np.zeros((1,la-1)))
    result = np.zeros(aext.shape)
    # Update ith entry of result[] with the product of a[] and b[] for every instance in time 
    for i in range(len(a)+len(b)-2):
        for j in range(len(a)):
            result[i] += aext[j]*bext[i-j+1]
    return result*step_s
#Must scale results by the stepsize of the time vector 


step_s = .1
#-----step size to seconds multifactor = 100
endtime = 20
t = np.arange(0, endtime + step_s,step_s)
lt =len(t)
t_ext = np.arange(0,2*t[lt-1]+step_s,step_s)

f1 = f1(t)
f2 = f2(t)
f3 = f3(t)


c1 = convolve(f1, f2)
c2 = convolve(f2, f3)
c3 = convolve(f1, f3)

f5= sig.convolve(f1,f2, mode = 'same')

plt.plot(t,f5)
plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,f1)
plt.grid()
plt.xlabel('time')
plt.title('Function 1 ')
plt.subplot(3,1,2)
plt.plot(t,f2)
plt.grid()
plt.xlabel('time')
plt.title('Function 2 ')
plt.subplot(3,1,3)
plt.plot(t,f3)
plt.grid()
plt.xlabel('time')
plt.title('Function 3')
plt.tight_layout()


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t_ext,c1)
plt.grid()
plt.xlabel('time')
plt.title('Convolution of f1 and f2')
plt.subplot(3,1,2)
plt.plot(t_ext,c2)
plt.grid()
plt.xlabel('time')
plt.title('Convolution of f2 and f3')
plt.subplot(3,1,3)
plt.plot(t_ext,c3)
plt.grid()
plt.xlabel('time')
plt.title('Colvolution of f1 and f3')
plt.tight_layout()