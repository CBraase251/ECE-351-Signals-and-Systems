# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:20:53 2020

@author: Connor
"""

#---------Connor Braase----------------------------------------
#---------ECE 351 Section 1------------------------------------
#---------Labs 4-----------------------------------------
#---------Due: 9/29/2020---------------------------------------





import numpy as np
import scipy.signal as sig 
import time
import matplotlib.pyplot as plt



# time vector Definitons 
step_s = .1
#-----step size to seconds multifactor = 100
endtime = 10
t = np.arange(-10, endtime + step_s,step_s)
lt =len(t)
t_ext = np.arange(t[0]*2,2*t[lt-1]+step_s,step_s)
f_0 = .25
w_0 = 2*np.pi*f_0

def ramp(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y

def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
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


def h1(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
       y = np.exp(2*t)*step(1-t)
    return y

def h2(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
       y = step(t-2)-step(t-6)
    return y


def h3(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
       y = np.cos(w_0*t)*step(t)
    return y



def h1_hand_convolve(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
       y = (.5)*(np.exp(2*t))*(step(1-t)) + np.exp(2)*step(t-1)
    return y

def h2_hand_convolve(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
       y = ramp(t-2) - ramp(t-6)
    return y


def h3_hand_convolve(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
       y = 1/(w_0)*np.sin(w_0*t)*step(t)
    return y


h1 = h1(t)
h2 = h2(t)
h3 = h3(t)
u = step(t)

r1 = convolve(h1,u)
r2 = convolve(h2,u)
r3 = convolve(h3,u)

hc1 = h1_hand_convolve(t_ext)
hc3 = h3_hand_convolve(t_ext)
hc2 = h2_hand_convolve(t_ext)



plt.plot(t_ext,hc2)

plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,h1)
plt.grid()
plt.xlabel('time')
plt.title('Function 1')
plt.subplot(3,1,2)
plt.plot(t,h2)
plt.grid()
plt.xlabel('time')
plt.title('Function 2 ')
plt.subplot(3,1,3)
plt.plot(t,h3)
plt.grid()
plt.xlabel('time')
plt.title('Function 3')
plt.tight_layout()


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t_ext,r1)
plt.grid()
plt.xlabel('time')
plt.title('Step response of h1')
plt.subplot(3,1,2)
plt.plot(t_ext,r2)
plt.grid()
plt.xlabel('time')
plt.title('Step response of h2')
plt.subplot(3,1,3)
plt.plot(t_ext,r3)
plt.grid()
plt.xlabel('time')
plt.title('Step response of h3')
plt.tight_layout()


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t_ext,hc1)
plt.grid()
plt.xlabel('time')
plt.title('Hand Done Step response of h1')
plt.subplot(3,1,2)
plt.plot(t_ext,hc2)
plt.grid()
plt.xlabel('time')
plt.title('Hand Done Step response of h2')
plt.subplot(3,1,3)
plt.plot(t_ext,hc3)
plt.grid()
plt.xlabel('time')
plt.title('Hand Done Step response of h3')
plt.tight_layout()

