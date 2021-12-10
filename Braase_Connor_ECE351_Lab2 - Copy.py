#---------Connor Braase----------------------------------------
#---------ECE 351 Section 1------------------------------------
#---------Labs 2-----------------------------------------
#---------Due: 9/15/2020---------------------------------------




# -*- coding: utf-8 -*
"""
Created on Tue Sep  8 19:20:15 2020

@author: Connor
"""


import numpy as np
import scipy.signal as sig 
import time
import matplotlib.pyplot as plt


#%%----------------Part 1--------------------------
plt.rcParams.update({'font.size': 14}) #---- not currently working
step_s = .01
#-----step size to seconds multifactor = 100
endtime = 10
t = np.arange(-5, endtime + step_s,step_s)

print('Length of time Vector: ', len(t))

def funct1(t):
    y = np.zeros(t.shape) # set y to an array of zeros 
    
    for i in range(len(t)):
        if i < 0:
            y[i] = 0
        else: 
            y[i] = np.cos(t[i])
    return y
y = funct1(t)

plt.figure(figsize = (10,7))
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t) = Cos(t)')
plt.title('Lab 2 Part 1 Graph')

#%%---------------Part 2---------------------

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
r = ramp(t)
u = step(t)
  
#plt.plot(t,r,t,u)

#--------Function From Part 2----------------

def pt2(t):
      y = np.zeros(t.shape)
      for i in range(len(t)):
          y= ramp(t) - step(t-1) - ramp(t-2) - ramp(t-3)
      return y
  
y = pt2(t)
plt.figure(figsize = (10,7))
plt.plot(t,y)
plt.xlim([-5,10])
plt.grid()
plt.ylabel('Derived Function for Part 2')
plt.title('Part 2 ')




#%%------------------Part 3--------------------------


# Task 1
