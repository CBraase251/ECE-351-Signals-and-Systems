# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:36:17 2020

@author: Connor
"""

import numpy as np
import scipy.signal as sig 
import time
import matplotlib.pyplot as plt



def ak (k): 
    ak = 0
    return ak
    
def bk (k):
    
    bk = np.cos(np.pi*k)*(-2/(np.pi*k)) + (2/(np.pi*k))
    return bk


print("ak(0):", ak(0))
print("ak(1):", ak(1))
print("bk(1):", bk(1))
print("bk(2):", bk(2))
print("bk(3):", bk(3))



def Fourier (t,k,w):
    x = np.zeros(t.shape)
    for i in range(1,k+1):
        for j in range(len(t)):
            x[j] +=  ak(i)*np.cos(i*w*t[j]) + bk(i)*np.sin(i*w*t[j])
    x = x+ .5*ak(0)
    return x;

step_s= 10e-4
#-----step size to seconds multifactor = 1000
endtime = 2
t = np.arange(0, endtime + step_s,step_s)
lt =len(t)

w = (2*np.pi)/1
k = 1

x_1 = Fourier(t,k,w)
x_3 = Fourier(t,k*3,w)
x_15 = Fourier(t,k*15,w)
x_50 = Fourier(t,k*50,w)
x_150 = Fourier(t,k*150,w)
x_1500 = Fourier(t,k*1500,w)



plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,x_1)
plt.grid()
plt.xlabel('time')
plt.title('k = 1')
plt.subplot(3,1,2)
plt.plot(t,x_3)
plt.grid()
plt.xlabel('time')
plt.title('k = 3')
plt.subplot(3,1,3)
plt.plot(t,x_15)
plt.grid()
plt.xlabel('time')
plt.title('k = 15')
plt.tight_layout()



plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,x_50)
plt.grid()
plt.xlabel('time')
plt.title('k = 50')
plt.subplot(3,1,2)
plt.plot(t,x_150)
plt.grid()
plt.xlabel('time')
plt.title('k = 150')
plt.subplot(3,1,3)
plt.plot(t,x_1500)
plt.grid()
plt.xlabel('time')
plt.title('k = 1500')
plt.tight_layout()




