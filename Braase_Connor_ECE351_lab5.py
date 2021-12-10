# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:41:13 2020

@author: Connor
"""

import numpy as np
import scipy.signal as sig 
import time
import matplotlib.pyplot as plt




def step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def sin_meth(t,R,L,C):
    y = np.zeros(t.shape)
    
    
    
    alpha = -1/(2*R*C)
    omega = (1/2)*np.sqrt((1/(R*C))**2 - 4*(1/(np.sqrt(L*C)))**2 + 0*1j)
    p = alpha + omega
    
    g = (1/(R*C))*(p)
    g_mag = np.abs(g)
    g_rad = np.angle(g)
    
    y =(g_mag/np.abs(omega)*np.exp(alpha*t)*np.sin(np.abs(omega)*t+g_rad)*step(t))*step_s
    return y 
       


step_s= 10e-6
#-----step size to seconds multifactor = 100
endtime = 1.2*.001
t = np.arange(0, endtime + step_s,step_s)
lt =len(t)
t_ext = np.arange(t[0]*2,2*t[lt-1]+step_s,step_s)



R= 1000
C = 100E-9
L = 27*1E-3

num = [0,(1/R*C),0]
den = [1, (1/(R*C)),(1/(L*C))]

y = sin_meth(t,R,L,C)     
xt, xx = sig.impulse((num,den), T= t)


plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.xlabel('time')
plt.title('Hand Done Impulse')
plt.subplot(2,1,2)
plt.plot(t,xx)
plt.grid()
plt.xlabel('time')
plt.title('sig.impulse ')
plt.tight_layout()


#--------------------------part2-----------------------

zt,zz = sig.step((num,den), T= t)

plt.figure(figsize = (10,7))
plt.subplot(1,1,1)
plt.plot(t,zz)
plt.grid()
plt.xlabel('time')
plt.title('Sig.Step')

   
