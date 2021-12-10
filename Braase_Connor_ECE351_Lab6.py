# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:41:17 2020

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


def step_response(t):
    y = np.zeros(t.shape)
    y = ( .5 + np.exp(-6*t) - .5*(np.exp(-4*t)) )*step(t)
    return y

def Polar(x): 
    mag = np.sqrt(np.real(x)**2 + np.imag(x)**2)
    ang = np.angle(x)
    return mag, ang 

def cosines_method(t,R,P):
    y = np. zeros(t.shape)
    for i in range(len(R)):
        alpha = np.real(P[i])
        omega = np.imag(P[i])
    
        y += np.absolute(R[i])*np.exp(alpha*t)*np.cos(omega*t + np.angle(R[i]))
    return y
    
step_s= 10e-3
#-----step size to seconds multifactor = 1000
endtime = 2
t = np.arange(0, endtime + step_s,step_s)
lt =len(t)
t_ext = np.arange(t[0]*2,2*t[lt-1]+step_s,step_s)


## Part -----------------------------
step_re = step_response(t)


num_s =[1,6,12]
den_s =[1,10,24]

num=[1,6,12]
den =[1,10,24,0]

Residue, Poles, _ = sig.residue(num,den)

xt, xx = sig.step((num_s,den_s), T= t)


print('Part 1 Residue and Poles ')
print(Residue, Poles)


plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,step_re)
plt.grid()
plt.xlabel('time')
plt.title('Hand Done Step Response ')
plt.subplot(2,1,2)
plt.plot(t,xx)
plt.grid()
plt.xlabel('time')
plt.title('sig.step()  Step Response')
plt.tight_layout()

## Part 2



step_s= 10e-3
#-----step size to seconds multifactor = 1000
endtime = 4.5
t = np.arange(0, endtime + step_s,step_s)

num2 = [25250]
den2 = [1,18,218,2036,9085,25250,0]

num2_s = [25250]
den2_s = [1,18,218,2036,9085,25250,]

R,P,K = sig.residue(num2, den2)
P_mag, P_ang = Polar(P)

print('Residue and Poles for the step response of the y(t) in part 2')
print(R,P,K)
#print(P_mag, P_ang)


step2 =cosines_method(t,R,P)
xt, step2_auto = sig.step((num2_s,den2_s),T=t)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(t,step2)
plt.grid()
plt.xlim = (0,4.5)
plt.xlabel('time')
plt.title('User Generated Step Response ')
plt.subplot(2,1,2)
plt.plot(t,step2_auto)
plt.grid()
plt.xlabel('time')
plt.title(' Part 2 sig.step Step Response')
plt.tight_layout()


