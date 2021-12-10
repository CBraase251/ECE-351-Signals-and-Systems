# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:44:03 2020

@author: Connor
"""

import numpy as np
import scipy.signal as sig 
import scipy.fftpack
import time
import matplotlib.pyplot as plt
import control as con 



def trans_mag (R,L,C, w ):
    x = np.zeros(w.shape)
    x = (w*(1/(R*C)))/(np.sqrt(w**4 + ((w**2)*((1/(R*C))**2 - (2/(L*C)))) + (1/(L*C))**2))
    x = 20*np.log10(x)
    return x

def trans_phase (R,L,C, w ):
    x = np.zeros(w.shape)
    x = np.pi/2 - np.arctan(( w/(R*C))/(-(w**2) + (1/(L*C))))
    for i in range(len(x)):
        if x[i] >= np.pi/2:
            x[i]  = x[i] -(np.pi)
        else :
            x[i] = x[i]
    return x 
        
def part_2(t): 
    x = np.zeros(t.shape)
    for i in range(len(t)):
        x[i] = np.cos(2*np.pi * 100*t[i]) + np.cos(2*np.pi * 3024*t[i]) + np.sin(2*np.pi * 50000)
    return x


R = 3000
L = 27e-3
C = 2.598e-7

step_s = 1e3

fs = np.pi*100000
time_step = 1/(fs)

H_num =[1/(R*C),0]
H_den =[1, 1/(R*C), 1/(L*C)]


w = np.arange(1e3, 1e6+step_s ,step_s)
t = np.arange(0,.01+time_step, time_step)
#%% Task 1
mag = trans_mag(R,L,C,w)
phi = trans_phase(R, L, C, w)


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.semilogx(w,mag)
plt.grid()
plt.xlabel('Radian / Seccond')
plt.title('Bode Plot in dB')
plt.subplot(3,1,2)
plt.semilogx(w,phi)
plt.grid()
plt.xlabel('Frequency (rad/s)')
plt.title('Phase Plot ')
plt.tight_layout()
plt.show()
#%% Task 2

freq2,mag2,phase2 = sig.bode((H_num, H_den),w)


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.semilogx(freq2,mag2)
plt.grid()
plt.xlabel('Radian / Seccond')
plt.title('Task 2 Bode Plot in dB')
plt.subplot(3,1,2)
plt.semilogx(freq2,phase2)
plt.grid()
plt.xlabel('Frequency (rad/s)')
plt.title(' Task 2 Phase Plot ')
plt.tight_layout()
plt.show()
#%% Task 3

sys = con.TransferFunction(H_num,H_den)

_ = con.bode(sys, w, dB = True, Hz = True , deg = True, Plot = True)

#%% Task 4

signal = part_2(t)

Z_num, Z_den = sig.bilinear(H_num,H_den, fs)

filtered = sig.lfilter(Z_num,Z_den, signal)


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,signal)
plt.grid()
plt.xlabel('Time')
plt.title('Signal For Part 2')

plt.subplot(3,1,2)
plt.plot(t,filtered)
plt.grid()
plt.xlabel('Time')
plt.title(' Filtered Signal ')
plt.tight_layout()
