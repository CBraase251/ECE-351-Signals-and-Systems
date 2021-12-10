# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:31:23 2020

@author: Connor
"""

import numpy as np
import scipy.signal as sig 
import scipy.fftpack
import time
import matplotlib.pyplot as plt




def my_fft (x, fs): 
    
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    return X_mag, X_phi, freq



def my_fft_clean (x, fs): 
    
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    for j in range(len(X_phi)):
        if X_mag[j] < 1e-10:
            X_phi[j] = 0
        
    return X_mag, X_phi, freq



def task_1 (t):
    x = np.zeros(t.shape)
    for i in range(len(t)):
        x[i] = np.cos(2*np.pi*t[i])
    return x 
def task_2 (t):
    x = np.zeros(t.shape)
    for i in range(len(t)):
        x[i] = 5*np.sin(2*np.pi*t[i])
    return x 
def task_3(t):
    x = np.zeros(t.shape)
    for i in range(len(t)):
        x[i] = (2*np.cos(2*np.pi*2*t[i])) + (np.sin((2*np.pi*6*t[i])+3)**2)
    return x 

#%% Fourier Series Code from last lab 



def ak (k): 
    ak = 0
    return ak
    
def bk (k):
    bk = np.cos(np.pi*k)*(-2/(np.pi*k)) + (2/(np.pi*k))
    return bk

def Fourier (t,k,w):
    x = np.zeros(t.shape)
    for i in range(1,k+1):
        for j in range(len(t)):
            x[j] +=  ak(i)*np.cos(i*w*t[j]) + bk(i)*np.sin(i*w*t[j])
    x = x+ .5*ak(0)
    return x;



#%% time a sampling definintions
fs =1e2
step_s = 1/fs
#-----step size to seconds multifactor = 10000
endtime = 2
t = np.arange(0, endtime ,step_s) # Time vecotr must match sampling freq exactly 
#%% Task 1 Computations
x1 = task_1(t)
x1_mag, x1_phi, x1_freq = my_fft(x1,fs)


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,x1)
plt.grid()
plt.xlabel('time')
plt.title('Task 1 x(t)')

plt.subplot(3,2,3)
plt.stem(x1_freq,x1_mag)
plt.grid()
plt.xlabel('Frequency')
plt.title('Task 1 fft Mag')

plt.subplot(3,2,4)
plt.stem(x1_freq,x1_mag)
plt.grid()
plt.xlabel('Frequency')
plt.xlim(-5,5)
plt.title('Task 1 fft Mag')

plt.subplot(3,2,5)
plt.stem(x1_freq,x1_phi)
plt.grid()
plt.xlabel('Frequency')
plt.title('Task 1 fft Angle')


plt.subplot(3,2,6)
plt.stem(x1_freq,x1_phi)
plt.grid()
plt.xlabel('frequency Hz')
plt.xlim(-5,5)
plt.title('Task 1 fft  Angle')

plt.tight_layout()

#%% Task 2 Computations 
x2 = task_2(t)
x2_mag, x2_phi, x2_freq = my_fft(x2,fs)

plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,x2)
plt.grid()
plt.xlabel('time')
plt.title('Task 2 x(t)')

plt.subplot(3,2,3)
plt.stem(x2_freq,x2_mag)
plt.grid()
plt.xlabel('Frequency')
plt.title('Task 2 fft Mag')

plt.subplot(3,2,4)
plt.stem(x2_freq,x2_mag)
plt.grid()
plt.xlabel('Frequency')
plt.xlim(-5,5)
plt.title('Task 2 fft Mag')

plt.subplot(3,2,5)
plt.stem(x2_freq,x2_phi)
plt.grid()
plt.xlabel('Frequency')
plt.title('Task 2 fft Angle')


plt.subplot(3,2,6)
plt.stem(x2_freq,x2_phi)
plt.grid()
plt.xlabel('Frequency')
plt.xlim(-5,5)
plt.title('Task 2 fft  Angle')

plt.tight_layout()

#%% Task 3 Computations 
x3 = task_3(t)
x3_mag, x3_phi, x3_freq = my_fft(x3,fs)

plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,x3)
plt.grid()
plt.xlabel('time')
plt.title('Task 3 x(t)')

plt.subplot(3,2,3)
plt.stem(x3_freq,x3_mag)
plt.grid()
plt.xlabel('Frequency Hz')
plt.title('Task 3 fft Mag')

plt.subplot(3,2,4)
plt.stem(x3_freq,x3_mag)
plt.grid()
plt.xlabel('Frequency Hz')
plt.xlim(-15,15)
plt.title('Task 3 fft Mag')

plt.subplot(3,2,5)
plt.stem(x3_freq,x3_phi)
plt.grid()
plt.xlabel('Frequency Hz')
plt.title('Task 3 fft Angle')


plt.subplot(3,2,6)
plt.stem(x3_freq,x3_phi)
plt.grid()
plt.xlabel('Frequency Hz')
plt.xlim(-15,15)
plt.title('Task 3 fft  Angle')

plt.tight_layout()



#%% Task 1 Clean FFT


x1 = task_1(t)
x1_mag, x1_phi, x1_freq = my_fft_clean(x1,fs)


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,x1)
plt.grid()
plt.xlabel('time')
plt.title('Task 1 x(t)')

plt.subplot(3,2,3)
plt.stem(x1_freq,x1_mag)
plt.grid()
plt.xlabel('Frequency')
plt.title('Task 1 fft Mag')

plt.subplot(3,2,4)
plt.stem(x1_freq,x1_mag)
plt.grid()
plt.xlabel('Frequency')
plt.xlim(-5,5)
plt.title('Task 1 fft Mag')

plt.subplot(3,2,5)
plt.stem(x1_freq,x1_phi)
plt.grid()
plt.xlabel('Frequency')
plt.title('Task 1 fft Clean Angle')


plt.subplot(3,2,6)
plt.stem(x1_freq,x1_phi)
plt.grid()
plt.xlabel('frequency Hz')
plt.xlim(-5,5)
plt.title('Task 1 fft Clean  Angle')

plt.tight_layout()

#%% Task 2 Clean Computations 
x2 = task_2(t)
x2_mag, x2_phi, x2_freq = my_fft_clean(x2,fs)

plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,x2)
plt.grid()
plt.xlabel('time')
plt.title('Task 2 x(t)')

plt.subplot(3,2,3)
plt.stem(x2_freq,x2_mag)
plt.grid()
plt.xlabel('Frequency')
plt.title('Task 2 fft Mag')

plt.subplot(3,2,4)
plt.stem(x2_freq,x2_mag)
plt.grid()
plt.xlabel('Frequency')
plt.xlim(-5,5)
plt.title('Task 2 fft Mag')

plt.subplot(3,2,5)
plt.stem(x2_freq,x2_phi)
plt.grid()
plt.xlabel('Frequency')
plt.title('Task 2 fft Clean Angle')


plt.subplot(3,2,6)
plt.stem(x2_freq,x2_phi)
plt.grid()
plt.xlabel('Frequency')
plt.xlim(-5,5)
plt.title('Task 2 fft Clean Angle')

plt.tight_layout()

#%% Task 3 Clean  Computations 
x3 = task_3(t)
x3_mag, x3_phi, x3_freq = my_fft_clean(x3,fs)

plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,x3)
plt.grid()
plt.xlabel('time')
plt.title('Task 3 x(t)')

plt.subplot(3,2,3)
plt.stem(x3_freq,x3_mag)
plt.grid()
plt.xlabel('Frequency Hz')
plt.title('Task 3 fft Mag')

plt.subplot(3,2,4)
plt.stem(x3_freq,x3_mag)
plt.grid()
plt.xlabel('Frequency Hz')
plt.xlim(-15,15)
plt.title('Task 3 fft Mag')

plt.subplot(3,2,5)
plt.stem(x3_freq,x3_phi)
plt.grid()
plt.xlabel('Frequency Hz')
plt.title('Task 3 fft Clean Angle')


plt.subplot(3,2,6)
plt.stem(x3_freq,x3_phi)
plt.grid()
plt.xlabel('Frequency Hz')
plt.xlim(-15,15)
plt.title('Task 3 fft Clean Angle')

plt.tight_layout()

##My Fourier Series Transform 




k = 15 
T = 8
w = (2*np.pi)/T
t = np.arange(0, 16 ,step_s)

x4 = Fourier(t,k,w)

x4_mag, x4_phi, x4_freq = my_fft_clean(x4,fs)


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,x4)
plt.grid()
plt.xlabel('time')
plt.title('Task 5 x(t)')

plt.subplot(3,2,3)
plt.stem(x4_freq,x4_mag)
plt.grid()
plt.xlabel('Frequency Hz')
plt.title('Task 5 fft Mag')

plt.subplot(3,2,4)
plt.stem(x4_freq,x4_mag)
plt.grid()
plt.xlabel('Frequency Hz')
plt.xlim(-3,3)
plt.title('Task 5 fft Mag')

plt.subplot(3,2,5)
plt.stem(x4_freq,x4_phi)
plt.grid()
plt.xlabel('Frequency Hz')
plt.title('Task 5 fft Clean Angle')


plt.subplot(3,2,6)
plt.stem(x4_freq,x4_phi)
plt.grid()
plt.xlabel('Frequency Hz')
plt.xlim(-3,3)
plt.title('Task 5  fft Clean Angle')

plt.tight_layout()

