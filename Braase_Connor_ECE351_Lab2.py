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
plt.rcParams.update({'font.size': 14}) 
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
plt.xlabel('time')
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



plt.figure(figsize = (10,7))
plt.plot(t,u,'--', label='u(t)')
plt.plot(t,r, label='r(t)')
plt.xlabel('t')
plt.ylabel('ramp and step functions')
plt.xlim([-5,10])
plt.ylim([-2,10])
plt.grid()
plt.title('Ramp and Step Examples')
plt.legend()


#--------Function From Part 2----------------

def pt2(t):
      y = np.zeros(t.shape)
      for i in range(len(t)):
          y= ramp(t) + 5*step(t-3) - ramp(t-3) - 2*step(t-6) -2*ramp(t-6)
      return y
  
y = pt2(t)
plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t,y)
plt.xlim([-5,10])
plt.grid()
plt.ylabel('Derived Function for Part 2')
plt.xlabel('time')
plt.title('Part 2 ')




#%%------------------Part 3--------------------------


# Task 1
t = np.arange(-10, 10 + step_s,step_s)
y=pt2(-t)


plt.figure(figsize = (10,7))
plt.plot(t,y)
plt.xlim([-10,5])
plt.grid()
plt.ylabel('')
plt.xlabel('time')
plt.title(' Time Reversal')

# Task 2
t = np.arange(-5, 14 + step_s,step_s)
y=pt2(t-4)


plt.figure(figsize = (10,7))
plt.plot(t,y)
plt.grid()
plt.xlim([-5,15])
plt.ylabel('')
plt.xlabel('time')
plt.title(' Time Shift t - 4')

t = np.arange(-15,10 + step_s,step_s)
y=pt2((-t )- 4)


plt.figure(figsize = (10,7))
plt.plot(t,y)
plt.xlim([-15,5])
plt.grid()
plt.xlabel('time')
plt.ylabel('')
plt.title(' Time Shift: -t -4')


# Task 3 
t = np.arange(-5,10 + step_s,step_s)
y=pt2(t/2)

plt.figure(figsize = (10,7))
plt.plot(t,y)
plt.xlim([-5,10])
plt.grid()
plt.xlabel('time')
plt.ylabel('')
plt.title(' Time Shift: t/2')


t = np.arange(-5,10 + step_s,step_s)
y=pt2(t*2)

plt.figure(figsize = (10,7))
plt.plot(t,y)
plt.xlim([-5,10])
plt.grid()
plt.ylabel('')
plt.xlabel('time')
plt.title(' Time Shift: t*2')






# Task 4
# DRAW .IO 

# Task 5
t = np.arange(-5,10 + step_s,.25)
y = pt2(t)
dt = np.diff(t)
dy = np.diff(y, axis=0)/dt
print(len(dy))

plt.figure(figsize = (10,7))
plt.plot(t,y,'--', label='y(t)')
plt.plot(t[range(len(dy))],dy,label ='dy(t)/dt')
plt.xlabel('t')
plt.ylabel('y(t), dy(t)/dt')
plt.xlim([-5,10])
plt.ylim([-2,10])
plt.grid()
plt.title('Derivative with Respect to Time')
plt.legend()
