#---------Connor Braase----------------------------------------
#---------ECE 351 Section 1------------------------------------
#---------Labs 0 and 1-----------------------------------------
#---------Due: 9/8/2020---------------------------------------



# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:56:51 2020

@author: Connor
"""

import numpy as np
import scipy.signal as sig
import time 

t =  1 
print(t)
print("t =", t)
print ("t = ", t, "seconds")
print ("t is now= ", t/3, "\n.... and can be rounded using round()", round(t/3, 4))


list1 = [0,1,2,3]
print('list1:', list1)
list2 = [[0],[1],[2],[3]]
print('list2:', list2)
list3 = [[0,1],[2,3]]
print('list3:', list3)
array1 = np.array([0,1,2,3])
print('array1:', array1)
# the tow arrays below are 2x2 matricies 
array2 =  np.array([[0], [1], [2], [3]])
print('array2:', array2)
array3 =  np.array([[0,1], [2,3]])
print('array3:', array3)

#I used the easier imports earlier 

print(np.pi)

print(np.arange(4), '\n',
      np.arange(0,2,0.5), '\n',
      np.linspace(0,1.5,4))

list1 =[1,2,3,4,5]
array1 = np.array(list1)
print('lsit1 :', list1[0], list1[4])
print('array1:', array1[0], array1[4])
array2 = np.array([[1,2,3,4,5], [6,7,8,9,10]])
list2 = list(array2)
print('array2:' , array2[0,2], array2[1,4])
print('list2 :', list2[0],list2[1])

print(array2[:,2], array2[0,:])

print('1x3:',np.zeros(3))
print('2x2:', np.zeros((2,2)))
print('2x3:', np.ones((2,3)))

#%% Section 2: Plots 

import matplotlib.pyplot as plt
# Define some variables 
step = 0.1
x = np.arange(-2,2+step,step)

y1 = x +2
y2 = x**2

#code for plots
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(x,y2)
plt.title('My first Plot :) ') 
plt.ylabel('subplot 1')
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(x,y2)
plt.ylabel('subplot 2')
plt.grid(which='both')

plt.subplot(3,1,3)
plt.plot(x,y1, '--r',label='y1')
plt.plot(x,y2, 'o', label='y2')
plt.axis([-2.5,2.5,-0.5,4.5])
plt.grid(True)
plt.legend(loc='lower right')
plt.xlabel('x')
plt.ylabel('Subplot 3')
plt.show()