# -*- coding: utf-8 -*-
"""
    Braase, Connor 
    Student ID: V00550908
    Section #: 51
    Major: Electrical Engineering 
"""


import numpy as np
import scipy.signal as sig 
import time
import matplotlib.pyplot as plt




#%% Question # 1 
"""
There are a few key difference between numpy.array(), numpy.arrange(), and numpy.linspace()
    --numpy.array() takes lists of numbers, comma separated and delinated by square brackets to create 
    an array where each set of numbers in square brackets acts like a row in a matrix it takes explicit sets of values as arguments
    -- numpy.arange() takes a starting and ending vlaue and a step size and creates a 1 dimentional array of the values with the upper limit not contained 
    --numpy.linspace() takes a starting value and an ending value  and a number of entries and outputs an array that corresponds. It also has kwarg arguments that
    allow more careful handling of end points. Generally if a needed step size between array entries is not an integer, this is a better option than numpy.arrange()
    

"""

def array_loop (size): # User Defined function to create an array 
    
    x = np.zeros(101)
    array1 = np.array(x)
    for i in range(len(x)):
            array1[i] = array1[i] + 13*i
    return array1 


size = 13  # size of step between arrat values 

array1 = array_loop(size)
array2 = np.arange(0,101*size,size) 
array3 = np.linspace(0,1300,101, endpoint= True)

print('Using np.array()', array1)
print('Using np.arrange()', array2)
print('Using np.linspace()', array3)

#%% Question# 2 


step_s= 10e-3 
endtime = 1.3
t = np.arange(0, endtime + step_s,step_s)  # Setting up a time vector 

x1 = .5*(np.cos(2*np.pi*2*t))
x2 = 3*(np.sin(2*np.pi*t) + np.cos(2*np.pi*t))
x3 = 4*np.exp(-1*t)*np.sin(2*np.pi*6*t)
x4 = np.exp(-2*t) + 3*np.exp(-6*t) + 6*np.exp(-12*t)


plt.figure(figsize = (10,7)) # Outputs a single plot with 4 sub plots 
plt.subplot(3,2,1)
plt.plot(t,x1)
plt.grid()
plt.xlabel('time')
plt.title('x1(t)')

plt.subplot(3,2,2)
plt.plot(t,x2)
plt.grid()
plt.xlabel('time')
plt.title('x2(t)')

plt.subplot(3,2,3)
plt.plot(t,x3)
plt.grid()
plt.xlabel('time')
plt.title('x3(t)')

plt.subplot(3,2,4)
plt.plot(t,x4)
plt.grid()
plt.xlabel('time')
plt.title('x4(t)')

plt.tight_layout()


#%%% Question# 3 

endtime2 = 2
t2 = np.arange(0, endtime2 + step_s,step_s)

H_num = [1,0,0,5,17,0,-39,-20]
H_den = [1,2,0,13,0,2,0,16,200]
H_den_step_response = [1,2,0,13,0,2,0,16,200,0] # s domain step response multiplies by 1/s adds 1 term 

yt,y1 = sig.impulse((H_num,H_den), T = t2)
zt,z1 = sig.step((H_num,H_den), T = t2)


RS,PS,_ = sig.residue(H_num, H_den_step_response)
RI, PI,_ = sig.residue(H_num, H_den)


plt.figure(figsize = (10,7))
plt.plot(t2,y1, color = "k", label = 'Impulse Response')
plt.plot(t2,z1, color = "m", label = 'Step Response')
plt.legend()
plt.grid()
plt.xlabel('time')
plt.title('Impulse and Step Responses')

print("Impulse Resposne Residue and Poles")
print('Residue', RI, 'Poles', PI)

print("Step Resposne Residue and Poles")
print('Residue', RS, 'Poles', PS)


"""
    --By Looking at the Poles for Both the Impulse and Step responses, 
    neihter is stable.   Both the step and impulse responses have poles
    that are not in the left half plane of an imaginary, real coordinate plane
    --Another way to tell that they are not stable is to increase the time over which
    the responses are plotted. When t2 is increased to 10 seconds, both plots show that they tend 
    towards negative infinity. 

"""
#%% Question 4 
def convolve(a,b): #Convolution Function Written for lab 4
  
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
    return result

lt =len(t)
t_ext = np.arange(t[0]*2,2*t[lt-1]+step_s,step_s)
ltext = len(t_ext)



f1 = convolve(x1,x3) # f1 convolved using user generated function
f1_check = sig.convolve(x1,x3) # a double check on f1 using a built in convolution
f2 = sig.convolve(f1, x4) 
f3 = sig.convolve(f2,x2)

t_f2 = np.arange(0, (len(f2)*step_s), step_s) # defining time vector of proper size
t_f3 = np.arange(0, (len(f3)*step_s), step_s) # defining time vector of proper size


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t_ext,f1, color = "m", label = "f1")
plt.grid()
plt.legend()
plt.xlabel('time')
plt.title('Convolution of x1 and x3')

plt.subplot(3,1,2)
plt.plot(t_f2,f2, color = "b", label = "f2")
plt.grid()
plt.legend()
plt.xlabel('time')
plt.title('Convolution of f1 and x4')

plt.subplot(3,1,3)
plt.plot(t_f3,f3, color = "g", label = "f3")
plt.grid()
plt.legend()
plt.xlabel('time')
plt.title('Convolution of f2 and x2')

plt.tight_layout()
