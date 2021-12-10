




import numpy as np
import scipy.signal as sig 
import time
import matplotlib.pyplot as plt



def cosines_method(t,R,P):
    y = np. zeros(t.shape)
    for i in range(len(R)):
        alpha = np.real(P[i])
        omega = np.imag(P[i])
    
        y += np.absolute(R[i])*np.exp(alpha*t)*np.cos(omega*t + np.angle(R[i]))
    return y
    


Num_G = [1,9]
Den_G = sig.convolve([1,-6,-16],[1,4])

Num_A = [1,4]
Den_A = [1,4,3]

Num_B = [1,26, 168]
Den_B = [1]

zA , pA,_ = sig.tf2zpk(Num_A,Den_A)
zB, pB,_ = sig.tf2zpk(Num_B,Den_B)
zG, pG,_ = sig.tf2zpk(Num_G,Den_G)



print('Zeros A ',zA)
print('Poles A', pA)
print('Zeros B ',zB)
print('PolesB', pB)
print('Zeros G ',zG)
print('Poles G', pG)

Num_OLT = sig.convolve(Num_A, Num_G)
Den_OLT = sig.convolve(Den_A, Den_G)

print( 'OLT NUM ', Num_OLT)
print( 'OLT DEN ', Den_OLT)


T, p1 = sig.step((Num_OLT, Den_OLT)) 
zOLT, pOLT,_= sig.tf2zpk(Num_OLT, Den_OLT)

print('Open Loop Transfer Function Zeros:', zOLT)
print('Open Loop Transfer Function Poles:', pOLT)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(T, p1)
plt.grid()
plt.xlabel('time')
plt.title('Open Loop Step Response ')

#%% Part 2

Num_CLT = sig.convolve(Num_A, Num_G)
Den_CLT = sig.convolve((Den_G + sig.convolve(Num_G, Num_B)),Den_A)


print( 'CLT NUM ', Num_CLT)
print( 'CLT DEN ', Den_CLT)



T2, p2 = sig.step((Num_CLT, Den_CLT)) 
zCLT, pCLT,_= sig.tf2zpk(Num_CLT, Den_CLT)

print('Closed Loop Transfer Function Zeros:', zCLT)
print('Closed Loop Transfer Function Poles:', pCLT)


plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(T2, p2)
plt.grid()
plt.xlabel('time')
plt.title('Closed Loop Step Response ')
