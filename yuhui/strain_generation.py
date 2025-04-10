#!/usr/bin/env python

# Script to generate random strain histories for input to RVE simulations

import numpy as np
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct, ConstantKernel,
                                              Exponentiation)
import warnings #for deprecated decorator
import functools #for deprecated decorator
import inspect #for deprecated decorator

from time import time #for Timer
from datetime import datetime #for Timer
import os, glob #for clearVtkOutputDirectory

from matplotlib import pyplot as plt

# TODO: The docstring describes arguments that are not in the function signature
# Figure out what is actually happening and update it
def GPSample(control_points, step_points, lower, upper, kernelID = 0, numSamples = 1):
    '''Gaussian Process supperle generator
     
    Args:
        tlist (array of dim 1): time points at which you want the values
        tchar (float): characteristic time of the GP
        upper (float): the upperlitude of the proces. 
        kernelID (int): specify the kernel to use in the GP (range from 0 to 8)
        numSamples (int): specifies the number of sequences in the output
        cont_level (int): continuity level at origin. If it is 0 (for C0), the 
            value of the GP is close to zero at the origin. If cont_level is 1 
            (for C1), the value and the derivative of the GP are close to zero
            at the origin. The values can't be guaranteed to be exactly zero for
            numerical reasons.
    Returns:
        N by numSupperles array, where N is the length of tlist, and each column 
            represents the values of a gaussian process.
    
    '''
    # TODO: Make characteristic time variable?
    # TODO: Pass kwargs for different kernels insted of hardcoding
    ndtchar = 1
    match kernelID:
        case 0: # Radial basis function kernel
            ker = upper**2 * RBF(length_scale=ndtchar, length_scale_bounds=(1e-1, 10.0)) # TODO: why do we multiply by upper ?!?!?!
        case 1: # Rational Quadratic kernel
            ker = upper**2 * RationalQuadratic(length_scale=ndtchar, alpha=0.1)
        case 2: # Exp-Sine-Squared (aka Periodic kernel)
            ker = upper**2 * ExpSineSquared(length_scale=ndtchar, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0))
        case 4: # TODO: some kind of DIY kernel
            ker = ConstantKernel(upper**2, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2)
        case 5: # Matern kernel
            ker = upper**2 * Matern(length_scale=ndtchar, length_scale_bounds=(1e-1, 10.0), nu=1.5)
        case 6: # TODO: some kind of DIY kernel
            ker = ConstantKernel(constant_value=upper**2,constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
        case 7: # Squared RBF Kernel
            ker = Exponentiation( upper**2 * RBF(length_scale=1, length_scale_bounds=(1e-10, 1e10)), exponent=2)
        case 8: # Squared Rational Quadratic kernel
            ker = Exponentiation( upper**2 * RationalQuadratic(length_scale=1, alpha=0.1), exponent=2)

    gp = GaussianProcessRegressor(kernel=ker, alpha=1e-10,optimizer=None,n_restarts_optimizer=10)
    control_values = np.random.uniform(lower, upper, len(control_points))
    control_values[0] = 0.0 # Enforce zero initial strain

    gp.fit(control_points.reshape(-1, 1), control_values.reshape(-1, 1))
    print(gp.sample_y(step_points.reshape(-1, 1), numSamples).shape)
    seqs = gp.sample_y(step_points.reshape(-1, 1), numSamples).reshape(-1, 1)
    
    plt.plot(step_points, seqs, label='pred', marker='s')
    plt.plot(control_points, control_values, label='control', marker="*")
#     plt.legend()
    return gp.sample_y(step_points.reshape(-1, 1), numSamples).reshape(-1, 1)


###corelation length?

# In[5]:

tmax = 20 # total time
N_timesteps = 300 # number of timesteps
N_cp = 20 # number of control points

d_cp = tmax / N_cp
d_step = tmax / N_timesteps

control_points = np.arange(0, tmax, d_cp)
step_points = np.arange(0, tmax, d_step) #times at which we are making getting the values

# Strain bounds
lower_bound = 0.0 # Should we allow negative values ?
upper_bound = 2e-2

# theta . # TODO: what is this ?
amp_theta = np.pi / 12

e_max = 2 * upper_bound


# In[ ]:


import numpy
for aa in range(100):

    kernelID = 6 #kernel type. It's specified in the function definition

    num_seqs = 1
    numPerSeq = 1
    e1_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq])
    e2_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq]) 
    e3_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq]) 
    theta_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq])

    for i in range(num_seqs):
        e1_seqs[0 : N_timesteps, i * numPerSeq:(i+1) * numPerSeq] = GPSample(control_points, step_points, lower_bound, upper_bound, kernelID, numPerSeq)
        e2_seqs[0 : N_timesteps, i * numPerSeq:(i+1) * numPerSeq] = GPSample(control_points, step_points, lower_bound, upper_bound, kernelID, numPerSeq)
        e3_seqs[0 : N_timesteps, i * numPerSeq:(i+1) * numPerSeq] = GPSample(control_points, step_points, lower_bound, upper_bound, kernelID, numPerSeq)
    
    
    numpy.savetxt("e1_seqs_"+str(aa)+".csv", e1_seqs, delimiter = ",")
    numpy.savetxt("e2_seqs_"+str(aa)+".csv", e2_seqs, delimiter = ",")
    numpy.savetxt("e3_seqs_"+str(aa)+".csv", e3_seqs, delimiter = ",")


# In[ ]:


kernelID = 6 #kernel type. It's specified in the function definition

num_seqs = 1
numPerSeq = 1
e1_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq])
e2_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq]) 
e3_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq]) 
theta_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq])

for i in range(num_seqs):
    e1_seqs[0 : N_timesteps, i * numPerSeq:(i+1) * numPerSeq] = GPSample(control_points, step_points, lower_bound, upper_bound, kernelID, numPerSeq)
    e2_seqs[0 : N_timesteps, i * numPerSeq:(i+1) * numPerSeq] = GPSample(control_points, step_points, lower_bound, upper_bound, kernelID, numPerSeq)
    e3_seqs[0 : N_timesteps, i * numPerSeq:(i+1) * numPerSeq] = GPSample(control_points, step_points, lower_bound, upper_bound, kernelID, numPerSeq)



# In[11]:


#print(e1_seqs)
print(e2_seqs.shape)
# print(theta_seqs.shape)


# In[12]:


import numpy
numpy.savetxt("e1_seqs_2.csv", e1_seqs,
              delimiter = ",")


# In[13]:


numpy.savetxt("e2_seqs_2.csv", e2_seqs,
              delimiter = ",")


# In[14]:


numpy.savetxt("e3_seqs_2.csv", e3_seqs,
              delimiter = ",")


# In[ ]:


a = plt.plot(e1_seqs, marker="*")


# In[28]:


a = plt.plot(e2_seqs)


# In[29]:


a = plt.plot(theta_seqs)


# In[30]:


plt.scatter(e1_seqs[:,0],e2_seqs[:,0], marker='.',s=1)
plt.scatter(e1_seqs[:,1],e2_seqs[:,1], marker='.',s=1)
plt.scatter(e1_seqs[:,2],e2_seqs[:,2], marker='.',s=1)
plt.scatter(e1_seqs[:,3],e2_seqs[:,3], marker='.',s=1)
plt.scatter(e1_seqs[:,4],e2_seqs[:,4], marker='.',s=1)
plt.scatter(e1_seqs[:,5],e2_seqs[:,5], marker='.',s=1)
plt.scatter(e1_seqs[:,6],e2_seqs[:,6], marker='.',s=1)
plt.scatter(e1_seqs[:,7],e2_seqs[:,7], marker='.',s=1)
plt.scatter(e1_seqs[:,8],e2_seqs[:,8], marker='.',s=1)
plt.scatter(e1_seqs[:,9],e2_seqs[:,9], marker='.',s=1)
plt.xlim(-0.03, 0.03) 
plt.ylim(-0.03, 0.03)
plt.xlabel('$\epsilon_1$')
plt.ylabel('$\epsilon_2$')
plt.axis('equal')


# In[10]:


e1=np.reshape(e1_seqs,-1,order='F')
e2=np.reshape(e2_seqs,-1,order='F')
theta=np.reshape(theta_seqs,-1,order='F')


# In[11]:


a=plt.plot(e1)
a=plt.plot(e2)
a=plt.plot(theta)


# In[14]:


counts,ybins,xbins,image = plt.hist2d(e1,e2,50, range =[[-e_max, e_max], [-e_max, e_max]],cmap=plt.cm.Reds,vmin=0,vmax=10000)
plt.colorbar()
plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],levels=[0, 2000, 4000, 6000, 8000, 10000],linewidths=1)
plt.axis('equal')


# In[15]:


plt.scatter(e1,e2,marker='.')
plt.axis('equal')


# In[15]:


e_11 = (e1-e2)*np.cos(theta)*np.cos(theta)+e2
e_12 = np.sin(theta)*np.cos(theta)*(e1-e2)
e_22 = (-e1+e2)*np.cos(theta)*np.cos(theta)+e1


# In[16]:


a,b,c = plt.hist(e_11,100)
a,b,c = plt.hist(e_12,100)
a,b,c = plt.hist(e_22,100)


# In[19]:


e_1 = 0.5*e_22+0.5*e_11+0.5*np.sqrt(e_11*e_11-2*e_11*e_22+4*e_12*e_12+e_22*e_22)
e_2 = 0.5*e_22+0.5*e_11-0.5*np.sqrt(e_11*e_11-2*e_11*e_22+4*e_12*e_12+e_22*e_22)


# In[20]:


counts,ybins,xbins,image = plt.hist2d(e_1,e_2,50, range =[[-e_max, e_max], [-e_max, e_max]],cmap=plt.cm.Reds,vmin=0,vmax=20000)
plt.colorbar()
plt.contour(counts,extent=[-xbins.min(),-xbins.max(),-ybins.min(),-ybins.max()],levels=[0, 4000, 8000, 12000, 16000, 20000],linewidths=1)
plt.axis('equal')


# In[21]:


plt.hist2d(e1_seqs)


# In[ ]:


a=np.arange(N_timesteps)
a.shape

e1_seqs.shape

b=np.array([a,a])
b.shape

c=np.array([a,b])
c.shape
