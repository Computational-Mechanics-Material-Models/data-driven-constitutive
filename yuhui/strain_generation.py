#!/usr/bin/env python

# Script to generate random strain histories for input to RVE simulations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct, ConstantKernel,
                                              Exponentiation)

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
    control_values = np.random.uniform(lower, upper, len(control_points)) # TODO: should that be an LHS to better cover the strain space?
    control_values[0] = 0.0 # Enforce zero initial strain

    gp.fit(control_points.reshape(-1, 1), control_values.reshape(-1, 1))
    return gp.sample_y(step_points.reshape(-1, 1), numSamples).reshape(-1, 1)
###corelation length?

# TODO: There seems to be multiple mechanisms to create multiple data sets
# e.g., numRealizations, num_seqs, numPreSeq, do we need all of them ?
def generate_strain_histories(control_points, step_points, lower_bound, upper_bound, kernelID, numRealizations, num_seqs, numPerSeq):
    N_timesteps = len(step_points)
    e1_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq])
    e2_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq]) 
    e3_seqs = np.zeros([N_timesteps, num_seqs * numPerSeq])
    for n in range(numRealizations):
        for i in range(num_seqs):
            e1_seqs[:, i * numPerSeq : (i+1) * numPerSeq] = GPSample(control_points, step_points, lower_bound, upper_bound, kernelID, numPerSeq)
            e2_seqs[:, i * numPerSeq : (i+1) * numPerSeq] = GPSample(control_points, step_points, lower_bound, upper_bound, kernelID, numPerSeq)
            e3_seqs[:, i * numPerSeq : (i+1) * numPerSeq] = GPSample(control_points, step_points, lower_bound, upper_bound, kernelID, numPerSeq)
        np.savetxt(f"e1_seqs_{n}.csv", e1_seqs, delimiter = ",")
        np.savetxt(f"e1_seqs_{n}.csv", e2_seqs, delimiter = ",")
        np.savetxt(f"e1_seqs_{n}.csv", e3_seqs, delimiter = ",")
    return e1_seqs, e2_seqs, e3_seqs


def main():
    from matplotlib import pyplot as plt

    tmax = 20 # total time
    N_timesteps = 300 # number of timesteps
    N_cp = 20 # number of control points

    control_points = np.linspace(0, tmax, N_cp)
    step_points = np.linspace(0, tmax, N_timesteps) #times at which we are making getting the values

    # Strain bounds
    min_strain = 0.0 # TODO: Should we allow negative values ? Bias in positive strain eigenvalues
    max_strain = 2e-2

    kernelID = 6
    numRealizations = 1
    num_seqs = 1
    numPerSeq = 1

    # Generate random strain histories
    e1_seqs, e2_seqs, e3_seqs = generate_strain_histories(control_points, step_points,
                                                          min_strain, max_strain,
                                                          kernelID,
                                                          numRealizations, num_seqs, numPerSeq)

    # Plots
    lo = 2 * min_strain
    hi = 2* max_strain

    # Plot generated strain history
    plt.figure(1)
    plt.plot(step_points, e1_seqs, label='e1')
    plt.plot(step_points, e2_seqs, label='e2')
    plt.plot(step_points, e3_seqs, label='e3')
    plt.xlabel('Time')
    plt.ylabel('Strain')
    plt.legend()
    plt.show()

    # Plot 2D strain path
    plt.figure(2)
    plt.plot(e1_seqs[:,0],e2_seqs[:,0],marker='>')
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel('e1')
    plt.ylabel('e2')
    plt.axis('equal')
    plt.show()

    # Plot 2D histogram of # TODO: what si this ?
    plt.figure(3)
    counts,ybins,xbins,image = plt.hist2d(e1_seqs[:,0],e2_seqs[:,0],50, range =[[lo, hi], [lo, hi]],cmap=plt.cm.Reds,vmin=0,vmax=10000)
    plt.colorbar()
    plt.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],levels=[0, 2000, 4000, 6000, 8000, 10000],linewidths=1)
    plt.xlabel('e1')
    plt.ylabel('e2')
    plt.axis('equal')
    plt.show()

    # TODO: I deleted the code below that made more histograms and was dependent
    # on theta, which was not well-defined in the Jupyter notebook.
    # See if we need it, it is in the git history


if __name__ == "__main__":
    main()