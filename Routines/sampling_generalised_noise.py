'''This script generates generalised noise samples at time t, given a smooth (at least around t) autocorrelation function K,
where each order of generalised noise is one dimensional.
Assuming that the noise is a mean zero smooth Gaussian process with the given autocorrelation function'''

import numpy as np
import sympy as sp
from scipy.stats import multivariate_normal
import scipy


def formal_serial_derivs(K, wrt, order):  # formal symbolic expressions of the derivatives of K up to certain order (including order 0)
    '''Input: Autocorrelation function K given as a symbolic expression of wrt using sympy
    wrt= argument with respect to whichWe differentiate K
    order=Natural number being the order of the derivatives including order zero, ie if we want up to the fifth derivative then we require order =6
    '''
    form_derivs = [None] * order #create empty list of given length
    for n in range(order):
        form_derivs[n] = K
        K = K.diff(wrt)
    return form_derivs

def num_serial_derivs(K, wrt, at, order):  # expressions of derivatives where 'wrt' is evaluated at 'at'
    form_derivs = formal_serial_derivs(K, wrt, order)
    num_derivs = [None] * order #create empty list of given length
    for n in range(order):
        num_derivs[n] = form_derivs[n].subs(wrt, at)
    return num_derivs

def gen_covariance(K, wrt, at, order): #order=number of generalised coordinates including 0
    gen_cov = np.empty([order, order])
    num_derivs = num_serial_derivs(K, wrt, at, 2 * order - 1)
    for n in range(order):
        for m in range(order):
            gen_cov [n, m] = (-1) ** n * num_derivs[n + m]
    return gen_cov

def dist_gen_noise(K,wrt,at,order): #distribution of generalised noise at some time point 'at'
    gen_cov = gen_covariance(K, wrt, at, order)
    return multivariate_normal(mean= np.zeros(order), cov=gen_cov)

def sample_gen_noise(K,wrt,at,order,N): #N=number of samples,
    dist=dist_gen_noise(K,wrt,at,order)
    return dist.rvs(size=N).T

#def sample_gen_noise_nd(K,wrt,at,order,N,dim=1): #flawed method. Correct one below #dim=Dimension of each order of generalised noise. This is to generate n-dimensional noise that is independent across dimensions
#    dist=dist_gen_noise(K,wrt,at,order)
#    samples=dist.rvs(size=N*dim).T
#    return samples.reshape([dim,order,N])

#def sample_gen_noise_nd(K,wrt,at,order,N,dim=1): #flawed method. Correct one below #dim=Dimension of each order of generalised noise. This is to generate n-dimensional noise that is independent across dimensions
#    dist=dist_gen_noise(K,wrt,at,order)
#    temporal_samples=dist.rvs(size=N).T
#    spatial_samples=np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim),size=N).T
#    samples=np.empty([dim, order, N])
#    for n in range(N):
#        samples[:,:,n]=np.outer(spatial_samples[:,n],temporal_samples[:,n])
#    return samples

def gen_covariance_nd(K, wrt, at, order,dim=1):
    #one could use this function to generate the generalised covariance of additive noise with spatial covariance S by replacing np.eye(dim) by S
    gen_cov = gen_covariance(K, wrt, at, order)
    gen_cov_nd= np.kron(gen_cov,np.eye(dim))
    return gen_cov_nd

def sample_gen_noise_nd(K,wrt,at,order,N,dim=1): #dim=Dimension of each order of generalised noise. This is to generate n-dimensional noise that is independent across dimensions
    gen_cov_nd = gen_covariance_nd(K, wrt, at, order,dim)
    samples_temp=np.random.multivariate_normal(mean=np.zeros(order*dim), cov=gen_cov_nd, size=N).T
    samples=np.empty([dim, order, N])
    for o in range(order):
        for d in range(dim):
            #print(o,d,o*dim+d)
            samples[d,o,:]= samples_temp[o*dim+d,:]
    return samples



'''Test of method'''
#beta = sp.symbols('beta')
# beta=1
# h = sp.symbols('h')
#
# K = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h) # NORMALISED Gaussian kernel given as a symbolic expression
# K = sp.exp(-beta / 2 * h * h) # UNNORMALISED Gaussian kernel given as a symbolic expression
# wrt=h
# at=0
# order =3
# #N=10**6
# #dim=2
#
# form_derivs=formal_serial_derivs(K, wrt, order)
# num_derivs=num_serial_derivs(K, wrt, at, order)
#gen_cov=gen_covariance(K, wrt, at, order)
#dist =dist_gen_noise(K,wrt,at,order)
#sample=sample_gen_noise(K,wrt,at,order,N)
#sample_nd=sample_gen_noise_nd(K,wrt,at,order,N,dim)

#errors= np.zeros([dim, order, order])
#for d in range(dim):
#    errors[d,:,:]= gen_cov-np.cov(sample_nd[d,:,:])

#print(errors[0,:,:])
#print(errors[1,:,:])

'''Finding the sequence of derivatives for Gaussian kernel'''
# #beta = sp.symbols('beta')
# beta=1
# h = sp.symbols('h')
#
# K = sp.exp(-beta / 2 * h * h) # UNNORMALISED Gaussian kernel given as a symbolic expression
# wrt=h
# at=0
# order =4
#
# num_derivs=num_serial_derivs(K, wrt, at, order)
# print(num_derivs)
#
# num_derivs=np.array(num_derivs)
# print(num_derivs)
#
# odd=[i for i in range(-1,order) if i%2 != 0]
# attempt_derivs=scipy.special.factorial2(odd)
# print(attempt_derivs)
#
# indices= [i+1 for i in odd]
# num_derivs=num_derivs[indices]
# print(num_derivs/attempt_derivs)

'''Finding the generalised covariance and precision in the white noise limit of beta to \infty'''

# h = sp.symbols('h')
# wrt=h
# at=0
# order =4
#
# for beta in np.geomspace(1,10000,6):
#     print('beta',beta)
#     K = sp.sqrt(beta / (2 * sp.pi))*sp.exp(-beta / 2 * h * h)  # UNNORMALISED Gaussian kernel given as a symbolic expression
#     gen_cov=gen_covariance(K, wrt, at, order)
#     gen_pres= np.linalg.inv(gen_cov)
#     print('--cov',gen_cov[0,0])
#     print('--pres', gen_pres[0,0])

