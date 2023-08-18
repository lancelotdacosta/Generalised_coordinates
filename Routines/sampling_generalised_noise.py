'''This script generates generalised noise samples at time t, given a smooth (at least around t) autocorrelation function K,
Assuming that the noise is a mean zero smooth Gaussian process with the given autocorrelation function'''

import numpy as np
import sympy as sp
from scipy.stats import multivariate_normal


def formal_serial_derivs(K, wrt, order):  # formal symbolic expressions of the derivatives of K up to certain order (including order 0)
    '''Input: Autocorrelation function K given as a symbolic expression of wrt using sympy
    wrt= argument with respect to whichWe differentiate K
    order=Natural number being the order of the videos
    '''
    form_derivs = [None] * (order + 1) #create empty list of given length
    for n in range(order + 1):
        form_derivs[n] = K
        K = K.diff(wrt)
    return form_derivs

def num_serial_derivs(K, wrt, at, order):  # expressions of derivatives where 'wrt' is evaluated at 'at'
    form_derivs = formal_serial_derivs(K, wrt, order)
    num_derivs = [None] * (order + 1) #create empty list of given length
    for n in range(order + 1):
        num_derivs[n] = form_derivs[n].subs(wrt, at)
    return num_derivs

def gen_covariance(K, wrt, at, order): #order=number of generalised coordinates including 0
    gen_cov = np.empty([order + 1, order + 1])
    num_derivs = num_serial_derivs(K, wrt, at, 2 * order)
    for n in range(order + 1):
        for m in range(order + 1):
            gen_cov [n, m] = (-1) ** n * num_derivs[n + m]
    return gen_cov

def dist_gen_noise(K,wrt,at,order): #distribution of generalised noise at some time point 'at'
    gen_cov = gen_covariance(K, wrt, at, order)
    return multivariate_normal(mean= np.zeros(order + 1), cov=gen_cov)

def sample_gen_noise(K,wrt,at,order,N): #N=number of samples
    dist=dist_gen_noise(K,wrt,at,order)
    return dist.rvs(size=N).T


'''Test of method'''
#beta = sp.symbols('beta')
#beta=1
#h = sp.symbols('h')

#K = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h) # NORMALISED Gaussian kernel given as a symbolic expression
#wrt=h
#at=0
#order =3
#N=10

#form_derivs=formal_serial_derivs(K, wrt, order)
#num_derivs=num_serial_derivs(K, wrt, at, order)
#gen_cov=gen_covariance(K, wrt, at, order)
#dist =dist_gen_noise(K,wrt,at,order)
#sample=sample_gen_noise(K,wrt,at,order,N)
