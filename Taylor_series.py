'''This script takes as input the sequence of time derivatives of a function on the time domain defined at some time point
And returns the corresponding Taylor polynomial on a specified time grid'''


import numpy as np
from functools import partial


def taylor_eval(derivs, at, Time): #evaluation of Taylor polynomial at some time grid 'Time', where the polynomial is specified by a sequence of derivatives 'derivs' at some point 'at'
    ''' #Input:
    derivs = A sequence of time derivatives up to order N (including order zero) specified as a 1D numpy array of size N+1
    at = The time Point at which the derivatives are taken, Specified as a scalar
    Time = A sequence of time points specified as a 1D numpy array at which to return the values of the Taylor polynomial'''
    order = derivs.size # number of derivatives +1 (counting for the zeroth derivative)
    Taylor = np.zeros(Time.size)
    for n in range(order):
        Taylor += derivs[n] * (Time - at) ** n / np.math.factorial(n) #add subsequent orders of the Taylor expansion evaluated on the time grid
    return Taylor

def taylor_pol(derivs, at):
    ''' #Input:
    derivs = A sequence of time derivatives up to order N (including order zero) specified as a 1D numpy array of size N+1
    at = The time Point at which the derivatives are taken, Specified as a scalar'''
    return partial(taylor_eval, derivs, at) #return the Taylor polynomial that can be evaluated at any time grid

'''Test of method'''
#derivs=np.ones(3)
#t=0
#Time=np.arange(0,5,1)

#Taylor_eval=taylor_eval(derivs, at, Time)
#Taylor_pol=taylor_pol(derivs, at)
#Taylor_eval2=Taylor_pol(Time)
