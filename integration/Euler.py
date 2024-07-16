'''this script integrates an SDE pathwise, given sample paths of the noise process wt, on a time grid Time, and given an initial condition x0
This is done using the Euler-Maruyama integration method'''

import numpy as np

def Euler_integration(x0,f,wt,Time):
    xt = np.empty(wt.shape)
    xt[:, 0, :]=x0
    T=Time.size
    for t in range(T-1):
        step=Time[t+1]-Time[t]
        xt[:,t+1,:]=xt[:,t,:] + step*(f(xt[:,t,:])+wt[:,t,:])
    return xt

def RK45_integration(x0,f,wt,Time):
    xt = np.empty(wt.shape)
    xt[:, 0, :]=x0
    T=Time.size
    for t in range(T-1):
        step=Time[t+1]-Time[t]
        xt[:,t+1,:]=xt[:,t,:] + step*(f(xt[:,t,:])+wt[:,t,:])
    return xt