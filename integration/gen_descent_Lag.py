'''In this script, we aim to recover the path of least action by minimizing the Lagrangian in a moving frame of reference.
For this, we need the Lagrangian, more specifically it's gradient, and also Dx.
It will then be a matter of using Euler to integrate dot x=Dx-grad L(x)
Therefore, we just need to specify a Flow function defined as f(x)=Dx-grad L(x).'''

import numpy as np
import Routines.gen_coords as gen_coords
import sympy as sp


def flow_gen_gd_Lag(grad_Lagrangian,x,t,order):
    #recall order is the top derivative, so the vector genx is of size o+1 in the order dimension
    #the assumption is that grad_Lagrangian depends on all components of genx
    genx = gen_coords.serial_derivs_xt(x, t, order + 1)  # shape [dim,order+1]
    return flow_gengd_Lag(grad_Lagrangian,genx)

def flow_gengd_Lag(grad_Lagrangian,genx):
    Dgenx = gen_coords.operator_D(genx)         # shape [dim,order+1]
    return Dgenx-grad_Lagrangian


def Euler_gen_flow(tilde_x0,Flow,genx,Time):
    print('====Euler integration of generalised flow====')
    T = Time.size
    [dim,order]=tilde_x0.shape
    tilde_xt = np.empty([dim,order,T])
    tilde_xt[:,:, 0]=tilde_x0
    F_lambdify = sp.lambdify(genx, Flow, "numpy")
    for t in range(T-1):
        if t % 10**5 == 0:
            print(t, '/', T)
        step=Time[t+1]-Time[t]
        tilde_xt[:,:,t+1]=tilde_xt[:,:,t] + step * F_lambdify(*tilde_xt[:,:,t].ravel())
    return tilde_xt



def Euler_gen_flow_slow(tilde_x0,Flow,genx,Time):
    #Unlike the one above this method has not been optimized for speed
    #they both yield the same result though
    print('====Euler integration of generalised flow====')
    T = Time.size
    [dim,order]=tilde_x0.shape
    tilde_xt = np.empty([dim,order,T])
    tilde_xt[:,:, 0]=tilde_x0
    for t in range(T-1):
        if t % 100 == 0:
            print(t, '/', T)
        step=Time[t+1]-Time[t]
        tilde_xt[:,:,t+1]=tilde_xt[:,:,t] + step * gen_coords.gen_coord_eval(Flow,genx,tilde_xt[:,:,t])
    return tilde_xt

























