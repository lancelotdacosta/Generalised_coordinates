'''In this script, we aim to recover the path of least action by minimizing the Lagrangian in a moving frame of reference.
For this, we need the Lagrangian, more specifically it's gradient, and also Dx.
It will then be a matter of using the Euler sub routine to integrate dot x=Dx-grad L(x)
Therefore, we just need to specify a flow function defined as f(x)=Dx-grad L(x).'''

import numpy as np
import sympy as sp

def Lagrangian():
    'This needs the flow of the original equation and the generalized covariance of fluctuations'

'''Suppose that we are given a sympy flow'''

'''We also need a certain number of orders of motion to carry out the computation'''

# F,x,t

def generalised_flow_gen(F,t,order):
    '''This function inputs the sympy flow and the number of orders of motion and
     outputs the generalized flow in sympy form'''

    dim = F.shape[0]

    genF = sp.zeros(dim,
                     order)  # serial time derivatives of F for each order and dimension (starting with order 0)
    for d in range(dim):
        for o in range(order):
            genF[d, o] = sp.diff(F[d], (t, o))  # o-th time derivative of the d-th component of the function F(x(t))

    return genF

def serial_derivs_xt(x,t,order):

    dim = x.shape[0]

    dxdtn= sp.zeros(dim,order) #initialise serial time derivatives of x for each order and dimension (starting with order 0)
    for d in range(dim):
        for o in range(order):
            dxdtn[d,o]=sp.diff(x[d], (t,o)) #o-th time derivative of the d-th component of the function x(t)

    return dxdtn

def generalised_flow_eval_2(genF,dxdtn, tilde_x,order):

    [dim,order] = genF.shape

    # for o in range(1, order + 1):
    #     # evaluate (o-1)-th time derivative of F(x(t)) at the [ (o-1)-th,...,1st, 0th time derivatives of x(t) at t=0 ]
    #     for r in reversed(range(o)):  # starts by o-1 and steps backward to 0
    #         for d in range(dim):
    #             genF[:, o - 1] = genF[:, o - 1].subs(dxdtn[d, r], tilde_x[d, r])

    for r in reversed(range(order)):  # starts by o-1 and steps backward to 0
        for d in range(dim):
            genF = genF.subs(dxdtn[d, r], tilde_x[d, r])

    return genF


def generalised_flow_eval(genF, dxdtn, tilde_x):
    [dim, order] = genF.shape

    for o in range(1, order + 1):
        # evaluate (o-1)-th time derivative of F(x(t)) at the [ (o-1)-th,...,1st, 0th time derivatives of x(t) at t=0 ]
        for r in reversed(range(o)):  # starts by o-1 and steps backward to 0
            for d in range(dim):
                genF[:, o - 1] = genF[:, o - 1].subs(dxdtn[d, r], tilde_x[d, r])

    # for r in reversed(range(order)):  # starts by o-1 and steps backward to 0
    #     for d in range(dim):
    #         genF = genF.subs(dxdtn[d, r], tilde_x[d, r])

    return genF

