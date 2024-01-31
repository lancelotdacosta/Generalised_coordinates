'''Subroutines for methods using generalised coordinates'''
import numpy as np
import sympy as sp
from Routines.sampling_generalised_noise import gen_covariance_nd
from Routines import sampling_generalised_noise, colourline
import matplotlib.pyplot as plt


def generalised_flow(F,t,order):
    '''This function inputs the sympy flow and the number of orders of motion and
     outputs the generalized flow in sympy form'''

    dim = F.shape[0]

    genF = sp.zeros(dim,
                     order)  # serial time derivatives of F for each order and dimension (starting with order 0)
    for d in range(dim):
        for o in range(order):
            # if not lin:
            genF[d, o] = sp.diff(F[d], (t, o))  # o-th time derivative of the d-th component of the function F(x(t))
            # elif o>=1:
            #     genF[d, o] = sp.diff(F[d], (t, 1))  # o-th time derivative of the d-th component of the function F(x(t)) (linearised version)
    return genF

def generalised_flow_lin(F, x, t, order):
    '''This does the same as above but by ignoring all derivatives of order higher than one of the flow.
    That is it operates under the local linear approximation'''
    dim = F.shape[0]

    genF = sp.zeros(dim,
                    order)  # serial time derivatives of F for each order and dimension (starting with order 0)

    for o in range(order):
        if o == 0:
            genF[:, o] = F  # o-th time derivative of the d-th component of the function F(x(t))
        else:
            for d in range(dim):
                genF[d, o] = sp.diff(F[d], (x, 1)).T @ sp.diff(x, (t, o))

    return genF


def serial_derivs_xt(x,t,order):

    dim = x.shape[0]

    genx= sp.zeros(dim,order) #initialise serial time derivatives of x for each order and dimension (starting with order 0)
    for d in range(dim):
        for o in range(order):
            genx[d,o]=sp.diff(x[d], (t,o)) #o-th time derivative of the d-th component of the function x(t)

    return genx


def gen_coord_replace(genF,genx,tilde_x):
    [dim, order] = genx.shape
    for o in reversed(range(order)):  # starts by o-1 and steps backward to 0
        for d in range(dim):
            # print(d, o)
            #genF[:,o:] = genF[:,o:].subs(genx[d, o], tilde_x[d, o])
            genF = genF.subs(genx[d, o], tilde_x[d, o])
    return genF


def sympy_to_numpy(genF):
    return np.array(genF).astype(np.float64)
def gen_coord_eval(genF,genx,tilde_x):
    return sympy_to_numpy(gen_coord_replace(genF,genx,tilde_x))

# def gen_coord_eval2(genF,genx,tilde_x):
#     gen=sp.lambdify(genx,genx,"numpy")
#     res=gen(tilde_x)
#     return np.array(genF).astype(np.float64)#.reshape(tilde_x.shape)



def operator_D(genx):
    '''applies the operator D onto generalised x'''
    [dim,order]=genx.shape
    Dgenx=sp.zeros(dim,order)
    Dgenx[:,:-1]=genx[:,1:]
    return Dgenx

def truncate_order(genx):
    '''truncate last order'''
    return genx[:,:-1]

def matrix_D(dim,order):
    D = np.zeros([(order + 1) * dim, (order + 1) * dim])
    for i in range(order):
        j = dim * i
        h = j + dim
        k = h + dim
        D[j:h, h:k] = np.eye(dim)
    return D

def Lagrangian(F, x, t, K, h, order, lin=False, trun=True, epsilon_w=1):
    # epsilon_w is the additive scaling of the state space noise

    genx = serial_derivs_xt(x, t, order + 1)  # shape [dim,order+1]
    if lin:
        genF = generalised_flow_lin(F, x, t, order+1)  # shape [dim,order+1]
    else:
        genF = generalised_flow(F, t, order+1)  # shape [dim,order+1]

    Dgenx = operator_D(genx)
    dim=F.shape[0]

    if trun:
        vec = truncate_order(Dgenx - genF)
        rvec = (vec.T).reshape(dim * order, 1)
        gen_cov_nd = epsilon_w**2*sampling_generalised_noise.gen_covariance_nd(K=K, wrt=h, at=0, order=order, dim=dim)
    else:
        vec = Dgenx - genF
        rvec = (vec.T).reshape(dim * (order+1), 1)
        gen_cov_nd = epsilon_w**2*sampling_generalised_noise.gen_covariance_nd(K=K, wrt=h, at=0, order=order+1, dim=dim)

    gen_pres_nd = np.linalg.inv(gen_cov_nd)  # precision of generalised fluctuations

    Lagrangian = rvec.T @ gen_pres_nd @ rvec / 2  # shape = 1

    return Lagrangian

def Lagrangian_gradient(F, x, t, K, h, order, lin=False, trun=True):

    Lag=Lagrangian(F, x, t, K, h, order, lin, trun)
    genx = serial_derivs_xt(x, t, order + 1)  # shape [dim,order+1]

    dim= x.shape[0]
    grad_Lagrangian = sp.zeros(dim, order + 1)
    for d in range(dim):
        for o in range(order + 1):
            # print(d,o)
            grad_Lagrangian[d, o] = sp.diff(Lag, genx[d, o])

    return grad_Lagrangian


def gen_gradient(F, genx):

    [dim,order_plus_one]= genx.shape
    grad_F = sp.zeros(dim, order_plus_one)
    for d in range(dim):
        for o in range(order_plus_one):
            # print(d,o)
            grad_F[d, o] = sp.diff(F, genx[d, o])

    return grad_F

def flatten_gen_coord(genx):
    [dim,order_plus_one]= genx.shape
    genx_flatten = (genx.T).reshape(dim * order_plus_one, 1)
    return genx_flatten


def gen_Hessian(F, genx):

    [dim,order_plus_one]= genx.shape
    grad_F = gen_gradient(F, genx)
    grad_F_flattened = flatten_gen_coord(grad_F)
    Hess_F= sp.zeros(dim* order_plus_one,dim* order_plus_one)
    for i in range(dim * order_plus_one):
        temp = gen_gradient(grad_F_flattened[i],genx)
        temp = flatten_gen_coord(temp)
        Hess_F[i,:]=temp.T

    return Hess_F



'Alternative routines for testing In the case of the OU process'

def Lagrangian_linear(B, x, t, K, h, order, trun=True):

    genx = serial_derivs_xt(x, t, order + 1)  # shape [dim,order+1]

    dim=B.shape[0]

    genF=sp.zeros(dim,order+1)
    for o in range(order+1):
        genF[:,o]=B@genx[:,o]

    Dgenx = operator_D(genx)

    if trun:
        vec = truncate_order(Dgenx - genF)
        rvec = (vec.T).reshape(dim * order, 1)
        gen_cov_nd = sampling_generalised_noise.gen_covariance_nd(K=K, wrt=h, at=0, order=order, dim=dim)
    else:
        vec = Dgenx - genF
        rvec = (vec.T).reshape(dim * (order+1), 1)
        gen_cov_nd = sampling_generalised_noise.gen_covariance_nd(K=K, wrt=h, at=0, order=order+1, dim=dim)

    gen_pres_nd = np.linalg.inv(gen_cov_nd)  # precision of generalised fluctuations

    Lagrangian = rvec.T @ gen_pres_nd @ rvec / 2  # shape = 1

    return Lagrangian

def Lagrangian_linear_gradient(B, x, t, K, h, order, trun=True):

    Lag=Lagrangian_linear(B, x, t, K, h, order, trun)
    genx = serial_derivs_xt(x, t, order + 1)  # shape [dim,order+1]

    dim= x.shape[0]
    grad_Lagrangian = sp.zeros(dim, order + 1)
    for d in range(dim):
        for o in range(order + 1):
            # print(d,o)
            grad_Lagrangian[d, o] = sp.diff(Lag, genx[d, o])

    return grad_Lagrangian