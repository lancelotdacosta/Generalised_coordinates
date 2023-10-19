'''Generalised coordinate solution using local linearised version of the flow
The following code is implemented using sympy for simplicity
A more sophisticated version could be done with automatic differentiation using, e.g. pytorch'''


import sympy as sp
import numpy as np
from integration.gen_coord import sol_gen_coords

def sol_lin_gen_coord(F,x,t,x0,tilde_w0):
    '''Input:
        d-dimensional * sample-dimensional initial condition x0 (specified as numpy array)
        d-dimensional * order-dimensional * sample-dimensional generalised fluctuations at the initial condition tilde_w0 (specified as numpy array)
        d-dimensional flow function F, function of d-dimensional x, itself a function of t, all specified in sympy
    Returns:
        tilde_x0, a d-dimensional x (order+1)-dimensional x sample-dimensional numpy array,
        Specifying the values in generalised coordinates of the solution at the initial condition to the differential equation
        dx(t)=F(x(t))dt+dw(t)
        where F has been linearised. In this case this means that we ignore all its derivatives of order higher than one
    '''

    [dim,order,N]=tilde_w0.shape

    dxdtn= sp.zeros(dim,order) #initialise serial time derivatives of x for each order and dimension (starting with order 0)
    for d in range(dim):
        for o in range(order):
            dxdtn[d,o]=sp.diff(x[d], (t,o)) #o-th time derivative of the d-th component of the function x(t)


    #compute Jacobian of F:
    Jf=sp.zeros(dim,dim)
    for d in range(dim):
        for e in range(dim):
            Jf[d,e]=sp.diff(F[d], x[e])

    tilde_x0= np.empty([dim, order+1, N]) #initialise generalised coordinates at zero of solution to SDE
    tilde_x0[:, 0, :]=x0 #say that order zero is the initial condition

    for n in range(N):
        if N>=10**2 and n%10**2==0:
            print('Linearised zig zag algorithm-- sample',n,'/',N) #count iterations

        # evaluate F at x0
        F_x0 = F
        for d in range(dim):
            F_x0 = F_x0.subs(x[d], x0[d, n])
        F_x0 = np.array(F_x0).astype(np.float64).reshape([dim])

        # evaluate Jacobian of F at x0:
        Jf_x0 = Jf
        for d in range(dim):
            Jf_x0 = Jf_x0.subs(x[d], x0[d,n])
        Jf_x0 = np.array(Jf_x0).astype(np.float64)

        for o in range(1,order +1):
            if o==1:
                tilde_x0[:, o, n]= F_x0 + tilde_w0[:,o-1,n]
            else:
                tilde_x0[:, o, n] = Jf_x0 @ tilde_x0[:, o-1, n] + tilde_w0[:, o - 1, n]

    return tilde_x0



'''Test involving OU process'''

# 'Setting up OU process'
# #symbolic time variable
# t = sp.Symbol("t")
#
# dim =2 #State space dimension
#
# #symbolic space variable
# x= sp.Matrix([sp.Function("x0")(t),
#               sp.Function("x1")(t)])
#
# #drift matrix of OU process
# B=np.random.normal(size=dim**2).reshape([dim,dim])
#
# #Flow of OU process sympy
# F=sp.Matrix(B)@x
#
# #Flow of OU process numpy
# def flow(x):
#     return B @ x
#
#
# '''Part 1: Specifying parameters of integration'''
#
# N = 1 #number of sample paths
#
# x0 = 10*np.ones([dim, N]) #initial condition
#
# 'Part 1a: specifying the serial derivatives of the noise process'
#
# order = 10
#
# tilde_w0= np.random.normal(size=dim*order*N).reshape([dim, order, N])
#
# '''Part 2: Generalised coordinates integration'''
#
# 'Part 2a: Manual method'
#
# tilde_x0_man = np.empty([dim, order+1, N])
# tilde_x0_man[:,0,:]=x0
#
# for i in range(order):
#     for n in range(N):
#         tilde_x0_man[:,i+1,n]= flow(tilde_x0_man[:,i,n]) + tilde_w0[:,i,n]
#
#
# 'Part 2b: General nonlinear generalised coordinates method'
#
# tilde_x0_gen = sol_gen_coords(F,x,t,x0,tilde_w0)
#
# 'Part 2c: Linearised generalised coordinates (ie this) method'
#
# tilde_x0_glin = sol_lin_gen_coord(F,x,t,x0,tilde_w0)
#
#
# 'Part 3: Comparison'
# #all results should agree and be zero
# print(np.sum(np.abs(tilde_x0_gen-tilde_x0_glin)))
# print(np.sum(np.abs(tilde_x0_man-tilde_x0_glin)))
# print(np.sum(np.abs(tilde_x0_man-tilde_x0_gen)))



