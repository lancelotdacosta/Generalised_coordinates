import sympy as sp
import numpy as np
from Routines.gen_coords import generalised_flow,serial_derivs_xt


def sol_gen_coords(F,x,t,x0,tilde_w0):
    '''Input:
        d-dimensional * sample-dimensional initial condition x0 (specified as numpy array)
        d-dimensional * order-dimensional * sample-dimensional generalised fluctuations at the initial condition tilde_w0 (specified as numpy array)
        d-dimensional flow function F, function of d-dimensional x, itself a function of t, all specified in sympy
        linearised boolean
    Returns:
        tilde_x0, a d-dimensional x (order+1)-dimensional x sample-dimensional numpy array,
        Specifying the values in generalised coordinates of the solution at the initial condition to the differential equation
        dx(t)=F(x(t))dt+dw(t)
        If linearised = True, then we linearised F. This means we set all derivatives of F of order higher than one to be zero
    '''

    [dim,order,N]=tilde_w0.shape

    dxdtn= serial_derivs_xt(x,t,order) #get serial time derivatives of x for each order and dimension (starting with order 0)

    tilde_x0= np.empty([dim, order+1, N]) #initialise generalised coordinates at zero of solution to SDE
    tilde_x0[:, 0, :]=x0 #say that order zero is the initial condition

    for n in range(N): #iterate over each sample
        if N>=10**2 and n%10**1==0:
            print('Zig zag algorithm-- sample',n,'/',N) #count iterations

        dFdtn = generalised_flow(F,t,order)  # serial time derivatives of F for each order and dimension (starting with order 0)

        # if linearised and order>1:
        #     dFdtn[:, 2:] = sp.zeros(dim,order-2) #in the linearised case ignore all derivatives of order strictly higher than one

        for o in range(1,order +1):
            #evaluate (o-1)-th time derivative of F(x(t)) at the [ (o-1)-th,...,1st, 0th time derivatives of x(t) at t=0 ]
            for r in reversed(range(o)): #starts by o-1 and steps backward to 0
                for d in range(dim):
                    dFdtn[:,o-1] = dFdtn[:,o-1].subs(dxdtn[d,r],tilde_x0[d,r,n])

            #evaluate o-th time derivative of x(t) at t=0
            tilde_x0[:,o,n]= np.array(dFdtn[:,o-1]).astype(np.float64).reshape([dim]) + tilde_w0[:,o-1,n]

    return tilde_x0



# '''B: Simulation and test involving Lorentz system'''
#
# 'Setting up Lorentz system'
# #symbolic time variable
# t = sp.Symbol("t")
#
# #symbolic space variable
# x= sp.Matrix([sp.Function("x0")(t),
#               sp.Function("x1")(t),
#               sp.Function("x2")(t)])
#
# #parameters of Lorentz system
# sigma = 10
# rho=28
# beta=8/3
#
# #Flow of Lorentz system
# F=sp.Matrix([sigma*(x[1]-x[0]),
#              x[0] * (rho-x[2]) - x[1],
#              x[0]* x[1] - beta * x[2]])
#
#
# 'Parameters of simulation'
#
# order = 3
# dim =3 #State space dimension
# N = 2 #number of sample paths
#
# np.random.seed(1)
# x0 = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=N).T #initial condition
# #x0 = np.array([1,2,3]).reshape([dim,N])
# #tilde_w0= np.random.multivariate_normal(mean=np.zeros(dim*order), cov=np.eye(dim*order), size=N).reshape([dim, order, N])
# tilde_w0 = np.zeros([dim,order,N])
#
# 'Simulation'
#
# tilde_x0 = sol_gen_coords(F,x,t,x0,tilde_w0)
#
# 'Test of results'
# #the following prints should be duplicates in the case that we set tilde_w0=0 for the lorentz system flow
#
# def flow(x): #Numpy form of flow
#     y=np.empty(x.shape)
#     y[0,:]= sigma*(x[1,:]-x[0,:])
#     y[1, :] = x[0,:]*(rho-x[2,:])-x[1,:]
#     y[2, :] = x[0,:]*x[1,:]-beta*x[2,:]
#     return y
#
# for n in range(N):
#     print(tilde_x0[:,0,n], x0[:,n])
#     print(tilde_x0[:,1,n], flow(x0)[:,n])
#
#     second_order= np.array([sigma*(tilde_x0[1,1,n]-tilde_x0[0,1,n]),
#                 tilde_x0[0,1,n]*(rho-tilde_x0[2,0,n])-tilde_x0[1,1,n]-tilde_x0[0,0,n]*tilde_x0[2,1,n],
#                 tilde_x0[0,1,n]*tilde_x0[1,0,n]+tilde_x0[0,0,n]*tilde_x0[1,1,n]-beta*tilde_x0[2,1,n]])
#     print(tilde_x0[:,2,n], second_order)
#
#     third_order= np.array([sigma*(tilde_x0[1,2,n]-tilde_x0[0,2,n]),
#                 tilde_x0[0,2,n]*(rho-tilde_x0[2,0,n]) -tilde_x0[1,2,n]-tilde_x0[0,0,n]*tilde_x0[2,2,n]-2*tilde_x0[0,1,n]*tilde_x0[2,1,n],
#                 tilde_x0[0,2,n]*tilde_x0[1,0,n]+tilde_x0[0,0,n]*tilde_x0[1,2,n]+2*tilde_x0[0,1,n]*tilde_x0[1,1,n]-beta*tilde_x0[2,2,n]])
#     print(tilde_x0[:,3,n], third_order)





