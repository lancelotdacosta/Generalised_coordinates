'''DRAFT!!!!'''


'''In this script, we aim to recover the path of least action by minimizing the Lagrangian in a moving frame of reference.
For this, we need the Lagrangian, more specifically it's gradient, and also Dx.
It will then be a matter of using Euler to integrate dot x=Dx-grad L(x)
Therefore, we just need to specify a Flow function defined as f(x)=Dx-grad L(x).'''

import numpy as np
import sympy as sp
from Routines.sampling_generalised_noise import gen_covariance_nd
from Routines import sampling_generalised_noise, colourline
from integration import zigzag, lin_zigzag, gen_descent_Lag
from Routines.gen_coords import generalised_flow,serial_derivs_xt
import matplotlib.pyplot as plt



# '''Suppose that we are given a sympy flow'''
#
# '''We also need a certain number of orders of motion to carry out the computation'''
#
# # F,x,t
#
#

def operator_D(genx):
    '''applies the operator D onto generalised x'''
    [dim,order]=genx.shape
    Dgenx=sp.zeros(dim,order)
    Dgenx[:,:-1]=genx[:,1:]
    return Dgenx

def operator_D_truncated(genx):
    '''applies the operator D truncated onto generalised x'''
    return genx[:,1:]


# def Lagrangian_gen():
#     'This needs the flow of the original equation and the generalized covariance of fluctuations'

def gen_coord_eval(genF,genx,tilde_x):
    [dim, order] = genx.shape
    for o in reversed(range(order)):  # starts by o-1 and steps backward to 0
        for d in range(dim):
            # print(d, o)
            genF = genF.subs(genx[d, o], tilde_x[d, o])
    return np.array(genF).astype(np.float64)#.reshape(tilde_x.shape)


def generalised_flow_lin(F, x, t, order):
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




'''test script involving Lotka volterra'''

'Part 0a: Specifying the parameters of the flow'

dim =2 #State space dimension

'Lotka volterra flow'
#parameters of Lorentz system
a = 1
b =1
c =1
d = 1
#only a/d affects the nature of the solution up to rescaling of space and time

'nd OU process flow'

alpha = 1 #Scaling of solenoidal flow

Q = np.array([0, 1, -1, 0]).reshape([dim, dim]) #solenoidal flow matrix

B= -(np.eye(dim)+alpha*Q) #drift matrix of OU process for a steady state distribution N(0,I)


'Part 0a: Specifying the numpy form of the flow'


'Lotka volterra flow'

# def flow(x):
#     y=np.empty(x.shape)
#     y[0,:]= a*x[0,:]-b*x[0,:]*x[1,:] #rate of change of prey
#     y[1, :] = c*x[0,:]*x[1,:]-d*x[1,:] #rate of change of predators
#     return y


'nd OU process flow'
def flow(x):
    return B @ x



'Part 0a: Specifying the sympy form of the flow'

#symbolic time variable
t = sp.Symbol("t")

#symbolic space variable
x= sp.Matrix([sp.Function("x0")(t),
              sp.Function("x1")(t)])


'Lotka volterra flow'
#Flow of Lotka Volterra system
# F=sp.Matrix([a*x[0]-b*x[0]*x[1],
#              c*x[0]*x[1]-d*x[1]])

'nd OU process flow'
F=B @ x


'Part 0b: Specifying the kernel of the noise process'

h = sp.symbols('h')
beta=1

K = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h) # NORMALISED Gaussian kernel given as a symbolic expression


'''Part 1: Specifying parameters of integration'''

N = 2 #number of sample paths

'Lotka volterra'
#initial condition
x0 = 1.5*np.ones([dim, N]) #initial condition representing the proportion of prey and predators

'nd OU process'
x0 = 10*np.ones([dim, N]) #initial condition representing the proportion of prey and predators


'Part 1a: getting the serial derivatives of the noise process'

order = 8

at=0

epsilon = 0 # scaling of noise
tilde_w0= epsilon*sampling_generalised_noise.sample_gen_noise_nd(K=K,wrt=h,at=at,order= order,N=N,dim=dim) #Generalised fluctuations at time zero



'''Part 2: Generalised coordinates integration'''

'Part 2a: Getting serial derivatives of solution for each sample at time <at> '

tilde_x0=gen_coord.sol_gen_coords(F,x,t,x0,tilde_w0)[:,:,0]
# tilde_x0_test=gen_coord_test.sol_gen_coords(F,x,t,x0,tilde_w0)

tilde_x0_lin=lin_gen_coord.sol_lin_gen_coord(F,x,t,x0,tilde_w0)[:,:,0]

'''Part 3: construction of lagrangian'''

'''Part 3a: initialise generalised x and flow in sympy'''

genx=serial_derivs_xt(x,t,order+1) #shape [dim,order] #what is very important is that genx be the same shape as the input
genF=generalised_flow(F,t,order) #shape [dim,order]

genF_lin=generalised_flow_lin(F,x,t,order) #shape [dim,order]


'''Part 3b: initialise quadratic term in lagrangian'''
Dgenx=operator_D(genx)
Dgenx_tr=operator_D_truncated(genx)
# vec=Dgenx-genF
vec_tr=Dgenx_tr-genF
vec_tr_lin= Dgenx_tr-genF_lin

'Part 3b1: test least action path'

#non linear case
# Dgenx_eval=gen_coord_eval(Dgenx,genx,tilde_x0[:,:,0])
genF_eval=gen_coord_eval(genF,genx,tilde_x0) #=gen_coord_eval(genF,genx,tilde_x0[:,:-1,0])
# vec_eval=gen_coord_eval(vec,genx,tilde_x0[:,:,0])
var_Dgenx_eval=tilde_x0[:,1:] #should be equal to line below
Dgenx_tr_eval=gen_coord_eval(Dgenx_tr,genx,tilde_x0)
# var_vec_eval=var_Dgenx_eval-genF_eval #==0
vec_tr_eval=gen_coord_eval(vec_tr,genx,tilde_x0)

#linear case
genF_lin_eval=gen_coord_eval(genF,genx,tilde_x0_lin)
Dgenx_tr_eval=gen_coord_eval(Dgenx_tr,genx,tilde_x0_lin)
vec_tr_lin_eval=gen_coord_eval(vec_tr_lin,genx,tilde_x0_lin)



'''Part 3c: initialise generalised covariance and precision terms'''

gen_cov_nd=sampling_generalised_noise.gen_covariance_nd(K=K,wrt=h,at=at,order= order,dim=dim)
# gen_cov=sampling_generalised_noise.gen_covariance(K, wrt, at, order)

gen_pres_nd=np.linalg.inv(gen_cov_nd) #precision of generalised fluctuations

# np.linalg.eigvals(gen_pres_nd)

'''Part 3d: initialise lagrangian'''
[dim,order]=vec_tr.shape
rvec=(vec_tr.T).reshape(dim*order,1) #This is such that spatial components are arranged as units, and orders as tens

rvec_lin=(vec_tr_lin.T).reshape(dim*order,1) #This is such that spatial components are arranged as units, and orders as tens


Lagrangian=rvec.T @ gen_pres_nd @ rvec /2 #shape = 1

Lagrangian_lin=rvec_lin.T @ gen_pres_nd @ rvec_lin /2


'Part 3d1: test with other function'

Lag=  gen_descent_Lag.Lagrangian(F,x,t,K,h,order,lin=False,trun=False)
#lin=True,trun=True => Lag =Lagrangian_lin
#lin=False,trun=True => Lag =Lagrangian_lin

'Part 3d1: test least action path'

Lagrangian_eval=gen_coord_eval(Lagrangian,genx,tilde_x0) #==0

Lagrangian_lin_eval=gen_coord_eval(Lagrangian_lin,genx,tilde_x0_lin)

'''Part 3e: lagrangian gradient'''
grad_Lagrangian=sp.zeros(dim,order+1)
for d in range(dim):
    for o in range(order+1):
        # print(d,o)
        grad_Lagrangian[d,o]=sp.diff(Lagrangian, genx[d, o])

grad_Lagrangian_lin=sp.zeros(dim,order+1)
for d in range(dim):
    for o in range(order+1):
        # print(d,o)
        grad_Lagrangian_lin[d,o]=sp.diff(Lagrangian_lin, genx[d, o])


'Part 3e1: test least action path'

grad_Lagrangian_eval=gen_coord_eval(grad_Lagrangian,genx,tilde_x0) #==0

grad_Lagrangian_lin_eval=gen_coord_eval(grad_Lagrangian_lin,genx,tilde_x0_lin) #==0



'''Part 3f: flow of the generalised gradient descent on the Lagrangian'''
Flow = Dgenx-grad_Lagrangian

Flow_lin = Dgenx-grad_Lagrangian_lin


'''Part 4: Euler integration of flow'''

'Part 4a: specify time'
T=5
timesteps=T*100
Time = np.linspace(0,T,timesteps)

'Part 4b: Euler integration of generalised flow'
def Euler_gen_flow(tilde_x0,Flow,genx,Time):
    print('====Euler integration of generalised flow====')
    T = Time.size
    [dim,order]=tilde_x0.shape
    tilde_xt = np.empty([dim,order,T])
    tilde_xt[:,:, 0]=tilde_x0
    for t in range(T-1):
        if t % 100 == 0:
            print(t, '/', T)
        step=Time[t+1]-Time[t]
        tilde_xt[:,:,t+1]=tilde_xt[:,:,t] + step * gen_coord_eval(Flow,genx,tilde_xt[:,:,t])
    return tilde_xt

# xt_leastr= Euler_gen_flow(tilde_x0,Flow,genx,Time)
# xt_leastr_D= Euler_gen_flow(tilde_x0,Dgenx,genx,Time)
# xt_leastr_grad= Euler_gen_flow(tilde_x0,-grad_Lagrangian,genx,Time)

# xt_leastr_lin= Euler_gen_flow(tilde_x0_lin,Flow_lin,genx,Time)
# xt_leastr_D= Euler_gen_flow(tilde_x0_lin,Dgenx,genx,Time)
# xt_leastr_lin_grad= Euler_gen_flow(tilde_x0_lin,-grad_Lagrangian_lin,genx,Time)

tilde_x0=np.zeros([dim,order+1])
tilde_x0[:,0]=10*np.ones([dim])
xt_leastr_lin= Euler_gen_flow(tilde_x0,Flow_lin,genx,Time)

'Part 4b1: plotting evolution of lagrangian'

# Lag_leastr=np.zeros(Time.shape)
# for t in range(timesteps):
#     print(t, '/', timesteps)
#     Lag_leastr[t]=gen_coord_eval(Lagrangian, genx, xt_leastr[:, :, t])

Lag_lin_leastr=np.zeros(Time.shape)
for t in range(timesteps):
    if t%100==0:
        print(t, '/', timesteps)
    Lag_lin_leastr[t]=gen_coord_eval(Lagrangian_lin, genx, xt_leastr_lin[:, :, t])

'Part 4c: Plots nd OU process'

plt.figure(0)
plt.clf()
plt.title(f'dot x = Dx - grad L', fontsize=14)
plt.plot(Time,xt_leastr_lin[0,0,:])
plt.plot(Time,xt_leastr_lin[1,0,:])
plt.plot(Time,Lag_lin_leastr,c='black')
plt.ylim(top=15)
# plt.yscale('log')

plt.figure(1)
plt.clf()
plt.suptitle(f'2D Coloured OU process', fontsize=16)
plt.title(f'Least action recovery vs least zig zag', fontsize=14)
colourline.plot_cool(xt_leastr_lin[0,0,:], xt_leastr_lin[1,0,:])
# colourline.plot_cmap(xt_leastr_D[0,0,:], xt_leastr_D[1,0,:])
# plt.xlim(right=2000,left=-2000)
# plt.ylim(top=2000,bottom=-2000)




'Part 4c: Plots Lotka Volterra'

# plt.figure(4)
# plt.clf()
# plt.ylim([-1,20])
# plt.title(f'dot x = Dx - grad L', fontsize=14)
# plt.plot(Time,xt_leastr_lin[0,0,:])
# plt.plot(Time,xt_leastr_lin[1,0,:])
# plt.plot(Time,Lag_lin_leastr,c='black')
#
#
# plt.figure(5)
# plt.clf()
# plt.title(f'dot x = - grad L', fontsize=14)
# # plt.yscale('log')
# plt.plot(Time,xt_leastr_lin_grad[0,0,:])
# plt.plot(Time,xt_leastr_lin_grad[1,0,:])
#
# # plt.figure(1)
# # plt.clf()
# # plt.ylim([-1,8])
# # plt.title(f'dot x = Dx - grad L', fontsize=14)
# # plt.plot(Time,xt_leastr[0,0,:])
# # plt.plot(Time,xt_leastr[1,0,:])
# # plt.plot(Time,Lag_leastr,c='black')
# #
# plt.figure(2)
# plt.clf()
# plt.title(f'dot x = Dx', fontsize=14)
# # plt.yscale('log')
# plt.plot(Time,xt_leastr_D[0,0,:])
# plt.plot(Time,xt_leastr_D[1,0,:])
# #
# # plt.figure(3)
# # plt.clf()
# # plt.title(f'dot x = - grad L', fontsize=14)
# # # plt.yscale('log')
# # plt.plot(Time,xt_leastr_grad[0,0,:])
# # plt.plot(Time,xt_leastr_grad[1,0,:])




# ==============================DRAFT CODE=============================================================================


# for o in range(1, order + 1):
#     # evaluate (o-1)-th time derivative of F(x(t)) at the [ (o-1)-th,...,1st, 0th time derivatives of x(t) at t=0 ]
#     for r in reversed(range(o)):  # starts by o-1 and steps backward to 0
#         for d in range(dim):
#             Flow[:, o - 1] = Flow[:, o - 1].subs(genx[d, r], tilde_x[d, r])



#
# def generalised_flow_eval_2(genF,dxdtn, tilde_x,order):
#
#     [dim,order] = genF.shape
#
#     # for o in range(1, order + 1):
#     #     # evaluate (o-1)-th time derivative of F(x(t)) at the [ (o-1)-th,...,1st, 0th time derivatives of x(t) at t=0 ]
#     #     for r in reversed(range(o)):  # starts by o-1 and steps backward to 0
#     #         for d in range(dim):
#     #             genF[:, o - 1] = genF[:, o - 1].subs(dxdtn[d, r], tilde_x[d, r])
#
#     for r in reversed(range(order)):  # starts by o-1 and steps backward to 0
#         for d in range(dim):
#             genF = genF.subs(dxdtn[d, r], tilde_x[d, r])
#
#     return genF
#
#
# def generalised_flow_eval(genF, dxdtn, tilde_x):
#     [dim, order] = genF.shape
#
#     for o in range(1, order + 1):
#         # evaluate (o-1)-th time derivative of F(x(t)) at the [ (o-1)-th,...,1st, 0th time derivatives of x(t) at t=0 ]
#         for r in reversed(range(o)):  # starts by o-1 and steps backward to 0
#             for d in range(dim):
#                 genF[:, o - 1] = genF[:, o - 1].subs(dxdtn[d, r], tilde_x[d, r])
#
#     # for r in reversed(range(order)):  # starts by o-1 and steps backward to 0
#     #     for d in range(dim):
#     #         genF = genF.subs(dxdtn[d, r], tilde_x[d, r])
#
#     return genF

