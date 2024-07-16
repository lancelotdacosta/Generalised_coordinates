'''Here we do gen filt of the two dimensional OU process with smooth noise
and a (possibly non-linear) observation function'''

import numpy as np
import sympy as sp
from Routines import colourline,gen_coords, convolving_white_noise,gen_mod, gen_filt_eval,data_embedding
import matplotlib.pyplot as plt
from integration import Euler, gen_descent_Lag, gen_filt

'''Parameters of the method'''

# seed
np.random.seed(0)
# local linearisation of flows in the generative model
loc_lin=True
# Presence of (log det) Hessian in free energy term (and its gradient)
Hess=False
# Method to compute determinant of Hessian
meth_Hess = 'berkowitz'  # choices: 'berkowitz' (default), 'det_LU', 'bareis'
# (it seems that if generative model is fully linear best use 'det_LU', if not 'berkowitz'
# [have had overflow issues with 'berkowitz' in the linear case])
# choice of data assimilation and augmentation method
meth_data = 'Taylor' #choices: 'findiff' or 'Taylor'
# computing covariance, ie uncertainty of inference
cov=True
# integration method
meth_int= 'RK45' #choices: 'Euler' or 'Heun' or 'RK45' (best is RK45)
# computing free energy over time
fe_time=True
# first figure number (counter)
figno=1
# scaling of free energy gradient in gradient descent (just for testing purposes)
beta=1
# figure titles
if Hess == True: printHess=meth_Hess
else: printHess = False
title=f'Loc lin: {loc_lin}, Hessian: {printHess}, Data embedding: {meth_data}, Integration: {meth_int}'

'''The state space model: OU process with additive smooth Gaussian noise'''

'Part 0a: Specifying the flow of the OU process'''

dim_x =2 #State space dimension

alpha = 1 #Scaling of solenoidal flow
Q = np.array([0, 1, -1, 0]).reshape([dim_x, dim_x]) #solenoidal flow matrix
B= -(np.eye(dim_x)+alpha*Q) #drift matrix of OU process for a steady state distribution N(0,I)

def flow(x): #numpy form of flow
    return B @ x

#symbolic time variable
t = sp.Symbol("t")

#symbolic space variable
x= sp.Matrix([sp.Function("x0")(t),
              sp.Function("x1")(t)])

F=B @ x #symbolic flow

'Part 0b: Specifying the kernel of the noise process'

hw = sp.symbols('h')
betaw=1

Kw = sp.sqrt(betaw / (2 * sp.pi)) * sp.exp(-betaw / 2 * hw * hw) # NORMALISED Gaussian kernel given as a symbolic expression

'''The observation model:'''

'Part 1a: the observation function g'

#Dimensionality of observations
dim_y= dim_x

#numpy form of the observation function
def g_obs(x):
    # return x
    # return -x
    # return x**2
    # return np.exp(-x)
    return np.tanh(x)

#sympy form of the observation function
# g= x
# g= -x
# g=x.applyfunc(lambda e: e**2)
# g=x.applyfunc(lambda e: sp.exp(-e))
g=x.applyfunc(sp.tanh)

'Part 1b: the observation noise z'

hz = sp.symbols('h')
betaz=1

#kernel of the likelihood noise
Kz = sp.sqrt(betaz / (2 * sp.pi)) * sp.exp(-betaz / 2 * hz **2) # NORMALISED Gaussian kernel given as a symbolic expression

'''The generative process'''

'Parameters of integration'

N = 2 #30 #number of sample paths

x0 = 10*np.ones([dim_x, N]) #initial condition

T= 10#2.8
timesteps=T*100#28*10**1)
Time = np.linspace(0,T,timesteps) #time grid over which we integrate
dt=Time[1]-Time[0]


'Part 2a: generate sample trajectories from state space model'

epsilon_w = 1 # additive scaling of State space noise
#Sampling white noise convolved with a Gaussian kernel
wt_conv = epsilon_w*convolving_white_noise.white_noise_conv_Gaussian_kernel_nd(dim_x,Time, N, betaw)

#Euler integration with the above noise samples
xt_Euler_conv= Euler.Euler_integration(x0=x0,f=flow, wt=wt_conv,Time=Time)

'Part 2b: Generates sample trajectories from observation model'

#Pass state space time series through the observation function
gxt= np.empty(xt_Euler_conv.shape)
for n in range(N):
    for tau in range(timesteps):
        gxt[:,tau,n]=g_obs(xt_Euler_conv[:,tau,n])

epsilon_z = 0.1 # additive scaling of Observation noise
#Sampling observation noise: white noise convolved with a Gaussian kernel
zt_conv = epsilon_z*convolving_white_noise.white_noise_conv_Gaussian_kernel_nd(dim_y,Time, N,betaz)

#add Observation noise to the time series, to obtain observation time series
yt=gxt+zt_conv

'''The generative model: given as an energy function (i.e. negative log probability)'''

'Part 3a: Specifying the prior Parameters'

order_x=6 #Number of orders of motion on the prior generative model

'Part 3b: Specifying The likelihood Parameters'

order_y=1 #Number of orders of motion (I.e. number of derivatives) on the likelihood generative model
if order_y >order_x: # This has to be lesser or equal than order_x
    raise TypeError('Orders of motion')

# Observation variable: one dimensional. Need to be careful that the dimensionality here coincides with dim_y
y= sp.Matrix([sp.Function("y0")(t),
              sp.Function("y1")(t)])

'Part 3c: The total energy of the generative model'

generative_energy=gen_mod.genmod_energy(F,x,t,order_x,Kw,hw,g,y,order_y,Kz,hz,epsilon_w,epsilon_z,lin=loc_lin,trun=True)

'''The variational free energy with optimal covariance under the Laplace approximation'''
#This is a function of the generalized mean tilde mu

'Definition of the variables in play'

mu= sp.Matrix([sp.Function("mu0")(t),
              sp.Function("mu1")(t)]) # It is absolutely crucial here that the dimensionality of the latter coincides with the dimensionality of the state space
# That is dim_x

# This is the generalized mean
genmu= gen_coords.serial_derivs_xt(mu, t, order_x + 1)  # shape [dim_x,order+1]

'Replacement of genx/tildex in the generative energy by genmu/tildemu'

# Symbolic vector of generalized coordinates for the states
genx = gen_coords.serial_derivs_xt(x, t, order_x + 1)  # shape [dim_x,order+1]
# This is the generative energyEvaluatedAt the generalized meanAnd the generalized data
generative_energy_genmu = gen_coords.gen_coord_replace(generative_energy, genx, genmu)
# Note that since we truncated the last order of motion in the Lagrangian,
# the last order of motion of x or mu is not present there,
# However it will come up in the D term when we do the generalized gradient descent later

if Hess:
    print('===log det Hess computation===')
    log_det_Hess= gen_mod.log_det_hess(generative_energy_genmu,genmu,meth_Hess=meth_Hess)
else:
    log_det_Hess=sp.Matrix([0])
print('===Free energy computation===')
FE_laplace = gen_mod.free_energy_laplace(generative_energy_genmu,log_det_Hess)

'''Free energy gradients'''
print('===Free energy gradients===')
grad_FE=gen_mod.grad_free_energy_laplace(FE_laplace,generative_energy_genmu,genmu,lin=not Hess)

'''Flow of generalised gradient on free energy'''

flow_gen_gd_FE = gen_descent_Lag.flow_gengd_Lag(beta*grad_FE, genmu)

'''Data embedding in generalised coordinates'''
print('===Data embedding===')
yt_embedded = data_embedding.embed_data(yt, Time, order_y,meth_data)

'''Generalised filtering'''
# symbolic observation variable in generalised coordinates
geny = gen_coords.serial_derivs_xt(y, t, order_y + 1)  # symbolic shape [dim_y,order_y+1]

# Generalised filtering
print(f'====Generalised filtering with {meth_int}====')
genmut, Time_gf= \
    gen_filt.integrator_gen_filt_N_samples(flow_gen_gd_FE,Time,yt_embedded,geny,genmu,methint=meth_int,tol=1e-2)

'''Visualisation'''

'Parameters of visualisation'
# sample to plot
n=0
# time to plot
plot_indices=range(0,Time.size)
# plot aesthetics
lw=1
alpha=0.3

'Output of generalised filtering (without confidence bars)'

plt.figure(figno-1)
plt.clf()
plt.suptitle(f'2D Coloured OU process', fontsize=16)
plt.title(title, fontsize=12)
plt.plot(xt_Euler_conv[0,plot_indices,n],xt_Euler_conv[1,plot_indices,n],alpha=alpha,label='Latent')
plt.plot(yt[0,plot_indices,n],yt[1,plot_indices,n],linestyle=':',alpha=alpha,label='Observation')
plt.plot(genmut[n][0,0,:],genmut[n][1,0,:],alpha=alpha,label='Inference')
plt.legend()

plt.figure(figno)
plt.clf()
plt.suptitle(f'2D Coloured OU process', fontsize=16)
plt.title(title, fontsize=12)
for d in range(dim_x):
    plt.plot(Time[plot_indices],xt_Euler_conv[d,plot_indices,n],lw=lw,label='Latent')
    plt.plot(Time[plot_indices],yt[d,plot_indices,n],linestyle=':',lw=lw,label='Observation')
    plt.plot(Time_gf[n], genmut[n][d, 0, :],lw=lw,label='Inference')
plt.legend()

'''Covariance dynamics'''

if cov:
    Hess_gen_energy=gen_coords.gen_Hessian(generative_energy_genmu,genmu)
    gencovt = gen_filt_eval.gen_filt_cov_N_samples_order0(Hess_gen_energy, Time_gf,genmut,genmu,Time,yt_embedded,geny)
    genstd = gen_filt_eval.meaningful_std(gencovt) #standard deviation

    plt.figure(figno+1)
    plt.clf()
    plt.suptitle(f'Inference with uncertainty', fontsize=16)
    plt.title(title, fontsize=12)
    for d in range(dim_x):
        plt.plot(Time[plot_indices], xt_Euler_conv[d, plot_indices, n], label='Latent')
        plt.plot(Time[plot_indices], yt[d, plot_indices, n], linestyle=':', label='Observation')
        plt.plot(Time_gf[n], genmut[n][d, 0, :], label='Inference')
        plt.fill_between(Time_gf[n],genmut[n][d,0,:]+genstd[n][d,d,:], genmut[n][d,0,:]-genstd[n][d,d,:], color='b', alpha=0.15)
    plt.ylim([-1,1.5])
    plt.legend()

    'Plot of covariance'

    plt.figure(figno + 2)
    plt.clf()
    plt.suptitle(f'Covariance value', fontsize=16)
    plt.title(title, fontsize=12)
    plt.plot(Time_gf[n], np.zeros(Time_gf[n].shape), label='Zero',lw=0.2)
    for d in range(dim_x):
        plt.plot(Time_gf[n], gencovt[n][d, d, :], label=f'Covariance {d}',alpha=alpha)
    plt.legend()

'''Free energy dynamics'''
# free energy and subcomponents over time

if fe_time:
    'Free energy over time'
    FEt =  gen_filt_eval.F_over_t(FE_laplace,Time_gf,genmut,genmu,Time,yt_embedded,geny)

    plt.figure(figno+3)
    plt.clf()
    plt.suptitle(f'Free energy', fontsize=16)
    plt.title(title, fontsize=12)
    colourline.plot_cool(Time_gf[n], FEt[n], lw=1, alpha=alpha)
    # plt.ylim([-10, 8])

    'Free energy gradient norm'

    grad_FE_norm=grad_FE.norm(1)
    grad_FE_norm_t = gen_filt_eval.F_over_t(grad_FE_norm, Time_gf, genmut, genmu, Time, yt_embedded, geny)
    flow_gen_gd_FE_norm=flow_gen_gd_FE.norm(1)
    flow_gen_gd_FE_norm_t = gen_filt_eval.F_over_t(flow_gen_gd_FE_norm, Time_gf, genmut, genmu, Time, yt_embedded, geny)

    plt.figure(figno + 4)
    plt.clf()
    plt.suptitle(f'Free energy gradient norm', fontsize=16)
    plt.title(title, fontsize=12)
    plt.plot(Time_gf[n], grad_FE_norm_t[n], label='Grad F norm',alpha=0.1)
    plt.plot(Time_gf[n], flow_gen_gd_FE_norm_t[n], label='D-Grad F norm',alpha=0.1)
    res=flow_gen_gd_FE_norm_t[n]-grad_FE_norm_t[n]
    # plt.plot(Time_gf[n], res, label='res',alpha=0.1)
    plt.legend()

    'Step size'
    plt.figure(figno + 6)
    plt.clf()
    plt.suptitle(f'Step size', fontsize=16)
    plt.title(title, fontsize=12)
    plt.plot(Time_gf[n][:-1], Time_gf[n][1:] - Time_gf[n][:-1], label='step size')

    'Performance statistics'

    print(f'Free action={np.mean(FEt[n])}')
    print(f'Free action (omitting nans)={np.nanmean(FEt[n])}')

    try:
        print(f'MSE={np.sum((xt_Euler_conv[0,:,n]-genmut[n][0,0,:])**2)/Time.size}')
    except Exception:
        pass

if fe_time and Hess:

    'Free energy components over time'
    gen_energy_t =  gen_filt_eval.F_over_t(generative_energy_genmu,Time_gf,genmut,genmu,Time,yt_embedded,geny)
    log_det_Hess_t=gen_filt_eval.F_over_t(log_det_Hess,Time_gf,genmut,genmu,Time,yt_embedded,geny)

    plt.figure(figno+4)
    plt.clf()
    plt.suptitle(f'Free energy components', fontsize=16)
    plt.title(title, fontsize=12)
    plt.plot(Time_gf[n], FEt[n], lw=1, alpha=alpha, label='FE')
    plt.plot(Time_gf[n], gen_energy_t[n],lw=1, alpha=alpha, label='gen_energy')
    plt.plot(Time_gf[n], log_det_Hess_t[n], lw=1, alpha=alpha, label='log_det_hess')
    plt.legend()

    'Determinant of Hessian'
    Hess_gen_energy = gen_coords.gen_Hessian(generative_energy_genmu, genmu)
    det_Hess_GE = log_det_Hess.applyfunc(sp.exp)
    det_Hess_t = gen_filt_eval.F_over_t(det_Hess_GE, Time_gf, genmut, genmu, Time, yt_embedded, geny)

    plt.figure(figno + 5)
    plt.clf()
    plt.suptitle(f'det Hess', fontsize=16)
    plt.title(title, fontsize=12)
    plt.plot(Time_gf[n], det_Hess_t[n], label='det Hess')
    plt.legend()

