'''Here we do gen filt of the one dimensional OU process with smooth noise
and a non-linear observation function'''

import numpy as np
import sympy as sp
from Routines import colourline,gen_coords, convolving_white_noise,gen_mod, gen_filt_eval,symbolic_algebra,data_embedding
import matplotlib.pyplot as plt
from integration import Euler, gen_descent_Lag, gen_filt

'''Parameters of the method'''

# seed
np.random.seed(0)
# linearisation method
meth_lin='local' #'local' or 'hybrid' or 'none'
# 'local' linearises the flows through the local linear approximation and ignores the Hessian term in the free energy gradient
# 'hybrid' does not linearise the flows but ignores the Hessian term in the free energy gradient
# 'none' is the full non-linearised method
# choice of data assimilation and augmentation method
meth_data = 'Taylor' #choices: 'findiff' or 'Taylor'
# computing covariance, ie uncertainty of inference
cov=True
# integration method
meth_int= 'RK45' #choices: 'Euler' or 'Heun' or 'RK45' (best is RK45)
# computing free energy over time
fe_time=True
# first figure number
figno=1
# scaling of free energy gradient in gradient descent (just for testing purposes)
beta=1

'''The state space model: OU process with additive smooth Gaussian noise'''

'Part 0a: Specifying the flow of the OU process'''

dim_x =1 #State space dimension

B= -1 #drift matrix of OU process for a steady state distribution N(0,I)

def flow(x): #numpy form of flow
    return B * x

#symbolic time variable
t = sp.Symbol("t")

x= sp.Matrix([sp.Function("x0")(t)])

F=B * x #symbolic flow

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

T= 20#2.8
timesteps=T*10#28*10**1)
Time = np.linspace(0,T,timesteps) #time grid over which we integrate
dt=Time[1]-Time[0]


'Part 2a: generate sample trajectories from state space model'

epsilon_w = 1 # scaling of State space noise
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

epsilon_z = 0.1 # scaling of Observation noise
#Sampling observation noise: white noise convolved with a Gaussian kernel
zt_conv = epsilon_z*convolving_white_noise.white_noise_conv_Gaussian_kernel_nd(dim_y,Time, N,betaz)

#add Observation noise to the time series, to obtain observation time series
yt=gxt+zt_conv

'''The generative model: given as an energy function (i.e. negative log probability)'''

'Part 3a: Specifying the prior Parameters'

order_x=6 #Number of orders of motion on the prior generative model

'Part 3b: Specifying The likelihood Parameters'

order_y=1 #Number of orders of motion (I.e. number of derivatives) on the likelihood generative model

# Observation variable: one dimensional. Need to be careful that the dimensionality here coincides with dim_y
y= sp.Matrix([sp.Function("y0")(t)])

'Part 3c: The total energy of the generative model'

if meth_lin=='hybrid' or meth_lin=='none': #'local' or 'hybrid' or 'none'
    lin_flow = False
elif meth_lin=='local':
    lin_flow = True

generative_energy=gen_mod.genmod_energy(F,x,t,order_x,Kw,hw,g,y,order_y,Kz,hz,epsilon_w,epsilon_z,lin=lin_flow,trun=True)

'''The variational free energy with optimal covariance under the Laplace approximation'''
#This is a function of the generalized mean tilde mu

'Definition of the variables in play'

mu= sp.Matrix([sp.Function("mu0")(t)])
# It is absolutely crucial here that the dimensionality of the latter coincides with the dimensionality of the state space
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

log_det_Hess_GE =gen_mod.log_det_hess(generative_energy_genmu,genmu)
FE_laplace= gen_mod.free_energy_laplace(generative_energy_genmu,log_det_Hess_GE)

'''Free energy gradients'''

if meth_lin=='hybrid' or meth_lin=='local': #'local' or 'hybrid' or 'none'
    ignore_Hessian_in_grad = True
elif meth_lin=='none':
    ignore_Hessian_in_grad = False

grad_FE=gen_mod.grad_free_energy_laplace(FE_laplace,generative_energy_genmu,genmu,lin=ignore_Hessian_in_grad)

'''Flow of generalised gradient on free energy'''

flow_gen_gd_FE = gen_descent_Lag.flow_gengd_Lag(beta*grad_FE, genmu)

'''Data embedding in generalised coordinates'''

yt_embedded = data_embedding.embed_data(yt, Time, order_y,meth_data)

'''Generalised filtering'''

geny = gen_coords.serial_derivs_xt(y, t, order_y + 1)  # shape [dim_x,order+1]

# Generalised filtering
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
plt.figure(figno)
plt.clf()
plt.suptitle(f'1D Coloured OU process', fontsize=16)
plt.title(f'Linearisation: {meth_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
plt.plot(Time[plot_indices],xt_Euler_conv[0,plot_indices,n],label='Latent')
plt.plot(Time[plot_indices],yt[0,plot_indices,n],linestyle=':',label='Observation')
plt.plot(Time_gf[n],genmut[n][0,0,:],label='Inference')
# plt.plot(Time[plot_indices],mut_nonlinear,label='Inference')
# colourline.plot_cool(Time[plot_indices],mut_order0[0,plot_indices,n], lw=1,alpha=alpha)
# plt.ylim([-1,2])
plt.legend()

'''Covariance dynamics'''

if cov:
    Hess_gen_energy=gen_coords.gen_Hessian(generative_energy_genmu,genmu)
    gencovt = gen_filt_eval.gen_filt_cov_N_samples_order0(Hess_gen_energy, Time_gf,genmut,genmu,Time,yt_embedded,geny)
    genstd = gen_filt_eval.meaningful_std(gencovt) #standard deviation

    plt.figure(figno+6)
    plt.clf()
    plt.suptitle(f'Inference with uncertainty', fontsize=16)
    plt.title(f'Linearisation: {meth_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
    plt.plot(Time[plot_indices], xt_Euler_conv[0, plot_indices, n], label='Latent')
    plt.plot(Time[plot_indices], yt[0, plot_indices, n], linestyle=':', label='Observation')
    plt.plot(Time_gf[n],genmut[n][0,0,:], label='Inference')
    plt.fill_between(Time_gf[n],genmut[n][0,0,:]+genstd[n][0,0,:], genmut[n][0,0,:]-genstd[n][0,0,:], color='b', alpha=0.15)
    plt.ylim([-0.7,2])
    plt.legend()

    'Plot of covariance'

    # Time_gf[n].size
    # np.max(gencovt[n][0, 0, :])
    # np.mean(gencovt[n][0, 0, :]) #negative
    # np.nanmax(gencovt[n][0, 0, :])
    # np.nanmean(gencovt[n][0, 0, :])

    plt.figure(figno + 7)
    plt.clf()
    plt.suptitle(f'Covariance value', fontsize=16)
    plt.title(f'Linearisation: {meth_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
    plt.plot(Time_gf[n], gencovt[n][0, 0, :], label='Covariance')
    plt.plot(Time_gf[n], np.zeros(Time_gf[n].shape), label='Zero',lw=0.2)
    plt.legend()

'''Free energy dynamics'''
# free energy and subcomponents over time

if fe_time:
    'Free energy over time'
    FEt =  gen_filt_eval.F_over_t(FE_laplace,Time_gf,genmut,genmu,Time,yt_embedded,geny)

    plt.figure(figno+1)
    plt.clf()
    plt.suptitle(f'Free energy', fontsize=16)
    plt.title(f'Linearisation: {meth_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
    plt.plot(Time[plot_indices], xt_Euler_conv[0, plot_indices, n], label='Latent')
    plt.plot(Time[plot_indices], yt[0, plot_indices, n], linestyle=':', label='Observation')
    plt.plot(Time_gf[n],genmut[n][0,0,:], label='Inference')
    colourline.plot_cool(Time_gf[n], FEt[n], lw=lw, alpha=alpha)
    # plt.ylim([-10, 8])

    'Free energy components over time'
    gen_energy_t =  gen_filt_eval.F_over_t(generative_energy_genmu,Time_gf,genmut,genmu,Time,yt_embedded,geny)
    log_det_Hess_t=gen_filt_eval.F_over_t(log_det_Hess_GE,Time_gf,genmut,genmu,Time,yt_embedded,geny)

    plt.figure(figno+2)
    plt.clf()
    plt.suptitle(f'Free energy components', fontsize=16)
    plt.title(f'Linearisation: {meth_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
    # plt.plot(Time[plot_indices], xt_Euler_conv[0, plot_indices, n], label='Latent')
    # plt.plot(Time[plot_indices], yt[0, plot_indices, n], linestyle=':', label='Observation')
    # plt.plot(Time[plot_indices], mut_order0[0, plot_indices, n], label='Inference')
    plt.plot(Time_gf[n], FEt[n], lw=lw, alpha=alpha, label='FE')
    plt.plot(Time_gf[n], gen_energy_t[n], lw=lw, alpha=alpha, label='gen_energy')
    plt.plot(Time_gf[n], log_det_Hess_t[ n], lw=lw, alpha=alpha, label='log_det_hess')
    # plt.plot(Time_gf[n], (gen_energy_t+log_det_Hess_t)[plot_indices, n], lw=lw, alpha=alpha, label='sum')
    plt.legend()

    'Determinant of Hessian'
    Hess_gen_energy = gen_coords.gen_Hessian(generative_energy_genmu, genmu)
    det_Hess_GE = symbolic_algebra.sympy_det(Hess_gen_energy)
    det_Hess_t = gen_filt_eval.F_over_t(det_Hess_GE, Time_gf, genmut, genmu, Time, yt_embedded, geny)
    np.sum(det_Hess_t[n][plot_indices] <= 0)

    plt.figure(figno + 3)
    plt.clf()
    plt.suptitle(f'det Hess', fontsize=16)
    plt.title(f'Linearisation: {meth_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
    plt.plot(Time_gf[n], det_Hess_t[n], label='det Hess')
    plt.legend()

    'Free energy gradient norm'

    grad_FE_norm=grad_FE.norm(1)
    grad_FE_norm_t = gen_filt_eval.F_over_t(grad_FE_norm, Time_gf, genmut, genmu, Time, yt_embedded, geny)
    flow_gen_gd_FE_norm=flow_gen_gd_FE.norm(1)
    flow_gen_gd_FE_norm_t = gen_filt_eval.F_over_t(flow_gen_gd_FE_norm, Time_gf, genmut, genmu, Time, yt_embedded, geny)

    plt.figure(figno + 4)
    plt.clf()
    plt.suptitle(f'Free energy gradient norm', fontsize=16)
    plt.title(f'Linearisation: {meth_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
    plt.plot(Time_gf[n], grad_FE_norm_t[n], label='Grad F norm',alpha=0.1)
    plt.plot(Time_gf[n], flow_gen_gd_FE_norm_t[n], label='D-Grad F norm',alpha=0.1)
    res=flow_gen_gd_FE_norm_t[n]-grad_FE_norm_t[n]
    # plt.plot(Time_gf[n], res, label='res',alpha=0.1)
    plt.legend()

    'Step size'
    plt.figure(figno + 5)
    plt.clf()
    plt.suptitle(f'Step size', fontsize=16)
    plt.title(f'Linearisation: {meth_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
    plt.plot(Time_gf[n][:-1], Time_gf[n][1:] - Time_gf[n][:-1], label='step size')

    'Performance statistics'

    print(f'Free action={np.mean(FEt[n])}')
    print(f'Free action (omitting nans)={np.nanmean(FEt[n])}')

    try:
        print(f'MSE={np.sum((xt_Euler_conv[0,:,n]-genmut[n][0,0,:])**2)/Time.size}')
    except Exception:
        pass

