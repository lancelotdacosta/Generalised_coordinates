'''this script is about generalized filtering of the [Noisily observed]
Lotka Volterra process with smooth noise'''

import numpy as np
import sympy as sp
from Routines import Taylor_series,sampling_generalised_noise,colourline,gen_coords, convolving_white_noise,gen_mod
import matplotlib.pyplot as plt
from integration import Euler, gen_descent_Lag, lin_zigzag
from scipy.linalg import expm
import time


'''The state space model:  Lotka Volterra process with additive smooth Gaussian noise'''

'Part 0a: Specifying the flow of the Lotka Volterra process'''

dim_x =2 #State space dimension

#parameters of Lotka Volterra system
a = 1
b =1
c =1
d = 1
#only a/d affects the nature of the solution up to rescaling of space and time

# Specifying the numpy form of the flow
def flow(x):
    y=np.empty(x.shape)
    y[0,:]= a*x[0,:]-b*x[0,:]*x[1,:] #rate of change of prey
    y[1, :] = c*x[0,:]*x[1,:]-d*x[1,:] #rate of change of predators
    return y

#symbolic time variable
t = sp.Symbol("t")

#symbolic space variable
x= sp.Matrix([sp.Function("x0")(t),
              sp.Function("x1")(t)])

#symbolic Flow of Lotka Volterra system
F=sp.Matrix([a*x[0]-b*x[0]*x[1],
             c*x[0]*x[1]-d*x[1]])


'Part 0b: Specifying the kernel of the noise process'

hw = sp.symbols('h')
betaw=1

Kw = sp.sqrt(betaw / (2 * sp.pi)) * sp.exp(-betaw / 2 * hw * hw) # NORMALISED Gaussian kernel given as a symbolic expression


'''The observation model:'''

'Part 1a: the observation function g'

#Dimensionality of observations
dim_y= dim_x


#sympy form of the observation function
Q = np.eye(dim_y)#np.array([0, 1, -1, 0]).reshape([dim_y, dim_x])
g= Q@x#x.applyfunc(sp.tanh)

#numpy form of the observation function
def g_obs(x):
    return Q@x #np.tanh(x)

'Part 1b: the observation noise z'

hz = sp.symbols('h')
betaz=1

#kernel of the likelihood noise
Kz = sp.sqrt(betaz / (2 * sp.pi)) * sp.exp(-betaz / 2 * hz **2) # NORMALISED Gaussian kernel given as a symbolic expression


'''The generative process'''

'Parameters of integration'

N = 1 #30 #number of sample paths

x0 = 1.5*np.ones([dim_x, N]) #initial condition

T= 6
timesteps=5*10**4
Time = np.linspace(0,T,timesteps) #time grid over which we integrate
dt=Time[1]-Time[0]

'Part 2a: generate sample trajectory from state space model'

epsilon_w = 0.1 # scaling of State space noise
#Sampling white noise convolved with a Gaussian kernel
wt_conv = epsilon_w*convolving_white_noise.white_noise_conv_Gaussian_kernel_nd(dim_x,Time, N, betaw)

#Euler integration with the above noise samples
xt_Euler_conv= Euler.Euler_integration(x0=x0,f=flow, wt=wt_conv,Time=Time)

'Part 2b: Generates sample trajectory from observation model'

#Pass state space time series through the observation function
gxt= np.empty(xt_Euler_conv.shape)
for n in range(N):
    for tau in range(timesteps):
        gxt[:,tau,n]=g_obs(xt_Euler_conv[:,tau,n])

epsilon_z = 0.5 # scaling of Observation noise
#Sampling observation noise: white noise convolved with a Gaussian kernel
zt_conv = epsilon_z*convolving_white_noise.white_noise_conv_Gaussian_kernel_nd(dim_y,Time, N,betaz)

#add Observation noise to the time series, to obtain observation time series
yt=gxt+zt_conv


'''The generative model: specified in terms of energy functions (i.e. negative probabilities)'''


'Part 3a: Specifying The prior Parameters'

order_x=4 #Number of orders of motion on the prior generative model


'Part 3b: Specifying The likelihood Parameters'

order_y=3 #Number of orders of motion (I.e. number of derivatives) on the likelihood generative model


# Observation variable: two dimensional. Need to be careful that the dimensionality here coincides with dim_y
y= sp.Matrix([sp.Function("y0")(t),
              sp.Function("y1")(t)])


'Part 3c: The total energy of the generative model'
lin=True
generative_energy=gen_mod.genmod_energy(F,x,t,order_x,Kw,hw,g,y,order_y,Kz,hz,epsilon_w,epsilon_z,lin,trun=True)


'''The variational free energy with optimal covariance under the Laplace approximation'''
#This is a function of the generalized mean tilde mu

'Definition of the variables in play'

mu= sp.Matrix([sp.Function("mu0")(t),
              sp.Function("mu1")(t)])
# It is absolutely crucial here that the dimensionality of the latter coincides with the dimensionality of the state space
# That is dim_x

# This is the generalized mean
genmu= gen_coords.serial_derivs_xt(mu, t, order_x + 1)  # shape [dim_x,order+1]

#Symbolic vector of generalized coordinates for the states
genx = gen_coords.serial_derivs_xt(x, t, order_x + 1)  # shape [dim_x,order+1]

'Replacement of genx/tildex in the generative energy by genmu/tildemu'

# This is the generative energyEvaluatedAt the generalized meanAnd the generalized data
generative_energy_genmu=gen_coords.gen_coord_replace(generative_energy,genx,genmu)
# Note that since we truncated the last order of motion in the Lagrangian,
# the last order of motion of x or mu is not present there,
# However it will come up in the D term when we do the generalized gradient descent later


'Log determinant term of the Hessian'

# First compute Hessian of generative Energy
Hess_gen_energy = gen_coords.gen_Hessian(generative_energy_genmu,genmu)


if not lin:
    # compute log determinant
    log_det_Hess_GE = sp.Matrix([sp.log(Hess_gen_energy.det()) / 2])

    'Free energy with optimal covariance under the Laplace approximation'

    FE = generative_energy_genmu + log_det_Hess_GE
else:
    #Approximation which is not theoretically grounded but which speeds up Things a great deal
    # this is by ignoring the hessian term in the free energy

    'Free energy with optimal covariance under the Laplace approximation'

    FE = generative_energy_genmu




'''Free energy gradients'''

'Gradient of generative energy'
grad_GE = gen_coords.gen_gradient(generative_energy_genmu, genmu)

'Gradient of log determinant Hessian term'
if not lin:
    grad_log_det_Hess_GE=gen_coords.gen_gradient(log_det_Hess_GE, genmu)

    'Free energy gradient'
    grad_FE = grad_GE + grad_log_det_Hess_GE
else:

    'Free energy gradient'
    grad_FE = grad_GE


'''Flow of generalised gradient on free energy'''

flow_gen_gd_FE = gen_descent_Lag.flow_gengd_Lag(grad_FE, genmu)


'''Data embedding in generalised coordinates'''


'Method 1: finite differences'
yt_embedded=np.zeros([dim_y,order_y+1,timesteps,N])

yt_embedded[:,0,:,:]=yt

for o in range(1,order_y+1):
    for tau in range(1,timesteps):
        yt_embedded[:, o, tau,  :] = (yt_embedded[:, o-1, tau,  :]-yt_embedded[:, o-1, tau-1,  :])/(Time[tau]-Time[tau-1])

'Method 2: inverting Taylor expansion'

Taylor=False #indicates whether we do the other method or not

#Build taylor expansion coefficients
def Taylor_coeffs(dt,order_y):
    coeffs=np.ones([order_y+1])
    for o in range(1,order_y+1):
        coeffs[o]=coeffs[o-1]*dt/o
    return coeffs

 # build taylor expansion matrix
def Taylor_matrix(dt, order_y):
    T_mx = np.empty([order_y + 1,order_y + 1])
    for o in range(order_y + 1):
        T_mx[o] = Taylor_coeffs(dt*(order_y-o),order_y)
    return T_mx


if Taylor:
    T_mx=Taylor_matrix(-dt, order_y)
    inv_T_mx=np.linalg.inv(T_mx)

    yt_embedded2=np.zeros([dim_y,order_y+1,timesteps,N])

    for tau in range(timesteps):
        for d in range(dim_y):
                if tau < order_y:
                    temp=np.tile(yt[d,0,:],order_y-tau).reshape([order_y-tau,N])
                    hist = np.concatenate([temp,yt[d, :tau + 1, :]])
                else:
                    hist= yt[d,tau-order_y:tau+1,:]
                yt_embedded2[d, :, tau, :] = inv_T_mx @ hist

    # test it works: the following should be small
    # print(np.sum(np.abs(yt_embedded2[:,0,:,:]-yt)))

    # test difference between embedding methods
    print(np.sum(np.abs(yt_embedded2-yt_embedded)))


'''Generalised filtering'''

# genFlow=flow_gen_gd_FE
# yt_embedded=yt_embedded[:,:,:,0] #TESTING FOR ONE SAMPLE FIRST

def Euler_gen_filt(genFlow,Time,yt_embedded,geny,genmu):
    # method developed for filtering ONE sample path only
    print('=Euler integration for generalised filtering=')

    [dim_x,order_x_pone]=genmu.shape
    [dim_y,order_y_pone,T]=yt_embedded.shape

    'Initialise means'
    genmut = np.zeros([dim_x, order_x_pone, T])

    'setup function'
    input_sympy= genmu.row_join(geny)
    genF_lambdify = sp.lambdify(input_sympy, genFlow, "numpy")

    for t in range(1,T):
        if t % 10**5 == 0:
             print(t, '/', T)
        step=Time[t]-Time[t-1]
        input_numpy = np.concatenate((genmut[:,:,t-1],yt_embedded[:,:,t-1]),axis=1)
        genmut[:,:,t]=genmut[:,:,t-1] + step * genF_lambdify(*input_numpy.ravel())
    return genmut

def Euler_adaptive_gen_filt(genFlow,Time,yt_embedded,geny,genmu):
    # method developed for filtering ONE sample path only
    print('=Euler integration for generalised filtering=')

    [dim_x,order_x_pone]=genmu.shape
    [dim_y,order_y_pone,T]=yt_embedded.shape

    'Initialise means'
    genmut = np.zeros([dim_x, order_x_pone, T])

    'setup function'
    input_sympy= genmu.row_join(geny)
    genF_lambdify = sp.lambdify(input_sympy, genFlow, "numpy")

    'parameters for adaptive step size'
    time=0 #continuous time
    prev_t=0
    t=0
    iters=0
    tol=10*Time.size
    genmu_inst=genmut[:,:,0] #instantaneous genmu
    genmu_inst_prev = genmut[:, :, 0]
    while time < Time[T]:
        # figure out between which indices we are in the time interval (t<=time<t+1)
        # also figures out whether we just went past a t (indicated by 'update')
        indices=Time<=time
        new_t=np.sum(indices)-1
        if new_t>t:
            update = True
        elif new_t==t:
            update = False
        else:
            raise TypeError('update')
        t = new_t
        # prepare the input for the flow
        input_numpy = np.concatenate((genmu_inst,yt_embedded[:,:,t]),axis=1)
        # flow output
        F_output=genF_lambdify(*input_numpy.ravel())
        # define time step size
        step = (Time[t+1] - Time[t])/np.sum(np.abs(F_output))
        # update
        genmu_inst = genmu_inst_prev + step * F_output
        # if need to update genmut
        if update:
            for i in range(prev_t,t):
                genmut[:, :, t - 1]=genmu_inst_prev
        'TODO THIS NEEDS A RETHINK'


    for t in range(1,T):
        step = Time[t] - Time[t - 1]
        input_numpy = np.concatenate((genmut[:,:,t-1],yt_embedded[:,:,t-1]),axis=1)
        genmut[:,:,t]=genmut[:,:,t-1] + step * genF_lambdify(*input_numpy.ravel())
    return genmut



def integrator_gen_filt_N_samples(genFlow,Time,yt_embedded,geny,genmu,methint='Euler'):
    print('====Euler integration for generalised filtering (N samples)====')

    [dim_x,order_x_pone]=genmu.shape
    [dim_y,order_y_pone,T,N]=yt_embedded.shape

    'Initialise means'
    genmut = np.empty([dim_x, order_x_pone, T,N])

    'Do generalised filtering for each sample'
    for n in range(N):
        print(f'---Sample {n}---')
        if methint== 'Euler':
            genmut[:,:,:,n]=Euler_gen_filt(genFlow,Time,yt_embedded[:,:,:,n],geny,genmu)
        elif methint=='Euler_adaptive':
            genmut[:, :, :, n] = Euler_adaptive_gen_filt(genFlow, Time, yt_embedded[:, :, :, n], geny, genmu)
        else: # If an exact match is not confirmed, this last case will be used if provided
            raise TypeError('integrator not supported')
        'THIS IS INTRODUCED TO NOT COMPUTE SAMPLE 2 TO MAKE THINGS FASTER. CAN REMOVE THIS'
        if N==2: break

    return genmut


geny = gen_coords.serial_derivs_xt(y, t, order_y + 1)  # shape [dim_x,order+1]


# 'DEBUG MODE'
#
# # genmutest=Euler_gen_filt(flow_gen_gd_FE,Time,yt_embedded[:,:,:,n],geny,genmu)
#
# 'END DEBUG MODE'

# Euler generalised filtering
genmut= integrator_gen_filt_N_samples(flow_gen_gd_FE,Time,yt_embedded,geny,genmu)
mut_order0=genmut[:,0,:,:]


# with alternative data assimilation method
if Taylor:
    genmut2= integrator_gen_filt_N_samples(flow_gen_gd_FE,Time,yt_embedded2,geny,genmu)
    mut2_order0=genmut2[:,0,:,:]

    print(np.sum(np.abs(mut_order0-mut2_order0)))

'''Covariance dynamics'''

def gen_filt_cov_N_samples(Hess_gen_energy,Time,genmut,yt_embedded,geny,genmu):
    print('====Gen filt cov (N samples)====')

    [dim_x,order_x_pone]=genmu.shape
    [dim_y,order_y_pone,T,N]=yt_embedded.shape

    shape_gencovt_flatenned = list(Hess_gen_energy.shape) + [T,N]
    gencovt_flatenned=np.empty(shape_gencovt_flatenned)

    input_sympy = genmu.row_join(geny)
    Hess_lambdify = sp.lambdify(input_sympy, Hess_gen_energy, "numpy")

    for n in range(N):
        print('sample = ', n)
        for t in range(Time.size):
            if t % 10 ** 5 == 0:
                print(t, '/', T)
            input_numpy = np.concatenate((genmut[:, :, t, n], yt_embedded[:, :, t, n]), axis=1)
            gencovt_flatenned[:,:,t,n]=np.linalg.inv(Hess_lambdify(*input_numpy.ravel()))

    return gencovt_flatenned

def gen_filt_cov_N_samples_order0(Hess_gen_energy,Time,genmut,yt_embedded,geny,genmu):
    # this is only the covariance at order zero for all times and all samples as this is the one we will plot later
    order_x_pone = genmu.shape[1]
    return gen_filt_cov_N_samples(Hess_gen_energy,Time,genmut,yt_embedded,geny,genmu)[:order_x,:order_x,:,:]

cov= False
if cov:
    gencovt_order0 = gen_filt_cov_N_samples_order0(Hess_gen_energy,Time,genmut,yt_embedded,geny,genmu)

if Taylor and cov:
    # with alternative data assimilation method
    gencovt2_order0 = gen_filt_cov_N_samples_order0(Hess_gen_energy,Time,genmut,yt_embedded2,geny,genmu)


'''Free energy dynamics'''
# look at free energy over time

def FEt_N_samples(FE,Time,genmut,yt_embedded,geny,genmu):
    print('====FE over time (N samples)====')

    [dim_y,order_y_pone,T,N]=yt_embedded.shape

    FEt=np.empty([T,N])

    input_sympy = genmu.row_join(geny)
    FE = sp.lambdify(input_sympy, FE, "numpy")

    for n in range(N):
        print('sample = ', n)
        for t in range(Time.size):
            if t % 10 ** 5 == 0:
                print(t, '/', T)
            input_numpy = np.concatenate((genmut[:, :, t, n], yt_embedded[:, :, t, n]), axis=1)
            FEt[t,n]=np.linalg.inv(FE(*input_numpy.ravel()))
        'THIS IS INTRODUCED TO NOT COMPUTE SAMPLE 2 TO MAKE THINGS FASTER. CAN REMOVE THIS'
        if N==2: break

    return FEt

FEt = FEt_N_samples(FE,Time,genmut,yt_embedded,geny,genmu)

if Taylor:
    # with alternative data assimilation method
    FEt2 = FEt_N_samples(FE,Time,genmut,yt_embedded2,geny,genmu)

    # observe lower free energy for taylor expansion assimilation method
    print(np.sum((FEt-FEt2)))



'''Visualisation'''

# sample to plot
n=0

# time to plot
plot_indices=range(10,Time.size)

# plot aesthetics
lw=1
alpha=0.3

plt.figure(1)
plt.clf()
plt.suptitle(f'2D Coloured Lotka Volterra', fontsize=16)
plt.title(f'Data assimilation: finite differences', fontsize=12)
plt.plot(plot_indices, xt_Euler_conv[0,plot_indices,n],label='Prey')
plt.plot(plot_indices, xt_Euler_conv[1,plot_indices,n],label='Predator')
plt.plot(plot_indices, yt[0,plot_indices,n],linestyle=':',label='Observation', lw=lw, c='blue')
plt.plot(plot_indices, yt[1,plot_indices,n],linestyle=':',label='Observation', lw=lw,c='orange')
plt.plot(plot_indices, mut_order0[0,plot_indices,n],label='Prey')
plt.plot(plot_indices, mut_order0[1,plot_indices,n],label='Predator')
plt.legend()
# plt.xlim(right=20, left=-20)
# plt.ylim(top=20,bottom=-40)
plt.savefig(f"LV_GF_g=id_ew={epsilon_w}_ez={epsilon_z}_step={Time[1]-Time[0]}.png", dpi=100)



plt.figure(2)
plt.clf()
plt.suptitle(f'Free energy', fontsize=16)
plt.title(f'Data assimilation: finite differences', fontsize=12)
colourline.plot_cool(Time[plot_indices], FEt[plot_indices,n], lw=lw,alpha=alpha)
plt.savefig(f"LV_GF-FEt_g=id_ew={epsilon_w}_ez={epsilon_z}_step={Time[1]-Time[0]}.png", dpi=100)


'WITH ALTERNATIVE DATA ASSIMILATION METHOD'

if Taylor:

    plt.figure(3)
    plt.clf()
    plt.suptitle(f'2D Coloured OU process gen filt', fontsize=16)
    plt.title(f'Data assimilation: inverse taylor expansion', fontsize=12)
    plt.plot(plot_indices, xt_Euler_conv[0,plot_indices,n],label='Prey')
    plt.plot(plot_indices, xt_Euler_conv[1,plot_indices,n],label='Predator')
    plt.plot(plot_indices, yt[0,plot_indices,n],linestyle=':',label='Observation', lw=lw, c='blue')
    plt.plot(plot_indices, yt[1,plot_indices,n],linestyle=':',label='Observation', lw=lw,c='orange')
    plt.plot(plot_indices, mut2_order0[0,plot_indices,n],label='Prey')
    plt.plot(plot_indices, mut2_order0[1,plot_indices,n],label='Predator')
    plt.legend()
    # plt.xlim(right=20, left=-20)
    # plt.ylim(top=20,bottom=-40)


    plt.figure(4)
    plt.clf()
    plt.suptitle(f'Free energy', fontsize=16)
    plt.title(f'Data assimilation: inverse taylor expansion', fontsize=12)
    colourline.plot_cool(Time[plot_indices], FEt2[plot_indices,n], lw=lw,alpha=alpha)


    'free energy comparisons'

    plt.figure(5)
    plt.clf()
    plt.suptitle(f'Free energy differences', fontsize=16)
    plt.title(f'F(data assimilation)-F(inverse taylor expansion)', fontsize=12)
    colourline.plot_cool(Time[plot_indices], (FEt-FEt2)[plot_indices,n], lw=lw,alpha=alpha)



