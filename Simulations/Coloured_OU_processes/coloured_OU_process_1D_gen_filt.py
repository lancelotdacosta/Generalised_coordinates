'''Here we do gen filt of the one dimensional OU process with smooth noise
and a non linear observation function'''

import numpy as np
import sympy as sp
from Routines import Taylor_series,sampling_generalised_noise,colourline,gen_coords, convolving_white_noise,gen_mod
import matplotlib.pyplot as plt
from integration import Euler, gen_descent_Lag, lin_zigzag
from scipy.linalg import expm
import time

'''Parameters of the method'''

# seed
np.random.seed(0)
# local linearisation
local_lin = False
# choice of data assimilation and augmentation method
meth_data = 'findiff' #choices: 'findiff' or 'Taylor'
# computing covariance, ie uncertainty of inference
cov=False
# integration method
meth_int= 'RK45' #choices: 'Euler' or 'adaptive_Euler' or 'Heun' or 'RK45'
# computing free energy over time
fe_time=True
# figure number
figno=1

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
    return np.tanh(x)

#sympy form of the observation function
# g= x
# g= -x
# g=x.applyfunc(lambda e: e**2)
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

epsilon_z = 1 # scaling of Observation noise
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

generative_energy=gen_mod.genmod_energy(F,x,t,order_x,Kw,hw,g,y,order_y,Kz,hz,epsilon_w,epsilon_z,lin=local_lin,trun=True)


'''The variational free energy with optimal covariance under the Laplace approximation'''
#This is a function of the generalized mean tilde mu

'Definition of the variables in play'

mu= sp.Matrix([sp.Function("mu0")(t)])
# It is absolutely crucial here that the dimensionality of the latter coincides with the dimensionality of the state space
# That is dim_x

# This is the generalized mean
genmu= gen_coords.serial_derivs_xt(mu, t, order_x + 1)  # shape [dim_x,order+1]

# Symbolic vector of generalized coordinates for the states
genx = gen_coords.serial_derivs_xt(x, t, order_x + 1)  # shape [dim_x,order+1]

'Replacement of genx/tildex in the generative energy by genmu/tildemu'

# This is the generative energyEvaluatedAt the generalized meanAnd the generalized data
generative_energy_genmu = gen_coords.gen_coord_replace(generative_energy, genx, genmu)

# Note that since we truncated the last order of motion in the Lagrangian,
# the last order of motion of x or mu is not present there,
# However it will come up in the D term when we do the generalized gradient descent later

def sympy_det(M):
    dim=M.shape[0]
    if dim==1:
        return M#sp.Matrix([M])
    else:
        det_M=sp.Matrix([0])
        col = range(1,dim)
        for r in range(dim):
            # print(r)
            row=[i for i in range(dim) if i != r]
            det_M += (-1)**r * M[r,0] * sympy_det(M[row,col])
        return det_M

def log_det_hess(generative_energy_genmu,genmu):
    'Log determinant term of the Hessian of the generative energy'
    # First compute Hessian of generative Energy
    Hess_gen_energy = gen_coords.gen_Hessian(generative_energy_genmu, genmu)
    # compute log determinant
    det_Hess_GE = sympy_det(Hess_gen_energy)
    log_det_Hess_GE= det_Hess_GE.applyfunc(sp.log)
    return log_det_Hess_GE
def free_energy_laplace(generative_energy_genmu,log_det_Hess_GE):

    '''THIS RETURNS THE FREE ENERGY WITH OPTIMAL COVARIANCE UNDER THE LAPLACE APPROXIMATION
    DID NOT IMPLEMENT FREE ENERGY UNDER THE LOCAL LINEAR ASSUMPTION,
    IE I DO NOT IGNORE SECOND ORDER DERIVATIVES OF THE FLOWS THAT ARISE IN THE HESSIAN'''

    #return free energy with optimal covariance under the Laplace approximation- up to a different constant
    return  generative_energy_genmu + log_det_Hess_GE /2

log_det_Hess_GE =log_det_hess(generative_energy_genmu,genmu)
FE_laplace= free_energy_laplace(generative_energy_genmu,log_det_Hess_GE)

'''Free energy gradients'''

def grad_free_energy_laplace(FE_laplace,generative_energy_genmu,genmu,lin=True):
    if lin:
        ''' DID NOT IMPLEMENT CHANGES IN THE ENERGY GRADIENT THAT ARISE FROM THE JACOBIANS OF THE FLOWS'''
        return gen_coords.gen_gradient(generative_energy_genmu, genmu)
    else:
        return gen_coords.gen_gradient(FE_laplace, genmu)

grad_FE=grad_free_energy_laplace(FE_laplace,generative_energy_genmu,genmu,lin=local_lin)

'''Flow of generalised gradient on free energy'''

flow_gen_gd_FE = gen_descent_Lag.flow_gengd_Lag(grad_FE, genmu)

'''Data embedding in generalised coordinates'''

'Method 1: finite differences'

if meth_data=='findiff':
    yt_embedded = np.zeros([dim_y, order_y + 1, timesteps, N])

    yt_embedded[:, 0, :, :] = yt

    for o in range(1, order_y + 1):
        for tau in range(1, timesteps):
            yt_embedded[:, o, tau, :] = (yt_embedded[:, o - 1, tau, :] - yt_embedded[:, o - 1, tau - 1, :]) / (
                        Time[tau] - Time[tau - 1])

'Method 2: inverting Taylor expansion'

if meth_data=='Taylor':

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

    T_mx=Taylor_matrix(-dt, order_y)
    inv_T_mx=np.linalg.inv(T_mx)

    yt_embedded=np.zeros([dim_y,order_y+1,timesteps,N])

    for tau in range(timesteps):
        for d in range(dim_y):
                if tau < order_y:
                    temp=np.tile(yt[d,0,:],order_y-tau).reshape([order_y-tau,N])
                    hist = np.concatenate([temp,yt[d, :tau + 1, :]])
                else:
                    hist= yt[d,tau-order_y:tau+1,:]
                yt_embedded[d, :, tau, :] = inv_T_mx @ hist


'''Generalised filtering'''


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

def Heun_gen_filt(genFlow,Time,yt_embedded,geny,genmu):
    # method developed for filtering ONE sample path only
    print('=Heun integration for generalised filtering=')

    [dim_x,order_x_pone]=genmu.shape
    [dim_y,order_y_pone,T]=yt_embedded.shape

    'Initialise means'
    genmut = np.zeros([dim_x, order_x_pone, T])

    'setup function'
    input_sympy= genmu.row_join(geny)
    genF_lambdify = sp.lambdify(input_sympy, genFlow, "numpy")

    for t in range(1,T):
        # if t % 10**5 == 0:
        #      print(t, '/', T)
        h=Time[t]-Time[t-1]
        input_numpy = np.concatenate((genmut[:,:,t-1],yt_embedded[:,:,t-1]),axis=1)
        temp_F=genF_lambdify(*input_numpy.ravel())
        temp_genmut = genmut[:,:,t-1] + h * temp_F
        input_numpy= np.concatenate((temp_genmut,yt_embedded[:,:,t-1]),axis=1)
        genmut[:, :, t] = (genmut[:, :, t - 1] + \
                           h/2 * (genF_lambdify(*input_numpy.ravel())+temp_F))
    return genmut


def linear_interpolation(genmut_variable,Times_variable,Time):
    # this function linearly interpolates the solution to the RK45 algorithm, ie (genmut_variable,Times_variable)
    # to obtain the solution on a different, eg coarser, time grid, ie Time.
    print('=Linear interpolation of RK45 solution=')

    [dim_x,order_x_pone]=genmut_variable.shape[:2]
    T=Time.shape[0]

    'Initialise means'
    genmut = np.empty([dim_x, order_x_pone, T])

    for i in range(T):
        'find time in time variable'
        t= Time[i]
        index = np.sum(Times_variable <= t)-1 #index in Times_variable
        tau = Times_variable[index]  #preceding time in Times_variable
        'find time in time variable'
        if tau == t:
            genmut[:,:,i]=genmut_variable[:,:,index]
        elif tau < t:
            # index2 = np.argmax(Times_variable  t)
            nextau= Times_variable[index+1] #subsequent time in Times_variable
            genmut[:, :, i] = (genmut_variable[:, :, index]*(t-tau)+genmut_variable[:, :, index+1]*(nextau-t))/(nextau-tau)

    return genmut


def RK45_gen_filt(genFlow,Time,yt_embedded,geny,genmu, tol=1e-2 , hmax=1e-1, hmin=1e-6):
    # method developed for filtering ONE sample path only
    print('=RK45 integration for generalised filtering=')

    [dim_x,order_x_pone]=genmu.shape
    [dim_y,order_y_pone,T]=yt_embedded.shape

    'setup function'
    input_sympy= genmu.row_join(geny)
    genF_lambdify = sp.lambdify(input_sympy, genFlow, "numpy")

    'parameters of the RK45 method'
    # weightings for time variable
    # a2 = 2.500000000000000e-01  # 1/4
    # a3 = 3.750000000000000e-01  # 3/8
    # a4 = 9.230769230769231e-01  # 12/13
    # a5 = 1.000000000000000e+00  # 1
    # a6 = 5.000000000000000e-01  # 1/2

    # weightings for space variable
    b21 = 2.500000000000000e-01  # 1/4
    b31 = 9.375000000000000e-02  # 3/32
    b32 = 2.812500000000000e-01  # 9/32
    b41 = 8.793809740555303e-01  # 1932/2197
    b42 = -3.277196176604461e+00  # -7200/2197
    b43 = 3.320892125625853e+00  # 7296/2197
    b51 = 2.032407407407407e+00  # 439/216
    b52 = -8.000000000000000e+00  # -8
    b53 = 7.173489278752436e+00  # 3680/513
    b54 = -2.058966861598441e-01  # -845/4104
    b61 = -2.962962962962963e-01  # -8/27
    b62 = 2.000000000000000e+00  # 2
    b63 = -1.381676413255361e+00  # -3544/2565
    b64 = 4.529727095516569e-01  # 1859/4104
    b65 = -2.750000000000000e-01  # -11/40

    # weightings for the truncation error estimate
    r1 = 2.777777777777778e-03  # 1/360
    r3 = -2.994152046783626e-02  # -128/4275
    r4 = -2.919989367357789e-02  # -2197/75240
    r5 = 2.000000000000000e-02  # 1/50
    r6 = 3.636363636363636e-02  # 2/55

    # weightings for the solution
    c1 = 1.157407407407407e-01  # 25/216
    c3 = 5.489278752436647e-01  # 1408/2565
    c4 = 5.353313840155945e-01  # 2197/4104
    c5 = -2.000000000000000e-01  # -1/5

    'Starting parameters'
    Tmax=np.max(Time) #Max time
    h=hmax #initial timestep, and defines current timestep
    t=0.0 #initial time, and defines current time

    genmut_variable = np.zeros([dim_x, order_x_pone, 1]) #initialise means
    Times_variable=   np.array([0]) #initialise vector of times

    while t < Tmax:

        'Find the current timestep in the original time grid'
        # this is to get the last observation of data
        o_index= np.sum(Time <= t)-1

        'set up timestep'
        if t + h > Tmax:
            h = Tmax - t

        'trapezoidal rule inputs'
        input_numpy1 = np.concatenate((genmut_variable[:,:,-1], yt_embedded[:,:,o_index]),axis=1)
        k1 = h * genF_lambdify(*input_numpy1.ravel())

        input_numpy2 = np.concatenate((genmut_variable[:,:,-1]+ b21 * k1,\
                                       yt_embedded[:,:,o_index]),axis=1)
        k2 = h * genF_lambdify(*input_numpy2.ravel())

        input_numpy3 = np.concatenate((genmut_variable[:,:,-1]+ b31 * k1 + b32 * k2,\
                                       yt_embedded[:,:,o_index]),axis=1)
        k3 = h * genF_lambdify(*input_numpy3.ravel())

        input_numpy4 = np.concatenate((genmut_variable[:, :, -1] + b41 * k1 + b42 * k2 + b43 * k3, \
                                       yt_embedded[:, :, o_index]), axis=1)
        k4 = h * genF_lambdify(*input_numpy4.ravel())

        input_numpy5 = np.concatenate((genmut_variable[:, :, -1] + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, \
                                       yt_embedded[:, :, o_index]), axis=1)
        k5 = h * genF_lambdify(*input_numpy5.ravel())

        input_numpy6 = np.concatenate((genmut_variable[:, :, -1] + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, \
                                       yt_embedded[:, :, o_index]), axis=1)
        k6 = h * genF_lambdify(*input_numpy6.ravel())

        # estimate of truncation error accross each order and dimension
        r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6)

        # take the maximum error along all orders and dimensions
        if r.size > 0:
            r = np.max(r)

        if r<=tol: #in which case we accept the step (tol is the error tolerance per step)
            t = t + h #increase time by timestep
            temp = genmut_variable[:, :, -1] + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 #estimate of solution
            Times_variable = np.append(Times_variable, t)
            genmut_variable = np.concatenate((genmut_variable, temp.reshape([temp.shape[0],temp.shape[1],1])), 2)

        # set up new timestep
        h = 0.9* h * (tol/r)**0.2
        if h > hmax:
            h = hmax
        elif h < hmin:# and Tmax -t >= hmin:
            # raise RuntimeError("Error: Could not converge to the required tolerance.")
            # break
            if Tmax -t > h:
                print("Warning: Could not converge to the required tolerance.")
            h = hmin

    return linear_interpolation(genmut_variable,Times_variable,Time)

# def adaptive_Euler_gen_filt(genFlow,Time,yt_embedded,geny,genmu):
#     # method developed for filtering ONE sample path only
#     print('=adaptive Euler integration for generalised filtering=')
#
#     [dim_x,order_x_pone]=genmu.shape
#     [dim_y,order_y_pone,T]=yt_embedded.shape
#
#     'Initialise means'
#     genmut = np.zeros([dim_x, order_x_pone, T])
#
#     'setup function'
#     input_sympy= genmu.row_join(geny)
#     genF_lambdify = sp.lambdify(input_sympy, genFlow, "numpy")
#
#     'parameters for adaptive step size'
#     time=0 #continuous time
#     prev_t=0
#     t=0
#     iters=0
#     tol=10*Time.size
#     genmu_inst=genmut[:,:,0] #instantaneous genmu
#     genmu_inst_prev = genmut[:, :, 0]
#     while time < Time[T]:
#         # figure out between which indices we are in the time interval (t<=time<t+1)
#         # also figures out whether we just went past a t (indicated by 'update')
#         indices=Time<=time
#         new_t=np.sum(indices)-1
#         if new_t>t:
#             update = True
#         elif new_t==t:
#             update = False
#         else:
#             raise TypeError('update')
#         t = new_t
#         # prepare the input for the flow
#         input_numpy = np.concatenate((genmu_inst,yt_embedded[:,:,t]),axis=1)
#         # flow output
#         F_output=genF_lambdify(*input_numpy.ravel())
#         # define time step size
#         step = (Time[t+1] - Time[t])/np.sum(np.abs(F_output))
#         # update
#         genmu_inst = genmu_inst_prev + step * F_output
#         # if need to update genmut
#         if update:
#             for i in range(prev_t,t):
#                 genmut[:, :, t - 1]=genmu_inst_prev
#         'TODO THIS NEEDS A RETHINK'
#
#     for t in range(1,T):
#         step = Time[t] - Time[t - 1]
#         input_numpy = np.concatenate((genmut[:,:,t-1],yt_embedded[:,:,t-1]),axis=1)
#         genmut[:,:,t]=genmut[:,:,t-1] + step * genF_lambdify(*input_numpy.ravel())
#     return genmut


def integrator_gen_filt_N_samples(genFlow,Time,yt_embedded,geny,genmu,methint='Euler'):

    print('====Integration for generalised filtering (N samples)====')

    [dim_x,order_x_pone]=genmu.shape
    [dim_y,order_y_pone,T,N]=yt_embedded.shape

    'Initialise means'
    genmut = np.empty([dim_x, order_x_pone, T,N])

    'Do generalised filtering for each sample'
    for n in range(N):
        if n>2:
            print(f'---Sample {n}---')
        if methint== 'Euler':
            genmut[:,:,:,n] = Euler_gen_filt(genFlow,Time,yt_embedded[:,:,:,n],geny,genmu)
        # elif methint=='adaptive_Euler':
        #     genmut[:, :, :, n] = adaptive_Euler_gen_filt(genFlow, Time, yt_embedded[:, :, :, n], geny, genmu)
        elif methint == 'Heun':
            genmut[:, :, :, n] = Heun_gen_filt(genFlow, Time, yt_embedded[:, :, :, n], geny, genmu)
        elif methint=='RK45':
            genmut[:, :, :, n] = RK45_gen_filt(genFlow, Time, yt_embedded[:, :, :, n], geny, genmu)
        else: # If an exact match is not confirmed, this last case will be used if provided
            raise TypeError('integrator not yet supported')

        'THIS IS INTRODUCED TO NOT COMPUTE SAMPLE 2 TO MAKE THINGS FASTER. CAN REMOVE THIS'
        if N==2: break

    return genmut

geny = gen_coords.serial_derivs_xt(y, t, order_y + 1)  # shape [dim_x,order+1]

# Generalised filtering
genmut= integrator_gen_filt_N_samples(flow_gen_gd_FE,Time,yt_embedded,geny,genmu,methint=meth_int)
mut_order0=genmut[:,0,:,:]

# 'CODE TO TEST'
# genFlow=flow_gen_gd_FE
# yt_embedded=yt_embedded[:, :, :, n]



'Visualisation'

# sample to plot
n=0

# time to plot
plot_indices=range(0,Time.size)

# plot aesthetics
lw=1
alpha=0.3

plt.figure(figno)
plt.clf()
plt.suptitle(f'1D Coloured OU process', fontsize=16)
plt.title(f'Local linear: {local_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
plt.plot(Time[plot_indices],xt_Euler_conv[0,plot_indices,n],label='Latent')
plt.plot(Time[plot_indices],yt[0,plot_indices,n],linestyle=':',label='Observation')
plt.plot(Time[plot_indices],mut_order0[0,plot_indices,n],label='Inference')
# plt.plot(Time[plot_indices],mut_nonlinear,label='Inference')
# colourline.plot_cool(Time[plot_indices],mut_order0[0,plot_indices,n], lw=1,alpha=alpha)
plt.ylim([-4,11])
plt.legend()


# print(f'MSE={np.sum((xt_Euler_conv[0,:,n]-mut_nonlinear)**2)/Time.size}')


# plt.figure(0)
# plt.clf()
# plt.suptitle(f'1D Coloured OU process', fontsize=16)
# plt.title(f'Local linear: {local_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
# plt.plot(Time,xt_Euler_conv[0,:,n],label='Latent')
# plt.plot(Time,yt[0,:,n],linestyle=':',label='Observation')
# plt.plot(Times_variable,genmut_variable[0,0,:],label='Inference')
# # plt.plot(Time,linear_interpolation(genmut_variable,Times_variable,Time)[0,0,:],label='Inference2')
# # colourline.plot_cool(Time[plot_indices],mut_order0[0,plot_indices,n], lw=1,alpha=alpha)
# plt.legend()


'''Covariance dynamics'''

def gen_filt_cov_N_samples(Hess_gen_energy,Time,genmut,yt_embedded,geny,genmu):
    'DID NOT IMPLEMENT CHANGES IN COVARIANCE THAT ARISE FROM THE LOCAL LINEARISATION'

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

if cov:
    Hess_gen_energy=gen_coords.gen_Hessian(generative_energy_genmu,genmu)
    gencovt_order0 = gen_filt_cov_N_samples_order0(Hess_gen_energy, Time, genmut, yt_embedded, geny, genmu)

'''Free energy dynamics'''
# look at free energy over time


def FEt_N_samples(FE,Time,genmut,yt_embedded,geny,genmu):
    print('====FE over time (N samples)====')

    [dim_y,order_y_pone,T,N]=yt_embedded.shape

    FEt=np.empty([T,N])

    input_sympy = genmu.row_join(geny)
    FE = sp.lambdify(input_sympy, FE, "numpy")

    for n in range(N):
        # print('sample = ', n)
        for t in range(Time.size):
            # if t % 10 ** 5 == 0:
                # print(t, '/', T)
            input_numpy = np.concatenate((genmut[:, :, t, n], yt_embedded[:, :, t, n]), axis=1)
            FEt[t,n]=FE(*input_numpy.ravel())
            if np.isnan(FEt[t,n]):
                print(f't{t},n{n}')
        'THIS IS INTRODUCED TO NOT COMPUTE SAMPLE 2 TO MAKE THINGS FASTER. CAN REMOVE THIS'
        if N==2: break

    return FEt

if fe_time:
    FEt =  FEt_N_samples(FE_laplace,Time,genmut,yt_embedded,geny,genmu)

    plt.figure(figno+1)
    plt.clf()
    plt.suptitle(f'Free energy', fontsize=16)
    plt.title(f'Local linear: {local_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
    plt.plot(Time[plot_indices], xt_Euler_conv[0, plot_indices, n], label='Latent')
    plt.plot(Time[plot_indices], yt[0, plot_indices, n], linestyle=':', label='Observation')
    plt.plot(Time[plot_indices], mut_order0[0, plot_indices, n], label='Inference')
    colourline.plot_cool(Time[plot_indices], FEt[plot_indices, n], lw=lw, alpha=alpha)
    # plt.ylim([-10, 8])


    gen_energy_t =  FEt_N_samples(generative_energy_genmu,Time,genmut,yt_embedded,geny,genmu)
    log_det_Hess_t=FEt_N_samples(log_det_Hess_GE,Time,genmut,yt_embedded,geny,genmu)/2

    plt.figure(figno+2)
    plt.clf()
    plt.suptitle(f'Free energy', fontsize=16)
    plt.title(f'Local linear: {local_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
    plt.plot(Time[plot_indices], xt_Euler_conv[0, plot_indices, n], label='Latent')
    plt.plot(Time[plot_indices], yt[0, plot_indices, n], linestyle=':', label='Observation')
    plt.plot(Time[plot_indices], mut_order0[0, plot_indices, n], label='Inference')
    plt.plot(Time[plot_indices], FEt[plot_indices, n], lw=lw, alpha=alpha, label='FE')
    plt.plot(Time[plot_indices], gen_energy_t[plot_indices, n], lw=lw, alpha=alpha, label='gen_energy')
    plt.plot(Time[plot_indices], log_det_Hess_t[plot_indices, n], lw=lw, alpha=alpha, label='log_det_hess')
    plt.plot(Time[plot_indices], (gen_energy_t+log_det_Hess_t)[plot_indices, n], lw=lw, alpha=alpha, label='sum')
    plt.legend()


    'Performance statistics'

    print(f'Free action={np.nansum(FEt[:, n])}')

print(f'MSE={np.sum((xt_Euler_conv[0,:,n]-mut_order0[0,:,n])**2)/Time.size}')

# test henceforth
np.sum(np.isnan(FEt[plot_indices, n]))
FEt.size
np.sum(np.isnan(log_det_Hess_t[plot_indices, n]))
np.sum(np.isnan(gen_energy_t[plot_indices, n]))
np.nansum(np.sum((gen_energy_t+log_det_Hess_t)[plot_indices, n]-FEt[plot_indices, n]))

input_sympy = genmu.row_join(geny)
Hess_gen_energy = gen_coords.gen_Hessian(generative_energy_genmu, genmu)
sp.simplify(Hess_gen_energy-Hess_gen_energy.T) #good because equals zero
Hess_gen_energy_lambdified = sp.lambdify(input_sympy,Hess_gen_energy , "numpy")
det_Hess_GE=sympy_det(Hess_gen_energy)
det_Hess_GE_lambdified = sp.lambdify(input_sympy,det_Hess_GE , "numpy")
t=190 #t=58 does not work
n=0
input_numpy = np.concatenate((genmut[:, :, t, n], yt_embedded[:, :, t, n]), axis=1)
Hess_t=Hess_gen_energy_lambdified(*input_numpy.ravel())
Hess_t.shape
np.linalg.det(Hess_t)
det_Hess_t=det_Hess_GE_lambdified(*input_numpy.ravel())

det_Hess_t = FEt_N_samples(det_Hess_GE, Time, genmut, yt_embedded, geny, genmu)
np.sum(det_Hess_t[plot_indices, n]<=0)

plt.figure(figno + 3)
plt.clf()
plt.suptitle(f'det Hess', fontsize=16)
plt.title(f'Local linear: {local_lin}, Data embedding: {meth_data}, Integration: {meth_int}', fontsize=12)
plt.plot(Time[plot_indices], det_Hess_t[plot_indices, n], label='det Hess')
plt.legend()


