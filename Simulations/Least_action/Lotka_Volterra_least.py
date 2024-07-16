'''in this script we approximately recover the path of least action of a Lotka-Volterra system
through a generalised gradient descent on the Lagrangian'''

import numpy as np
import sympy as sp
from Routines import Taylor_series,sampling_generalised_noise,colourline,gen_coords, convolving_white_noise
import matplotlib.pyplot as plt
from integration import gen_descent_Lag, lin_zigzag, zigzag, ODE
from scipy.linalg import expm
import time
import matplotlib.cm as cm

'''Part 0: specifying the process'''

'Part 0a: Specifying the parameters of the flow'

dim =2 #State space dimension

#parameters of Lorentz system
a = 1
b =1
c =1
d = 1
#only a/d affects the nature of the solution up to rescaling of space and time

'Part 0a: Specifying the numpy form of the flow'

def flow(x):
    y=np.empty(x.shape)
    y[0]= a*x[0]-b*x[0]*x[1] #rate of change of prey
    y[1] = c*x[0]*x[1]-d*x[1] #rate of change of predators
    return y

def flow_N_samples(x):
    y=np.empty(x.shape)
    y[0,:]= a*x[0,:]-b*x[0,:]*x[1,:] #rate of change of prey
    y[1, :] = c*x[0,:]*x[1,:]-d*x[1,:] #rate of change of predators
    return y

'Part 0a: Specifying the sympy form of the flow'

#symbolic time variable
t = sp.Symbol("t")

#symbolic space variable
x= sp.Matrix([sp.Function("x0")(t),
              sp.Function("x1")(t)])

#Flow of Lotka Volterra system
F=sp.Matrix([a*x[0]-b*x[0]*x[1],
             c*x[0]*x[1]-d*x[1]])

'Part 0b: Specifying the kernel of the noise process'

h = sp.symbols('h')
beta=1

K = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h) # NORMALISED Gaussian kernel given as a symbolic expression


'''Part 1: Specifying parameters of integration'''

N = 2 #30 #number of sample paths

x0 = 1.5*np.ones([dim,N]) #initial condition representing the proportion of prey and predators

T= 0.5
# timesteps=T*10**4#28*10**1)
Time = np.linspace(0,T,2) #time span


'''Part 2: True path of least action'''

xt_least=np.empty([dim,Time.size])
print('True path of least action')
# xt_least,Time_least=ODE.RK45_integration(x0,flow,Time)
xt_least,Time_least=ODE.RK45_integration(x0[:,0],flow,Time)


'''Part 3: Approximate paths of least action'''

'Part 3a: specifying hyperparameters'

order = 6 #10
scaling=range(3)
genx=gen_coords.serial_derivs_xt(x,t,order+1)

'Part 3b: Linearised case'

'Generalised gradient descent on Lagrangian'

# grad_Lag_lin=  gen_coords.Lagrangian_gradient(F,x,t,K,h,order,lin=True,trun=True)
#
# tilde_x0_lin= lin_zigzag.sol_lin_gen_coord(F,x,t,x0,np.zeros([dim,order,N]))[:,:,0]
# # tilde_x0_lin= zigzag.sol_gen_coords(F,x,t,x0,np.zeros([dim,order,N]))[:,:,0]
#
# xt_lin_Lag=np.zeros([len(scaling),dim,order+1,Time.size])
#
# s=1 #scaling
# flow_gen_gd_Lag_lin = gen_descent_Lag.flow_gen_gd_Lag(10 ** s * grad_Lag_lin, x, t, order)
#
# xt_lin_Lag,Time_lin=gen_descent_Lag.RK45_gen_flow(tilde_x0_lin,flow_gen_gd_Lag_lin,genx,Time,tol=1e-1 , hmax=1e-1, hmin=1e-6)
#
#
# 'Evaluation of Lagrangian over time'
# Lag_lin=  gen_coords.Lagrangian(F,x,t,K,h,order,lin=True,trun=True)
#
# Lag_lin_lambdify = sp.lambdify(genx, Lag_lin, "numpy")
#
# Lag_lin_xt=np.zeros([Time_lin.size])
# # Lag_lin_xt=np.zeros([len(scaling),Time.size])
#
# for tau in range(Time_lin.size):
#     # if tau % 10 ** 5 == 0:
#     #     print(tau / Time_lin.size)
#     Lag_lin_xt[tau] = Lag_lin_lambdify(*xt_lin_Lag[:, :, tau].ravel())
#     # Lag_lin_xt[s, tau] = Lag_lin_lambdify(*xt_lin_Lag[:, :, tau].ravel())

'Part 3b: Non-Linearised case'

lin=True

'Generalised gradient descent on Lagrangian'

grad_Lag=  gen_coords.Lagrangian_gradient(F,x,t,K,h,order,lin=lin,trun=True)

if lin:
    tilde_x0= lin_zigzag.sol_lin_gen_coord(F,x,t,x0,np.zeros([dim,order,N]))[:,:,0]
else:
    tilde_x0= zigzag.sol_gen_coords(F,x,t,x0,np.zeros([dim,order,N]))[:,:,0]

# tilde_x0=np.zeros([dim,order+1])
# tilde_x0[:,0]=x0[:,0]

xt_Lag=np.zeros([len(scaling),dim,order+1,Time.size])

s=2 #scaling
flow_gen_gd_Lag = gen_descent_Lag.flow_gen_gd_Lag(10 ** s * grad_Lag, x, t, order)

xt_Lag,Time_Lag=gen_descent_Lag.RK45_gen_flow(tilde_x0,flow_gen_gd_Lag,genx,Time,tol=1e-4 , hmax=1e-5, hmin=1e-6)


'Evaluation of Lagrangian over time'
Lag=  gen_coords.Lagrangian(F,x,t,K,h,order,lin=lin,trun=True)

Lag_lambdify = sp.lambdify(genx, Lag, "numpy")

Lag_xt=np.zeros([Time_Lag.size])
# Lag_lin_xt=np.zeros([len(scaling),Time.size])

for tau in range(Time_Lag.size):
    # if tau % 10 ** 5 == 0:
    #     print(tau / Time_lin.size)
    Lag_xt[tau] = Lag_lambdify(*xt_Lag[:, :, tau].ravel())
    # Lag_lin_xt[s, tau] = Lag_lin_lambdify(*xt_lin_Lag[:, :, tau].ravel())


'''Part 6: Plots'''

c_prey= cm.get_cmap('Blues')
c_predator= cm.get_cmap('Reds')
ylim_1D=(-0,2.5)
lw=0.5
alpha=0.3


'''1D plots'''

plt.figure(4)
print('Plot 4')
plt.clf()
# plt.suptitle(f'Stochastic Lotka Volterra', fontsize=16)
plt.title(f'True least action path', fontsize=14)
# colourline.plot_cmap(Time_least, xt_least[0, :], lw=lw,alpha=alpha,cmap=c_prey,crev=True)
# colourline.plot_cmap(Time_least, xt_least[1, :], lw=lw, alpha=alpha,cmap=c_predator,crev=True)
plt.plot(Time_least, xt_least[0,:], color='blue',linestyle=':')
plt.plot(Time_least, xt_least[1,:], color='red',linestyle=':')
plt.ylim(ylim_1D)
# plt.xscale('log')
plt.xlabel(r'Time')

# plt.figure(5)
# print('Plot 5')
# plt.clf()
# # plt.suptitle(f'Stochastic Lotka Volterra', fontsize=16)
# plt.title(f'Least action path linearised', fontsize=14)
# colourline.plot_cmap(Time_lin, xt_lin_Lag[0,0, :], lw=lw,alpha=alpha,cmap=c_prey,crev=True)
# colourline.plot_cmap(Time_lin, xt_lin_Lag[1,0, :], lw=lw, alpha=alpha,cmap=c_predator,crev=True)
# plt.plot(Time_least, xt_least[0,:], color='blue',linestyle=':')
# plt.plot(Time_least, xt_least[1,:], color='red',linestyle=':')
# plt.ylim(ylim_1D)
# # plt.xscale('log')
# plt.xlabel(r'Time')
#
# plt.figure(6)
# print('Plot 6')
# plt.clf()
# # plt.suptitle(f'Stochastic Lotka Volterra', fontsize=16)
# plt.title(f'Lagrangian over time linearised', fontsize=14)
# plt.plot(Time_lin, Lag_lin_xt)
# plt.xlabel(r'Time')


plt.figure(7)
print('Plot 7')
plt.clf()
# plt.suptitle(f'Stochastic Lotka Volterra', fontsize=16)
plt.title(f'Least action path, linear: {lin}', fontsize=14)
# colourline.plot_cmap(Time_Lag, xt_Lag[0,0, :], lw=lw,alpha=alpha,cmap=c_prey,crev=True)
# colourline.plot_cmap(Time_Lag, xt_Lag[1,0, :], lw=lw, alpha=alpha,cmap=c_predator,crev=True)
plt.plot(Time_Lag, xt_Lag[0,0, :], lw=lw,alpha=alpha, color='blue')
plt.plot(Time_Lag, xt_Lag[1,0, :], lw=lw, alpha=alpha, color='red')
plt.plot(Time_least, xt_least[0,:], color='blue',linestyle=':')
plt.plot(Time_least, xt_least[1,:], color='red',linestyle=':')
# plt.ylim(ylim_1D)
# plt.xscale('log')
plt.xlabel(r'Time')

plt.figure(8)
print('Plot 8')
plt.clf()
# plt.suptitle(f'Stochastic Lotka Volterra', fontsize=16)
plt.title(f'Lagrangian over time, linear: {lin}', fontsize=14)
plt.plot(Time_Lag, Lag_xt)
plt.xlabel(r'Time')

