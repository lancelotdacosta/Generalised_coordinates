'''in this code, we use the same two dimensional OU process as in the nearby script
and test how accurate the method for recovering the path of least action is
By doing a generalized gradient descent on the Lagrangian.
We use various different scalings of the Lagrangian in the generalized gradient descent
To see if this affects the dynamics and how.'''


import numpy as np
import sympy as sp
from Routines import Taylor_series,sampling_generalised_noise,colourline,gen_coords, convolving_white_noise
import matplotlib.pyplot as plt
from integration import Euler, gen_descent_Lag, lin_zigzag
from scipy.linalg import expm
import time




'''Part 0: specifying the OU process'''

'Part 0a: Specifying the flow of the OU process'''

dim =2 #State space dimension

alpha = 1 #Scaling of solenoidal flow

Q = np.array([0, 1, -1, 0]).reshape([dim, dim]) #solenoidal flow matrix

B= -(np.eye(dim)+alpha*Q) #drift matrix of OU process for a steady state distribution N(0,I)

def flow(x): #numpy form of flow
    return B @ x

#symbolic time variable
t = sp.Symbol("t")

x= sp.Matrix([sp.Function("x0")(t),
              sp.Function("x1")(t)])

F=B @ x #symbolic flow


'Part 0b: Specifying the kernel of the noise process'

h = sp.symbols('h')
beta=1

K = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h) # NORMALISED Gaussian kernel given as a symbolic expression



'''Part 1: Specifying parameters of integration'''

N = 2 #30 #number of sample paths

x0 = 10*np.ones([dim,N]) #initial condition

T= 100#2.8
timesteps=T*10**4#28*10**1)
Time = np.linspace(0,T,timesteps) #time grid over which we integrate

'Part 1a: getting the serial derivatives of the noise process'

order = 4 #10


'''Part 2: True path of least action'''


'Part 2a: getting the path of least action, ie path in the absence of noise'
xt_least=np.empty([dim,Time.size,N])
for tau in range(timesteps):
    if tau % 10 ** 5 == 0:
        print(tau,'/',Time.size)
    xt_least[:,tau,:]=expm(B*Time[tau])@x0


'''Part 3: Approximations to path of least action'''

'Part 5b: getting the path of least action, by doing a generalised gd on Lagrangian'
grad_Lag=  gen_coords.Lagrangian_gradient(F,x,t,K,h,order,lin=True,trun=True)
Lag=  gen_coords.Lagrangian(F,x,t,K,h,order,lin=True,trun=True)




'Initialize conditions for simulation'
test_tilde_x0= lin_gen_coord.sol_lin_gen_coord(F,x,t,x0[:,:2],np.zeros([dim,order,2]))[:,:,0]
genx=gen_coords.serial_derivs_xt(x,t,order+1)

# scaling of lagrangian
scaling=range(-2,3)

# Solution
xt_gen_gd_Lag=np.zeros([len(scaling),dim,order+1,Time.size])

# Lagrangian of solution
Lag_xt_gen_gd=np.zeros([len(scaling),Time.size])

Lag_lambdify = sp.lambdify(genx, Lag, "numpy")

for s in range(len(scaling)):
    print('scaling',scaling[s])

    'Flow least action path'
    flow_gen_gd_Lag= gen_descent_Lag.flow_gen_gd_Lag(10**scaling[s]*grad_Lag,x,t,order)

    'integrate path of least action'
    xt_gen_gd_Lag[s,:,:,:]=gen_descent_Lag.Euler_gen_flow(test_tilde_x0,flow_gen_gd_Lag,genx,Time)

    'Lagrangian of solution'
    for tau in range(Time.size):
        if t % 10**5 == 0:
            print(tau/Time.size)
        Lag_xt_gen_gd[s,tau]= Lag_lambdify(*xt_gen_gd_Lag[s,:,:,tau].ravel())


'''Part 6: Plots'''

plot_timesteps=timesteps#min(280,Time.size)
plot_indices=range(min(plot_timesteps,timesteps))
lw=0.5
alpha=0.3

'Part 6a: 2D plot of gen coord Zig-zag method sample paths'

#n=1 #trajectory to plot
plt.figure(0)
plt.clf()
plt.suptitle(f'2D Coloured OU process', fontsize=16)
plt.title(f'order={order+1},T={T},steps/time={timesteps/T}', fontsize=14)
for s in range(2,len(scaling)):
    plt.plot(Time, Lag_xt_gen_gd[s,plot_indices],linestyle=':',label=f'{10**scaling[s]}')
plt.legend()


plt.figure(1)
plt.clf()
plt.suptitle(f'2D Coloured OU process', fontsize=16)
plt.title(f'order={order+1},T={T},steps/time={timesteps/T}', fontsize=14)
plt.plot(xt_least[0,plot_indices,0], xt_least[1,plot_indices,0], color='black')
for s in range(2,len(scaling)):
    plt.plot(xt_gen_gd_Lag[s,0,0,plot_indices], xt_gen_gd_Lag[s,1,0,plot_indices],linestyle=':',label=f'{10**scaling[s]}')
plt.legend()
# plt.xlim(right=20, left=-20)
# plt.ylim(top=20,bottom=-40)

'Test influence of order On the motion by looking at the generalized precision matrix'

gen_cov_nd = sampling_generalised_noise.gen_covariance_nd(K=K, wrt=h, at=0, order=order, dim=dim)
gen_pres_nd = np.linalg.inv(gen_cov_nd)

# By looking at the entries of the Generalized precision matrix We can see that the higher orders of motion
# are assigned much less precision than the low orders of motion.
# This is a good thing because it means that the lower orders of motion will contribute the most to the Lagrangian
# and hence to the Generalized gradient descent


#
#
# xt_gen_gd_Lag_saved=xt_gen_gd_Lag
#
# #T=10, timesteps=10**-5
# plt.figure(3)
# plt.clf()
# plt.suptitle(f'2D Coloured OU process', fontsize=16)
# plt.title(f'order={order+1},T={T},steps/time={timesteps/T}', fontsize=14)
# plt.plot(xt_least[0,plot_indices,0], xt_least[1,plot_indices,0], color='black')
# for s in range(len(scaling)):
#     plt.plot(xt_gen_gd_Lag[s,0,0,plot_indices], xt_gen_gd_Lag[s,1,0,plot_indices],linestyle=':',label=f'{10**scaling[s]}')
# plt.legend()
#
# #T=10, timesteps=10**-5
# plt.figure(4)
# plt.clf()
# plt.suptitle(f'2D Coloured OU process', fontsize=16)
# plt.title(f'order={order+1},T={T},steps/time={timesteps/T}', fontsize=14)
# plt.plot(xt_least[0,plot_indices,0], xt_least[1,plot_indices,0], color='black')
# for s in range(len(scaling)):
#     plt.plot(xt_gen_gd_Lag[s,0,0,plot_indices], xt_gen_gd_Lag[s,1,0,plot_indices],linestyle=':',label=f'{10**scaling[s]}')
# plt.legend()
# plt.xlim(right=20, left=-20)
# plt.ylim(top=20,bottom=-40)

