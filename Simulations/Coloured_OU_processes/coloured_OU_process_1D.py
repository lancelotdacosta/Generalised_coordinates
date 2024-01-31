'''in this script, we compare two ways of generating the solutions of an OU process (ie linear SDE)
driven by white noise convolved with a Gaussian kernel (ie coloured noise)
For such a given two-dimensional process, we compare the sample paths of:
1) The Euler integration method, where the sample paths of the noise process will have been generated using the method of generalised coordinates
2) The full generalised coordinates method used to integrate the stochastic differential equation
'''

import numpy as np
import sympy as sp
from Routines import Taylor_series
from Routines import sampling_generalised_noise
import matplotlib.pyplot as plt
from integration import Euler
from Routines import convolving_white_noise
import matplotlib.cm as cm
import seaborn as sns



'''Part 0: specifying the OU process'''

'Part 0a: Specifying the drift matrix of the OU process'''

dim =1 #State space dimension

#alpha= 1 #Scaling of solenoidal flow

#Q = alpha * np.array([0, 1, -1, 0]).reshape([dim, dim]) #solenoidal flow matrix

#B= np.eye(dim)+Q #negative drift matrix of OU process for a steady state distribution N(0,I)

B=-np.ones(1)

def flow(x):
    return B @ x



'Part 0b: Specifying the kernel of the noise process'

h = sp.symbols('h')
beta=1

K = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h) # NORMALISED Gaussian kernel given as a symbolic expression



'''Part 1: Specifying parameters of integration'''

N= 10**5 #number of sample paths

x0= 10*np.ones([dim, N]) #initial condition


timesteps=10**3
Time = np.linspace(0,100,timesteps) #time grid over which we integrate

'Part 1a: getting the serial derivatives of the noise process'

order = 10

at=0

tilde_w0= sampling_generalised_noise.sample_gen_noise(K=K,wrt=h,at=at,order= order,N=N) #Generalised fluctuations at time zero


#Point of info: determinant of generalised covariance as a function of order grows faster than exponentially
order_simul=10 #order for this simulation
tilde_Sigma= sampling_generalised_noise.gen_covariance(K=K,wrt=h,at=at,order= order_simul)
dets = []
for i in range(1, order_simul+1):
    dets.append(np.linalg.det(tilde_Sigma[:i,:i]))
plt.figure(0)
plt.clf()
plt.plot(range(1, order_simul+1), dets)
plt.yscale('log')
plt.suptitle(r'Determinant of $\tilde \Sigma$ w.r.t. number of generalised orders', fontsize=16)



'''Part 2: Generalised coordinates integration'''
#Input: flow function

'Part 2a: Getting serial derivatives of solution for each sample at time at '

tilde_x0= np.empty([dim, order+1, N])
tilde_x0[:,0,:]=x0

for i in range(order):
    tilde_x0[:,i+1,:]=flow(tilde_x0[:,i,:]) + tilde_w0[i,:]


'Part 2b: Generating sample paths of solution'


xt= np.empty([dim, Time.size, N])

for i in range(N):
    xt[:,:, i] = Taylor_series.taylor_eval(derivs=tilde_x0[0,:,i] ,at=0,Time=Time) #redo dim




'''Part 3: Euler integration where white noise is given by generalised coordinate sample paths'''


'Part 3a: getting the sample paths of the noise process'

wt = np.empty([dim, Time.size, N])  # white noise generated by generalised coordinates
for i in range(N):
    wt[:,:, i] = Taylor_series.taylor_eval(derivs=tilde_w0[:, i], at=0,Time=Time)


'Part 3b: getting the solution by Euler integration'


xt_Euler_gen= Euler.Euler_integration(x0=x0,f=flow, wt=wt,Time=Time)


'''Part 4: Euler integration where white noise is convolved by a Gaussian kernel'''

'''Part 4a: Sampling white noise is convolved by a Gaussian kernel'''

wt_conv = convolving_white_noise.white_noise_conv_Gaussian_kernel_nd(dim, Time, N,beta)

'''Part 4b: Euler integration with the above noise samples'''

xt_Euler_conv= Euler.Euler_integration(x0=x0,f=flow, wt=wt_conv,Time=Time)


'''Part 5: Path of least action'''


'Part 5a: getting the path of least action, ie path in the absence of noise'
xt_least=np.empty([dim,Time.size,N])
for t in range(timesteps):
    xt_least[:,t,:]=np.exp(B*Time[t])@x0

#xt_least= Euler.Euler_integration(x0=x0,f=flow, wt=np.zeros(wt.shape),Time=Time)


'''Part 6: Plots'''

plot_indices=range(min(30,timesteps))
N_plot= min(N, 64)
lw=0.5
c_least= 'gray' #colour path of least action

'Part 6a: 1D plot of gen coord Zig-zag method'

plt.figure(1)
plt.clf()
for n in range(N_plot):  # Iterating over samples of white noise
    plt.plot(Time[plot_indices], xt[0,plot_indices, n], linewidth=lw,alpha=0.5)
plt.plot(Time[plot_indices], xt_least[0,plot_indices,0], color=c_least,linestyle=':')
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
plt.xlabel(r'Time $t$')
plt.ylabel(r'$x_t$')
plt.suptitle('Coloured OU process', fontsize=16)
plt.title(f'Zig-zag method, order={tilde_x0.shape[1]}', fontsize=14)
plt.ylim(top=11,bottom=-5)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)
plt.savefig("OU_1D_zigzag.png", dpi=100)



'Part 6b: 1D plot of hybrid Euler-Gen method'

plt.figure(2)
plt.clf()
for n in range(N_plot):  # Iterating over samples of white noise
    plt.plot(Time[plot_indices], xt_Euler_gen[0,plot_indices, n], linewidth=lw,alpha=0.5)
plt.plot(Time[plot_indices], xt_least[0,plot_indices,0], color=c_least,linestyle=':')
plt.xlabel(r'Time $t$')
plt.ylabel(r'$x_t$')
plt.suptitle('Coloured OU process', fontsize=16)
plt.title(f'Hybrid Euler-Gen method, order={tilde_x0.shape[1]}', fontsize=14)
plt.ylim(top=11,bottom=-5)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


'Part 6c: 1D plot of classical Euler-conv method'

plt.figure(3)
plt.clf()
for n in range(N_plot):  # Iterating over samples of white noise
    plt.plot(Time[plot_indices], xt_Euler_conv[0,plot_indices, n], linewidth=lw,alpha=0.5)
plt.plot(Time[plot_indices], xt_least[0,plot_indices,0], color=c_least,linestyle=':')
plt.xlabel(r'Time $t$')
plt.ylabel(r'$x_t$')
plt.suptitle('Coloured OU process', fontsize=16)
plt.title(f'Classical Euler-Conv method', fontsize=14)
plt.ylim(top=11,bottom=-5)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)
plt.savefig("OU_1D_Eulerconv.png", dpi=100)



'Part 6c: 1D plot of convolved white noise sample paths generated using generalised coordinates'

plt.figure(4)
plt.clf()
wt_least =np.zeros(Time.size)
for n in range(N_plot):  # Iterating over samples of white noise
    plt.plot(Time[plot_indices], wt[0,plot_indices, n], linewidth=lw,alpha=0.5)
plt.plot(Time[plot_indices], wt_least[plot_indices], color=c_least,linestyle=':')
plt.xlabel(r'Time $t$')
plt.ylabel(r'$w_t$')
plt.suptitle('Coloured noise', fontsize=16)#
plt.title(f'Generalised coordinates, order={order}', fontsize=14)
plt.ylim(top=3,bottom=-3)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


'Part 6c: 1D plot of white noise sample paths convolved with a Gaussian kernel'

plt.figure(5)
plt.clf()
wt_least =np.zeros(Time.size)
for n in range(N_plot):  # Iterating over samples of white noise
    plt.plot(Time[plot_indices], wt_conv[0,plot_indices, n], linewidth=lw,alpha=0.5)
plt.plot(Time[plot_indices], wt_least[plot_indices], color=c_least,linestyle=':')
plt.xlabel(r'Time $t$')
plt.ylabel(r'$w_t$')
plt.suptitle('Coloured noise', fontsize=16)#
plt.title(f'Standard convolution method', fontsize=14)
plt.ylim(top=3,bottom=-3)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)



'''Part 7: Check whether the generalised coordinate solution is a Gaussian process'''

'Part 7a: Plot the joint distribution of the solution---obtained through generalised coordinates---at two different times'
[T_1,T_2]=[5,15] #first and second timestep at which we plot
[plot_T_1,plot_T_2]=[Time[T_1],Time[T_2]]

plt.figure(6)
plt.hist2d(xt[0, T_1, :], xt[0, T_2, :], bins=(50, 50), cmap=cm.jet)
plt.suptitle('Joint probability at two different times')
plt.xlabel(r'$x_{0.5}$')
plt.ylabel(r'$x_{1.5}$')


# 2D histogram of joint distribution to show x is not a Gaussian process but has Gaussian marginals
# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
sns.set(style="white", color_codes=True)
sns.jointplot(x=xt[0, T_1, :], y=xt[0, T_2, :], kind='hex', space=0, cmap='Blues', color='skyblue')
plt.xlabel(r'$x_{0.5}$',labelpad=0)
plt.ylabel(r'$x_{1.5}$',labelpad=0)
#plt.xlim(-3, 3)
#plt.ylim(-3, 3)
plt.tick_params(axis='both', which='major', pad=-3)
plt.show()
fig = plt.gcf()
#plt.suptitle('$p(x_s)$')
ratio = 1.3
len = 5
fig.set_size_inches(ratio * len, len, forward=True)
#plt.savefig("non-Gaussian_diffprocess.png", dpi=100)






