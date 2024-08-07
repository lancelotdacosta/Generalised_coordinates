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
from Routines import colourline
import matplotlib.pyplot as plt
from integration import Euler
from scipy.linalg import expm
from Routines import convolving_white_noise



'''Part 0: specifying the OU process'''

'Part 0a: Specifying the drift matrix of the OU process'''

dim =2 #State space dimension

alpha = 1 #Scaling of solenoidal flow

Q = np.array([0, 1, -1, 0]).reshape([dim, dim]) #solenoidal flow matrix

B= -(np.eye(dim)+alpha*Q) #drift matrix of OU process for a steady state distribution N(0,I)

def flow(x):
    return B @ x

#B=-np.ones(1)

'Part 0b: Specifying the kernel of the noise process'

h = sp.symbols('h')
beta=1

K = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h) # NORMALISED Gaussian kernel given as a symbolic expression



'''Part 1: Specifying parameters of integration'''

N = 30 #number of sample paths

x0 = 10*np.ones([dim, N]) #initial condition


timesteps=10**4
Time = np.linspace(0,100,timesteps) #time grid over which we integrate

'Part 1a: getting the serial derivatives of the noise process'

order = 10

at=0

tilde_w0 = sampling_generalised_noise.sample_gen_noise_nd(K=K,wrt=h,at=at,order= order,N=N,dim=dim) #Generalised fluctuations at time zero


#Point of info: determinant of generalised covariance (of 1D noise) as a function of order grows faster than exponentially
#order_simul=10 #order for this simulation
#tilde_Sigma= sampling_generalised_noise.gen_covariance(K=K,wrt=h,at=at,order= order_simul)
#dets = []
#for i in range(1, order_simul+1):
#    dets.append(np.linalg.det(tilde_Sigma[:i,:i]))
#plt.figure(0)
#plt.clf()
#plt.plot(range(1, order_simul+1), dets)
#plt.yscale('log')
#plt.suptitle(r'Determinant of $\tilde \Sigma$ w.r.t. number of generalised orders', fontsize=16)



'''Part 2: Generalised coordinates integration'''
#Input: flow function

'Part 2a: Getting serial derivatives of solution for each sample at time at '

tilde_x0 = np.empty([dim, order+1, N])
tilde_x0[:,0,:]=x0

for i in range(order):
    for n in range(N):
        tilde_x0[:,i+1,n]= flow(tilde_x0[:,i,n]) + tilde_w0[:,i,n]


'Part 2b: Generating sample paths of solution'


xt= np.empty([dim, Time.size, N])

for n in range(N):
    xt[:,:, n] = Taylor_series.taylor_eval_nd(derivs=tilde_x0[:,:,n] ,at=0,Time=Time)




'''Part 3: Euler integration where white noise is given by generalised coordinate sample paths'''


'Part 3b: getting the sample paths of the noise process'

wt = np.empty([dim,Time.size, N])  # white noise generated by generalised coordinates
for n in range(N):
    wt[:, :, n] = Taylor_series.taylor_eval_nd(derivs=tilde_w0[:, :, n], at=0,Time=Time)


'Part 3b: getting the solution by Euler integration'


xt_Euler_gen= Euler.Euler_integration(x0=x0,f=flow, wt=wt,Time=Time)


'''Part 4: Euler integration where white noise is convolved by a Gaussian kernel'''

'''Part 4a: Sampling white noise is convolved by a Gaussian kernel'''

wt_conv = convolving_white_noise.white_noise_conv_Gaussian_kernel_nd(dim,Time, N,beta)

'''Part 4b: Euler integration with the above noise samples'''

xt_Euler_conv= Euler.Euler_integration(x0=x0,f=flow, wt=wt_conv,Time=Time)



'''Part 5: Path of least action'''


'Part 5a: getting the path of least act ion, ie path in the absence of noise'
xt_least=np.empty([dim,Time.size,N])
for t in range(timesteps):
    xt_least[:,t,:]=expm(B*Time[t])@x0
    #if t%20==0 and t<300:
    #    print(xt_least[:,t,:])

#xt_least= Euler.Euler_integration(x0=x0,f=flow, wt=np.zeros(wt.shape),Time=Time)


'''Part 6: Plots'''

plot_timesteps=min(280,Time.size)
plot_indices=range(min(plot_timesteps,timesteps))
lw=0.5
alpha=0.3

'Part 6a: 2D plot of gen coord Zig-zag method sample paths'

#n=1 #trajectory to plot

plt.figure(1)
plt.clf()
plt.suptitle(f'2D Coloured OU process', fontsize=16)
plt.title(f'Zig-zag method, order={tilde_x0.shape[1]}', fontsize=14)
plt.plot(xt_least[0,plot_indices,0], xt_least[1,plot_indices,0], color='gray',linestyle=':')
for n in range(N):
    #plt.plot(xt[0, plot_indices, n], xt[1, plot_indices, n], linewidth=lw, alpha=alpha)
    colourline.plot_cool(xt[0, plot_indices, n].reshape(plot_timesteps), xt[1, plot_indices, n].reshape(plot_timesteps), lw=lw,alpha=alpha)
plt.xlim(right=12,left=-5)
plt.ylim(top=12,bottom=-5)
plt.savefig("OU_2D_zigzag.png", dpi=100)



'Part 6b: 2D plot of hybrid Euler-Gen method'

plt.figure(2)
plt.clf()
plt.plot(xt_least[0,plot_indices,0], xt_least[1,plot_indices,0], color='gray',linestyle=':')
for n in range(N):  # Iterating over samples of white noise
    #plt.plot(xt_Euler_gen[0,plot_indices, n], xt_Euler_gen[1,plot_indices, n], linewidth=lw,alpha=alpha)
    colourline.plot_cool(xt_Euler_gen[0, plot_indices, n].reshape(plot_timesteps), xt_Euler_gen[1, plot_indices, n].reshape(plot_timesteps), lw=lw,alpha=alpha)
plt.suptitle('2D Coloured OU process', fontsize=16)
plt.title(f'Hybrid Euler-Gen method, order={tilde_x0.shape[1]}', fontsize=14)
plt.xlim(right=12,left=-5)
plt.ylim(top=12,bottom=-5)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


'Part 6c: 2D plot of classical Euler-conv method'

plt.figure(3)
plt.clf()
plt.plot(xt_least[0,plot_indices,0], xt_least[1,plot_indices,0], color='gray',linestyle=':')
for n in range(N):  # Iterating over samples of white noise
    #plt.plot(xt_Euler_gen[0,plot_indices, n], xt_Euler_gen[1,plot_indices, n], linewidth=lw,alpha=alpha)
    colourline.plot_cool(xt_Euler_conv[0, plot_indices, n].reshape(plot_timesteps), xt_Euler_conv[1, plot_indices, n].reshape(plot_timesteps), lw=lw,alpha=alpha)
plt.suptitle('2D Coloured OU process', fontsize=16)
plt.title(f'Classical Euler-Conv method', fontsize=14)
plt.xlim(right=12,left=-5)
plt.ylim(top=12,bottom=-5)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)
plt.savefig("OU_2D_Eulerconv.png", dpi=100)




'Part 6d: 2D plot of convolved white noise, generalised coordinate sample paths'

plt.figure(5)
plt.clf()
for n in range(N):  # Iterating over samples of white noise
    #plt.plot(wt[0,plot_indices, n], wt[1,plot_indices, n], linewidth=lw,alpha=0.5)
    colourline.plot_cool(wt[0, plot_indices, n].reshape(plot_timesteps), wt[1, plot_indices, n].reshape(plot_timesteps), lw=lw,alpha=0.5)
plt.suptitle('2D Coloured noise', fontsize=16)
plt.title(f'Generalised coordinates, order={order}', fontsize=14)
plt.xlim(right=2.5,left=-2.5)
plt.ylim(top=2.5,bottom=-2.5)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


'Part 6e: 2D plot of white noise sample paths convolved with a Gaussian kernel'

plt.figure(6)
plt.clf()
for n in range(N):  # Iterating over samples of white noise
    #plt.plot(wt[0,plot_indices, n], wt[1,plot_indices, n], linewidth=lw,alpha=0.5)
    colourline.plot_cool(wt_conv[0, plot_indices, n].reshape(plot_timesteps), wt_conv[1, plot_indices, n].reshape(plot_timesteps), lw=lw,alpha=0.5)
plt.suptitle('2D Coloured noise', fontsize=16)
plt.title(f'Standard convolution method', fontsize=14)
plt.xlim(right=2.5,left=-2.5)
plt.ylim(top=2.5,bottom=-2.5)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


'Part 6f: 1D plot of convolved white noise generalised coordinate sample paths'

plt.figure(7)
plt.clf()
for n in range(N):
    #plt.plot(Time[plot_indices], wt[1, plot_indices, n] / wt[0, plot_indices, n], linewidth=lw, alpha=1,color='b')
    for d in range(dim):  # Iterating over samples of white noise
        plt.plot(Time[plot_indices], wt[d,plot_indices, n], linewidth=lw,alpha=0.5)
plt.xlabel(r'Time $t$')
plt.ylabel(r'$w_t$')
plt.suptitle('Coloured noise', fontsize=16)#
plt.title(f'Generalised coordinates, order={order}', fontsize=14)
plt.ylim(top=10,bottom=-10)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)

'Part 6g: 1D plot of convolved white noise generalised coordinate sample paths'

plt.figure(8)
plt.clf()
for n in range(N):
    #plt.plot(Time[plot_indices], wt[1, plot_indices, n] / wt[0, plot_indices, n], linewidth=lw, alpha=1,color='b')
    for d in range(dim):  # Iterating over samples of white noise
        plt.plot(Time[plot_indices], wt_conv[d,plot_indices, n], linewidth=lw,alpha=0.5)
plt.xlabel(r'Time $t$')
plt.ylabel(r'$w_t$')
plt.suptitle('Coloured noise', fontsize=16)#
plt.title(f'Standard convolution method', fontsize=14)
plt.ylim(top=10,bottom=-10)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


