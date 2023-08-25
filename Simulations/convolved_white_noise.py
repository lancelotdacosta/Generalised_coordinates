'''In this file we compare two alternative ways generating White noise convolved with a Gaussian kernel
1) First, we generate this process in the usual way: We generate white noise and convolve it with a Gaussian kernel
2) Second, we use the method of generalised coordinates, where we first generate the serial derivatives of the process
at the time origin and then sum them in a Taylor series to obtain the process.
'''

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from Routines import Taylor_series
from Routines import sampling_generalised_noise

# plot parameters
lw = 0.5


'''Part 1: Standard generation of white noise convolved with a Gaussian kernel'''

'Part 1a: Setting up the time and sample parameters'

decimal_tol = 2 #decimal tolerance for timesteps
timestep =10**-decimal_tol # Number of time steps

#setting up master time interval
T=10 # boundary of master time interval
Time = np.arange(-T, T+timestep, timestep)
Time=np.round(Time,decimal_tol)

timesteps = Time.size # Number of time steps

N = 32 # Number of white noise sample paths


'Part 1b: Setting up the convolution kernel'

#def k(t, beta):  # unnormalised Gaussian kernel
#    return np.exp(-beta * t * t)

# Gaussian kernel parameter
beta = 1  # 10 ** -1  # scaling parameter

def k(t):  # normalised Gaussian kernel
    return np.exp(-beta * t * t)*np.sqrt(beta/np.pi)

# kernel = np.exp(-Time * Time) #Gaussian kernel

'Figure 1: Plotting Gaussian kernel'
plot_indices = range(-int(timesteps / 10) + int(timesteps / 2), int(timesteps / 10) + int(
    timesteps / 2))  # have a smaller plot time

plt.figure(0)
plt.clf()
plt.plot(Time[plot_indices], k(Time[plot_indices]), linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('Gaussian kernel', fontsize=16)
# plt.title(r'function of $b_{\mathrm{irr}}$ scaling factor $\theta$', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)

'Part 1b: Sampling the standard white noise'

#We define a larger time interval for padding of the convolution, so that the convolution on the time interval Time is accurate
pad = np.arange(0, T+10/np.sqrt(beta), timestep)+timestep
pad= np.round(pad,decimal_tol)
Time_padded=np.concatenate((Time.min()-np.flip(pad),Time,Time.max()+pad))

#sample N paths of standard white noise on the padded time interval
w_padded = np.random.normal(loc=0, scale=1, size=[Time_padded.size,N])

#N paths of standard white noise on the master time interval
padded_index_min=int(np.where(Time_padded == Time.min())[0])
padded_index_max=int(np.where(Time_padded == Time.max())[0])
w= w_padded[range(padded_index_min,padded_index_max+1),:]





'Figure 2: Plotting standard white noise'

plt.figure(1)
plt.clf()
for n in range(N):
    plt.plot(Time[plot_indices], w[plot_indices, n], linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('Standard white noise', fontsize=16)
plt.title(r'Zoomed', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)

plt.figure(2)
plt.clf()
for n in range(N):
    plt.plot(Time, w[:, n], linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('Standard white noise', fontsize=16)
# plt.title(r'function of $b_{\mathrm{irr}}$ scaling factor $\theta$', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


'1c: Convolution of white noise with a Gaussian kernel'

conv = np.empty(w.shape) # size: Time x N

for t in range(Time.size):
    conv[t,:]= k(Time[t]-Time_padded)@ w_padded*np.sqrt(timestep)
    #Note: I don't understand why this scaling factor of np.sqrt(timestep)
    #gives us the right variance in the end
    #Given the Riemann sum decomposition of the convolution integral
    #we would expect that the scaling here needs to be timestep



'Figure 2: Plotting white noise convolved with Gaussian kernel'


plt.figure(3)
plt.clf()
for n in range(N):  # Iterating over samples of white noise
    plt.plot(Time[plot_indices], conv[plot_indices, n], linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('White noise convolved with Gaussian kernel', fontsize=16)
plt.title(r'Convolution method, Zoomed', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)

plt.figure(4)
plt.clf()
for n in range(N):  # Iterating over samples of white noise
    plt.plot(Time, conv[:, n], linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('White noise convolved with Gaussian kernel', fontsize=16)
plt.title('Convolution method', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)




'''Part 2: Generation of white noise convolved with a Gaussian kernel using generalised coordinates'''

'Part 2a: Derivatives of the autocovariance of the process'

# define symbolic variable
h = sp.symbols('h')

# define symbolic equation for kernel/autocovariance/autocorrelation of the process given UNNORMALISED Gaussian kernel
#kappa = sp.sqrt(sp.pi / (2 * beta)) * sp.exp(-beta / 2 * h * h)

# '---' given NORMALISED Gaussian kernel
kappa = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h)


# autocovariance\autocorrelation of the process
# def kappa(h): # auto covariance at lag h
# beta = 10 ** -1
# return np.sqrt(np.pi/(2*beta))*k(h, beta/2) #numpy version
# return torch.sqrt(torch.pi/(2*beta))*torch.exp(-beta/2*h*h) #torch version

# serial derivatives of the autocovariance at the origin

at = 0
order = 9  # Number of orders of motion

#form_derivs= sampling_generalised_noise.formal_serial_derivs(kappa, h, 2 * order)
#num_derivs = sampling_generalised_noise.num_serial_derivs(kappa, h, at, 2 * order)

'Part 2b: covariance of generalised orders of motion'

#cov_wtilde=sampling_generalised_noise.gen_covariance(kappa, h, at, order)

'Part 2c: sampling from Gaussian with given covariance'

tilde_w0 = sampling_generalised_noise.sample_gen_noise(K=kappa,wrt=h,at=at,order=order,N=N)

'Part 2d: generating trajectories'

w_gen = np.zeros(w.shape)  # white noise generated by generalised coordinates
for i in range(N):
    w_gen[:, i] = Taylor_series.taylor_eval(derivs=tilde_w0[:, i], at=0,Time=Time)

# w_gen2 =np.zeros(w.shape) #Alternative method, may turn out to be more efficient, Not yet working
# for n in range(order+1):
#    w_gen2 += np.outer(tilde_w0[n,:],Time**n/np.math.factorial(n)).T


'Part 2e: plotting trajectories'

plt.figure(5)
plt.clf()
for n in range(N):  # Iterating over samples of white noise
    plt.plot(Time[plot_indices], w_gen[plot_indices, n], linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('White noise convolved with Gaussian kernel', fontsize=16)
plt.title(f'Generalised coordinates, order={order}', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


'''Part 3: tests'''

'Testing the similarity between the two timeseries on a very short time interval'

plot_test = range(-int(3 * timesteps / 100) + int(timesteps / 2), int(3 * timesteps / 100) + int(timesteps / 2))

plt.figure(6)
plt.clf()
for n in range(N):  # Iterating over samples of white noise
    plt.plot(Time[plot_test], conv[plot_test, n], linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('White noise convolved with Gaussian kernel', fontsize=16)
plt.title(f'Convolution method', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)

plt.figure(7)
plt.clf()
for n in range(N):  # Iterating over samples of white noise
    plt.plot(Time[plot_test], w_gen[plot_test, n], linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('White noise convolved with Gaussian kernel', fontsize=16)
plt.title(f'Generalised coordinates, order={order}', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


'Testing the empirical variance of w_t (at each t) for each method'
'Theoretically, it is time independent and should equal: '
'sqrt(pi/2beta) for the unnormalised Gaussian kernel'
'sqrt(beta/2pi) for the normalised Gaussian kernel'


var_conv= np.var(conv,axis=1)
var_gen= np.var(w_gen,axis=1)
var_theo=np.sqrt(beta/(2*np.pi))*np.ones(Time.shape)


plt.figure(8)
plt.clf()
plt.plot(Time[plot_test], var_conv[plot_test], label='Convolution method')
plt.plot(Time[plot_test], var_gen[plot_test], label='Generalised coordinates method')
plt.plot(Time[plot_test], var_theo[plot_test], label='Theory')
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.ylim((0.9*var_theo[0],1.1*var_theo[0]))
plt.legend()
plt.suptitle('Variance of each method over time', fontsize=16)
#plt.title(f'Generalised coordinates, order={order}', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


print(var_theo[0],np.median(var_conv[plot_test]),np.median(var_gen[plot_test]))