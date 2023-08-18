'''In this file we compare two alternative ways generating White noise convolved with a Gaussian kernel
1) First, we generate this process in the usual way: We generate white noise and convolve it with a Gaussian kernel
2) Second, we use the method of generalised coordinates, where we first generate the serial derivatives of the process
at the time origin and then sum them in a Taylor series to obtain the process.
'''

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import torch

# plot parameters
lw = 0.5

# Gaussian kernel parameters
beta = 1  # 10 ** -1  # scaling parameter

'''Part 1: Standard generation of white noise convolved with a Gaussian kernel'''

'Part 1a: Generation of standard white noise'

dim = 10 ** 3  # Number of time steps

mean = np.zeros(dim)

cov = np.eye(dim)

N = 100  # Number of white noise sample paths

w = np.random.multivariate_normal(mean, cov, size=N).T

Time = np.linspace(-15, 15, dim)

'Figure 1: Plotting standard white noise'
plot_indices = range(-int(dim / 10) + int(dim / 2), int(dim / 10) + int(
    dim / 2))  # have a smaller plot time for padding at the edges, useful during convolution
# plot_time = Time[] #

plt.figure(0)
plt.clf()
for i in range(N):
    plt.plot(Time[plot_indices], w[plot_indices, i], linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('Standard white noise', fontsize=16)
plt.title(r'Zoomed', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)

plt.figure(1)
plt.clf()
for i in range(N):
    plt.plot(Time, w[:, i], linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('Standard white noise', fontsize=16)
# plt.title(r'function of $b_{\mathrm{irr}}$ scaling factor $\theta$', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


'Part 1b: Convolution with a Gaussian kernel'


#def k(t, beta):  # unnormalised Gaussian kernel
#    return np.exp(-beta * t * t)


def k(t, beta):  # normalised Gaussian kernel
    return np.exp(-beta * t * t)*np.sqrt(beta/np.pi)


# kernel = np.exp(-Time * Time) #Gaussian kernel

'Figure 2: Plotting Gaussian kernel'
plt.figure(2)
plt.clf()
plt.plot(Time[plot_indices], k(Time[plot_indices], beta), linewidth=lw)
# plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
# plt.legend()
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('Gaussian kernel', fontsize=16)
# plt.title(r'function of $b_{\mathrm{irr}}$ scaling factor $\theta$', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)

'Convolution'

conv = np.empty(w.shape)

for n in range(N):  # Iterating over samples of white noise
    # conv[:,i]= np.convolve(w[:,i],kernel,mode='valid')
    for i in range(dim):
        conv[i, n] = w[:, n] @ k(Time[i] - Time, beta)

# conv /= len(kernel)


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
def formal_serial_derivatives(F, wrt, order):  # formal symbolic expressions
    form_derivs = []
    for n in range(order + 1):
        form_derivs.append(F)
        F = F.diff(wrt)
    return form_derivs


def serial_derivatives(F, wrt, at, order):  # numerical expressions evaluated at 'at'
    form_derivs = formal_serial_derivatives(F, wrt, order)
    num_derivs = np.zeros(order + 1)
    for n in range(order + 1):
        num_derivs[n] = form_derivs[n].subs(wrt, at)
    return num_derivs


at = 0
order = 9  # Number of orders of motion

form_derivs = formal_serial_derivatives(kappa, h, 2 * order)
num_derivs = serial_derivatives(kappa, h, at, 2 * order)

'Part 2b: covariance of generalised orders of motion'

cov_wtilde = np.empty([order + 1, order + 1])
for n in range(order + 1):
    for m in range(order + 1):
        # print(n+m)
        cov_wtilde[n, m] = (-1) ** n * num_derivs[n + m]

'Part 2c: sampling from Gaussian with given covariance'

tilde_w0 = np.random.multivariate_normal(np.zeros(order + 1), cov_wtilde, size=N).T

'Part 2d: generating trajectories'

w_gen = np.zeros(w.shape)  # white noise generated by generalised coordinates
for i in range(N):
    for n in range(order + 1):
        w_gen[:, i] += (tilde_w0[n, i] * Time ** n) / np.math.factorial(n)

# w_gen2 =np.zeros(w.shape) #Alternative method, supposed to be more efficient, Not yet working
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

plot_test = range(-int(3 * dim / 100) + int(dim / 2), int(3 * dim / 100) + int(dim / 2))

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
plt.legend()
plt.suptitle('Variance of each method over time', fontsize=16)
plt.yscale('log')
#plt.title(f'Generalised coordinates, order={order}', fontsize=14)
# plt.yscale('log')
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)
