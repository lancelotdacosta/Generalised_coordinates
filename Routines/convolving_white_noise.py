import numpy as np

'''Note of caution: this function only produces convolved white noise that is properly normalised in the case of beta=1'''
def white_noise_conv_Gaussian_kernel(Time, N, beta=1):
    '''Input:
    Time = arange of time (expect uniform timesteps)
    N = Number of sample paths
    beta = precision of Gaussian kernel
    Output: convolved white noise on desired time interval'''

    timestep=Time[1]-Time[0]

    'Part 1a: Setting up the convolution kernel'

    #def k(t, beta):  # unnormalised Gaussian kernel
    #    return np.exp(-beta * t * t)

    def Gaussian_kernel(t):  # normalised Gaussian kernel
        return np.exp(-beta * t * t)*np.sqrt(beta/np.pi)

    # kernel = np.exp(-Time * Time) #Gaussian kernel


    'Part 1b: Setting up the padding'

    #We define a larger time interval for padding of the convolution, so that the convolution on the time interval Time is accurate
    pad = np.arange(0, np.max(np.abs(Time))+10/np.sqrt(beta), timestep)+timestep
    #pad= np.round(pad,decimal_tol)
    Time_padded=np.concatenate((Time.min()-np.flip(pad),Time,Time.max()+pad))


    'Part 1c: Setting up the standard white noise'

    #sample N paths of standard white noise on the padded time interval
    w_padded = np.random.normal(loc=0, scale=1, size=[Time_padded.size,N])


    '1d: Convolution with a Gaussian kernel'

    w_conv = np.empty([Time.size,N]) # size: Time x N

    for t in range(Time.size):
        k_ts= Gaussian_kernel(Time[t]-Time_padded) #kernel evaluated at t-s
        w_conv[t,:] =  k_ts @ w_padded * np.sqrt(timestep)
        #print(np.sum(k_ts))
        #Note: I don't understand why this scaling factor of np.sqrt(timestep)
        #gives us the right variance in the end
        #Given the Riemann sum decomposition of the convolution integral
        #we would expect that the scaling here needs to be timestep instead

    return w_conv

def white_noise_conv_Gaussian_kernel_nd(dim, Time, N, beta=1):
    w_conv = np.empty([dim,Time.size, N])
    for d in range(dim):
        w_conv[d, :, :] = white_noise_conv_Gaussian_kernel(Time, N, beta)
    return w_conv


'Test of method: Setting up the time and sample parameters and execute'

# decimal_tol = 2  # decimal tolerance for timesteps
# timestep = 10 ** -decimal_tol  # Number of time steps
#
# # setting up master time interval
# T = 10  # boundary of master time interval
# Time = np.arange(-T, T + timestep, timestep)
# Time = np.round(Time, decimal_tol)
#
# N = 1024 # Number of white noise sample paths
# beta=1
#
# w_conv = white_noise_conv_Gaussian_kernel(Time, N,beta)