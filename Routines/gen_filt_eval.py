'Evaluation of functions and functionals from time series of generalised filtering'

'''Covariance routines for generalised filtering'''

import numpy as np
import sympy as sp
def gen_filt_cov_N_samples(Hess_gen_energy,Time_gf,genmut,genmu,Time,yt_embedded,geny):
    'DID NOT IMPLEMENT CHANGES IN COVARIANCE THAT ARISE FROM THE LOCAL LINEARISATION'

    print('====Gen filt cov====')

    # [dim_x,order_x_pone]=genmu.shape
    N=len(genmut)

    input_sympy = genmu.row_join(geny)
    Hess_lambdify = sp.lambdify(input_sympy, Hess_gen_energy, "numpy")

    gencovt_flatenned=[None]*N

    for n in range(N):
        T = Time_gf[n].size
        shape_gencovt_flatenned = list(Hess_gen_energy.shape) + [T]
        gencovt_flatenned[n]=np.empty(shape_gencovt_flatenned)
        for t in range(T):
            # if t % 10 ** 5 == 0:
            #     print(t, '/', T)
            o_index = np.sum(Time <= t) - 1
            input_numpy = np.concatenate((genmut[n][:, :, t], yt_embedded[:, :, o_index, n]), axis=1)
            gencovt_flatenned[n][:,:,t]=np.linalg.inv(Hess_lambdify(*input_numpy.ravel()))
        'THIS IS INTRODUCED TO NOT COMPUTE SAMPLE 2 TO MAKE THINGS FASTER. CAN REMOVE THIS'
        if N==2: break

    return gencovt_flatenned

def gen_filt_cov_N_samples_order0(Hess_gen_energy,Time_gf,genmut,genmu,Time,yt_embedded,geny):
    # this is only the covariance at order zero for all times and all samples as this is the one we will plot later
    gencovt_flatenned=gen_filt_cov_N_samples(Hess_gen_energy,Time_gf,genmut,genmu,Time,yt_embedded,geny)
    N = len(genmut)
    dim_x = genmu.shape[0]
    gencovt_flatenned_order0=[None]*N
    for n in range(N):
        gencovt_flatenned_order0[n]=gencovt_flatenned[n][:dim_x,:dim_x,:]
        'THIS IS INTRODUCED TO NOT COMPUTE SAMPLE 2 TO MAKE THINGS FASTER. CAN REMOVE THIS'
        if N == 2: break
    return gencovt_flatenned_order0

def meaningful_cov(gencovt):
    N=len(gencovt)
    for n in range(N):
        less_zero= gencovt[n]<0
        gencovt[n][less_zero]=np.nan
        'THIS IS INTRODUCED TO NOT COMPUTE SAMPLE 2 TO MAKE THINGS FASTER. CAN REMOVE THIS'
        if N == 2: break
    return gencovt

def meaningful_std(gencovt):
    N=len(gencovt)
    gencovt=meaningful_cov(gencovt)
    for n in range(N):
        gencovt[n]=np.sqrt(gencovt[n])
        'THIS IS INTRODUCED TO NOT COMPUTE SAMPLE 2 TO MAKE THINGS FASTER. CAN REMOVE THIS'
        if N == 2: break
    return gencovt

'''Free energy (or other function) over time from output of generalised filtering'''


def F_over_t(FE,Time_gf,genmut,genmu,Time,yt_embedded,geny):
    print('====F over time====')

    N=len(genmut)

    'Initialise list of values for the free energy over time'
    FEt=[None]*N

    input_sympy = genmu.row_join(geny)
    FE = sp.lambdify(input_sympy, FE, "numpy")

    for n in range(N):
        T=Time_gf[n].size
        FEt[n]=np.empty(T)
        for t in range(T):
            # this is to get the last observation of data
            o_index = np.sum(Time <= t) - 1
            input_numpy = np.concatenate((genmut[n][:, :, t], yt_embedded[:, :, o_index, n]), axis=1)
            FEt[n][t]=FE(*input_numpy.ravel())
            # if np.isnan(FEt[n][t]):
            #     print(f't{t},n{n}')
        'THIS IS INTRODUCED TO NOT COMPUTE SAMPLE 2 TO MAKE THINGS FASTER. CAN REMOVE THIS'
        if N==2: break

    return FEt
