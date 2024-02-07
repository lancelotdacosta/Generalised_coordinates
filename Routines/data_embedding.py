'data embedding procedures for generalised filtering'

import numpy as np
import sympy as sp

def findiff_embedding(yt, Time, order_y):
    'embedding data in generalised coordinates through a finite difference method'

    dim_y, timesteps, N= yt.shape
    yt_embedded = np.zeros([dim_y, order_y + 1, timesteps, N])

    yt_embedded[:, 0, :, :] = yt

    if order_y>0:
        for o in range(1, order_y):
            for tau in range(1, timesteps):
                yt_embedded[:, o, tau, :] = (yt_embedded[:, o - 1, tau, :] - yt_embedded[:, o - 1, tau - 1, :]) / (
                        Time[tau] - Time[tau - 1])

    return yt_embedded


def Taylor_coeffs(dt,order_y):
    # Build taylor expansion coefficients
    coeffs=np.ones([order_y+1])
    for o in range(1,order_y+1):
        coeffs[o]=coeffs[o-1]*dt/o
    return coeffs

def Taylor_matrix(dt, order_y):
    # build taylor expansion matrix
    T_mx = np.empty([order_y + 1,order_y + 1])
    for o in range(order_y + 1):
        T_mx[o] = Taylor_coeffs(dt*(order_y-o),order_y)
    return T_mx

def Taylor_embedding(yt, Time, order_y):
    'embedding data in generalised coordinates through a Taylor embedding method'

    dim_y, timesteps, N= yt.shape
    dt = Time[1] - Time[0]

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

    return yt_embedded

def embed_data(yt, Time, order_y,meth_data):
    if meth_data == 'findiff':
        yt_embedded = findiff_embedding(yt, Time, order_y)
    elif meth_data=='Taylor':
        yt_embedded = Taylor_embedding(yt, Time, order_y)
    else:
        raise TypeError("Data embedding method not supported")
    return yt_embedded




