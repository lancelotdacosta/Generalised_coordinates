'''In this script, we aim to recover the path of least action by minimizing the Lagrangian in a moving frame of reference.
For this, we need the Lagrangian, more specifically it's gradient, and also Dx.
It will then be a matter of using Euler to integrate dot x=Dx-grad L(x)
Therefore, we just need to specify a Flow function defined as f(x)=Dx-grad L(x).'''

import numpy as np
import Routines.gen_coords as gen_coords
import sympy as sp


def flow_gen_gd_Lag(grad_Lagrangian,x,t,order):
    #recall order is the top derivative, so the vector genx is of size o+1 in the order dimension
    #the assumption is that grad_Lagrangian depends on all components of genx
    genx = gen_coords.serial_derivs_xt(x, t, order + 1)  # shape [dim,order+1]
    return flow_gengd_Lag(grad_Lagrangian,genx)

def flow_gengd_Lag(grad_Lagrangian,genx):
    Dgenx = gen_coords.operator_D(genx)         # shape [dim,order+1]
    return Dgenx-grad_Lagrangian


def RK45_gen_flow(tilde_x0,genFlow,genx,Time, tol=1e-2 , hmax=1e-1, hmin=1e-6):
    # tilde_x0, Flow, genx, Time
    # method developed for integrating ONE sample path only
    print('=RK45 integration of generalised flow=')

    [dim_x,order_x_pone]=tilde_x0.shape

    'setup function'
    genF_lambdify = sp.lambdify(genx, genFlow, "numpy")

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

    tilde_xt = np.zeros([dim_x, order_x_pone, 1]) #initialise integration
    tilde_xt[:, :, 0] = tilde_x0
    Times_variable=   np.array([0]) #initialise vector of times

    while t < Tmax:
        'set up timestep'
        if t + h > Tmax:
            h = Tmax - t

        'trapezoidal rule inputs'
        k1 = h * genF_lambdify(*tilde_xt[:,:,-1].ravel())

        input_numpy = tilde_xt[:,:,-1]+ b21 * k1
        k2 = h * genF_lambdify(*input_numpy.ravel())

        input_numpy = tilde_xt[:,:,-1]+ b31 * k1 + b32 * k2
        k3 = h * genF_lambdify(*input_numpy.ravel())

        input_numpy = tilde_xt[:,:,-1] + b41 * k1 + b42 * k2 + b43 * k3
        k4 = h * genF_lambdify(*input_numpy.ravel())

        input_numpy = tilde_xt[:,:,-1] + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4
        k5 = h * genF_lambdify(*input_numpy.ravel())

        input_numpy = tilde_xt[:,:,-1] + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5
        k6 = h * genF_lambdify(*input_numpy.ravel())

        # estimate of truncation error accross each order and dimension
        r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6)

        # take the maximum error along all orders and dimensions
        if r.size > 0:
            r = np.max(r)

        if r<=tol: #in which case we accept the step (tol is the maximum error tolerance per step)
            t = t + h #increase time by timestep
            temp = tilde_xt[:,:,-1] + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 #approximate solution
            Times_variable = np.append(Times_variable, t) #add new time
            if len(Times_variable)%10**3==0:
                print('RK45: step=',len(Times_variable),', time=',t,', T=',Tmax)
            tilde_xt = np.concatenate((tilde_xt, temp.reshape([temp.shape[0],temp.shape[1],1])), 2) #add to timeseries of solutions

        # set up new timestep
        h = 0.9* h * (tol/r)**0.2
        if h > hmax:
            h = hmax
        elif h < hmin:# and Tmax -t >= hmin:
            # raise RuntimeError("Error: Could not converge to the required tolerance.")
            # break
            if Tmax -t > h:
                print("Warning: Could not converge to the required tolerance.")
                return tilde_xt, Times_variable
            h = hmin

    # return linear_interpolation(genmut_variable,Times_variable,Time)
    return tilde_xt, Times_variable

def Euler_gen_flow(tilde_x0,Flow,genx,Time):
    print('====Euler integration of generalised flow====')
    T = Time.size
    [dim,order]=tilde_x0.shape
    tilde_xt = np.empty([dim,order,T])
    tilde_xt[:,:, 0]=tilde_x0
    F_lambdify = sp.lambdify(genx, Flow, "numpy")
    for t in range(T-1):
        if t % 10**5 == 0:
            print(t, '/', T)
        step=Time[t+1]-Time[t]
        tilde_xt[:,:,t+1]=tilde_xt[:,:,t] + step * F_lambdify(*tilde_xt[:,:,t].ravel())
    return tilde_xt



def Euler_gen_flow_slow(tilde_x0,Flow,genx,Time):
    #Unlike the one above this method has not been optimized for speed
    #they both yield the same result though
    print('====Euler integration of generalised flow====')
    T = Time.size
    [dim,order]=tilde_x0.shape
    tilde_xt = np.empty([dim,order,T])
    tilde_xt[:,:, 0]=tilde_x0
    for t in range(T-1):
        if t % 100 == 0:
            print(t, '/', T)
        step=Time[t+1]-Time[t]
        tilde_xt[:,:,t+1]=tilde_xt[:,:,t] + step * gen_coords.gen_coord_eval(Flow,genx,tilde_xt[:,:,t])
    return tilde_xt

























