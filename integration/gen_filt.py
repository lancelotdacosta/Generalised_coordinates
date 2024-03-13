'''Integration routines for generalised filtering'''
import numpy as np
import sympy as sp



def Euler_gen_filt(genFlow,Time,yt_embedded,geny,genmu):
    # method developed for filtering ONE sample path only
    # print('=Euler integration for generalised filtering=')

    [dim_x,order_x_pone]=genmu.shape
    [dim_y,order_y_pone,T]=yt_embedded.shape

    'Initialise means'
    genmut = np.zeros([dim_x, order_x_pone, T])
    # genmut[:, :, 0] = yt_embedded[:, :, 0]

    'setup function'
    input_sympy= genmu.row_join(geny)
    genF_lambdify = sp.lambdify(input_sympy, genFlow, "numpy")

    for t in range(1,T):
        if t % 10**5 == 0:
             print(t, '/', T)
        step=Time[t]-Time[t-1]
        input_numpy = np.concatenate((genmut[:,:,t-1],yt_embedded[:,:,t-1]),axis=1)
        genmut[:,:,t]=genmut[:,:,t-1] + step * genF_lambdify(*input_numpy.ravel())
    return genmut

def Heun_gen_filt(genFlow,Time,yt_embedded,geny,genmu):
    # method developed for filtering ONE sample path only
    # print('=Heun integration for generalised filtering=')

    [dim_x,order_x_pone]=genmu.shape
    [dim_y,order_y_pone,T]=yt_embedded.shape

    'Initialise means'
    genmut = np.zeros([dim_x, order_x_pone, T])
    # genmut[:, :, 0]=yt_embedded[:, :, 0]

    'setup function'
    input_sympy= genmu.row_join(geny)
    genF_lambdify = sp.lambdify(input_sympy, genFlow, "numpy")

    for t in range(1,T):
        # if t % 10**5 == 0:
        #      print(t, '/', T)
        h=Time[t]-Time[t-1]
        input_numpy = np.concatenate((genmut[:,:,t-1],yt_embedded[:,:,t-1]),axis=1)
        temp_F=genF_lambdify(*input_numpy.ravel())
        temp_genmut = genmut[:,:,t-1] + h * temp_F
        input_numpy= np.concatenate((temp_genmut,yt_embedded[:,:,t-1]),axis=1)
        genmut[:, :, t] = (genmut[:, :, t - 1] + \
                           h/2 * (genF_lambdify(*input_numpy.ravel())+temp_F))
    return genmut


def linear_interpolation(genmut_variable,Times_variable,Time):
    # this function linearly interpolates the solution to the RK45 algorithm, ie (genmut_variable,Times_variable)
    # to obtain the solution on a different, eg coarser, time grid, ie Time.
    # print('=Linear interpolation of RK45 solution=')

    [dim_x,order_x_pone]=genmut_variable.shape[:2]
    T=Time.shape[0]

    'Initialise means'
    genmut = np.empty([dim_x, order_x_pone, T])

    for i in range(T):
        'find time in time variable'
        t= Time[i]
        index = np.sum(Times_variable <= t)-1 #index in Times_variable
        tau = Times_variable[index]  #preceding time in Times_variable
        'find time in time variable'
        if tau == t:
            genmut[:,:,i]=genmut_variable[:,:,index]
        elif tau < t:
            # index2 = np.argmax(Times_variable  t)
            nextau= Times_variable[index+1] #subsequent time in Times_variable
            genmut[:, :, i] = (genmut_variable[:, :, index]*(t-tau)+genmut_variable[:, :, index+1]*(nextau-t))/(nextau-tau)

    return genmut


def RK45_gen_filt(genFlow,Time,yt_embedded,geny,genmu, tol=1e-2 , hmax=1e-1, hmin=1e-6):
    # method developed for filtering ONE sample path only
    # print('=RK45 integration for generalised filtering=')

    [dim_x,order_x_pone]=genmu.shape
    # [dim_y,order_y_pone,T]=yt_embedded.shape

    'setup function'
    input_sympy= genmu.row_join(geny)
    genF_lambdify = sp.lambdify(input_sympy, genFlow, "numpy")

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

    genmut_variable = np.zeros([dim_x, order_x_pone, 1]) #yt_embedded[:,:,0] #initialise means
    Times_variable=   np.array([0]) #initialise vector of times

    while t < Tmax:

        'Find the current timestep in the original time grid'
        # this is to get the last observation of data
        o_index= np.sum(Time <= t)-1

        'set up timestep'
        if t + h > Tmax:
            h = Tmax - t

        'trapezoidal rule inputs'
        input_numpy = np.concatenate((genmut_variable[:,:,-1], yt_embedded[:,:,o_index]),axis=1)
        k1 = h * genF_lambdify(*input_numpy.ravel())

        input_numpy = np.concatenate((genmut_variable[:,:,-1]+ b21 * k1,\
                                       yt_embedded[:,:,o_index]),axis=1)
        k2 = h * genF_lambdify(*input_numpy.ravel())

        input_numpy = np.concatenate((genmut_variable[:,:,-1]+ b31 * k1 + b32 * k2,\
                                       yt_embedded[:,:,o_index]),axis=1)
        k3 = h * genF_lambdify(*input_numpy.ravel())

        input_numpy = np.concatenate((genmut_variable[:, :, -1] + b41 * k1 + b42 * k2 + b43 * k3, \
                                       yt_embedded[:, :, o_index]), axis=1)
        k4 = h * genF_lambdify(*input_numpy.ravel())

        input_numpy = np.concatenate((genmut_variable[:, :, -1] + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, \
                                       yt_embedded[:, :, o_index]), axis=1)
        k5 = h * genF_lambdify(*input_numpy.ravel())

        input_numpy = np.concatenate((genmut_variable[:, :, -1] + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, \
                                       yt_embedded[:, :, o_index]), axis=1)
        k6 = h * genF_lambdify(*input_numpy.ravel())

        # estimate of truncation error accross each order and dimension
        r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6)

        # take the maximum error along all orders and dimensions
        if r.size > 0:
            r = np.max(r)

        if r<=tol: #in which case we accept the step (tol is the maximum error tolerance per step)
            t = t + h #increase time by timestep
            temp = genmut_variable[:, :, -1] + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 #estimate of solution
            Times_variable = np.append(Times_variable, t)
            genmut_variable = np.concatenate((genmut_variable, temp.reshape([temp.shape[0],temp.shape[1],1])), 2)

        # set up new timestep
        h = 0.9* h * (tol/r)**0.2
        if h > hmax:
            h = hmax
        elif h < hmin:# and Tmax -t >= hmin:
            # raise RuntimeError("Error: Could not converge to the required tolerance.")
            # break
            if Tmax -t > h:
                print("Warning: Could not converge to the required tolerance.")
            h = hmin

    # return linear_interpolation(genmut_variable,Times_variable,Time)
    return genmut_variable, Times_variable

# def adaptive_Euler_gen_filt(genFlow,Time,yt_embedded,geny,genmu):
#     # method developed for filtering ONE sample path only
#     print('=adaptive Euler integration for generalised filtering=')
#
#     [dim_x,order_x_pone]=genmu.shape
#     [dim_y,order_y_pone,T]=yt_embedded.shape
#
#     'Initialise means'
#     genmut = np.zeros([dim_x, order_x_pone, T])
#
#     'setup function'
#     input_sympy= genmu.row_join(geny)
#     genF_lambdify = sp.lambdify(input_sympy, genFlow, "numpy")
#
#     'parameters for adaptive step size'
#     time=0 #continuous time
#     prev_t=0
#     t=0
#     iters=0
#     tol=10*Time.size
#     genmu_inst=genmut[:,:,0] #instantaneous genmu
#     genmu_inst_prev = genmut[:, :, 0]
#     while time < Time[T]:
#         # figure out between which indices we are in the time interval (t<=time<t+1)
#         # also figures out whether we just went past a t (indicated by 'update')
#         indices=Time<=time
#         new_t=np.sum(indices)-1
#         if new_t>t:
#             update = True
#         elif new_t==t:
#             update = False
#         else:
#             raise TypeError('update')
#         t = new_t
#         # prepare the input for the flow
#         input_numpy = np.concatenate((genmu_inst,yt_embedded[:,:,t]),axis=1)
#         # flow output
#         F_output=genF_lambdify(*input_numpy.ravel())
#         # define time step size
#         step = (Time[t+1] - Time[t])/np.sum(np.abs(F_output))
#         # update
#         genmu_inst = genmu_inst_prev + step * F_output
#         # if need to update genmut
#         if update:
#             for i in range(prev_t,t):
#                 genmut[:, :, t - 1]=genmu_inst_prev
#         'TODO THIS NEEDS A RETHINK'
#
#     for t in range(1,T):
#         step = Time[t] - Time[t - 1]
#         input_numpy = np.concatenate((genmut[:,:,t-1],yt_embedded[:,:,t-1]),axis=1)
#         genmut[:,:,t]=genmut[:,:,t-1] + step * genF_lambdify(*input_numpy.ravel())
#     return genmut


def integrator_gen_filt_N_samples(genFlow,Time,yt_embedded,geny,genmu,methint='Euler',tol=1e-2):
    # [dim_x,order_x_pone]=genmu.shape
    [dim_y,order_y_pone,T,N]=yt_embedded.shape

    'Initialise means'
    # genmut = np.empty([dim_x, order_x_pone, T,N])
    genmut=[None] * N

    'Initialise time grid'
    # some integration methods have an adaptive step size and return their own time grid
    Time_gf=[None] * N

    'Do generalised filtering for each sample'
    for n in range(N):
        if n>2:
            print(f'---Sample {n}---')
        if methint== 'Euler':
            genmut[n] = Euler_gen_filt(genFlow,Time,yt_embedded[:,:,:,n],geny,genmu)
            Time_gf[n]=Time
        # elif methint=='adaptive_Euler':
        #     genmut[:, :, :, n] = adaptive_Euler_gen_filt(genFlow, Time, yt_embedded[:, :, :, n], geny, genmu)
        elif methint == 'Heun':
            genmut[n] = Heun_gen_filt(genFlow, Time, yt_embedded[:, :, :, n], geny, genmu)
            Time_gf[n]=Time
        elif methint=='RK45':
            hmax=np.max(Time[1:]-Time[:-1])
            genmut[n],Time_gf[n] = RK45_gen_filt(genFlow, Time, yt_embedded[:, :, :, n], geny, genmu, hmax=hmax,tol=tol)
        else: # If an exact match is not confirmed, this last case will be used if provided
            raise TypeError('integrator not yet supported')

        'THIS IS TO NOT COMPUTE SAMPLE 2 TO MAKE THINGS FASTER. CAN REMOVE THIS'
        if N==2: break

    return genmut, Time_gf
