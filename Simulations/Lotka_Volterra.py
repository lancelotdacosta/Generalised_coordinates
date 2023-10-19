import numpy as np
import sympy as sp
from Routines import Taylor_series
from Routines import sampling_generalised_noise
from Routines import colourline
import matplotlib.pyplot as plt
from integration import Euler
from integration import gen_coord, lin_gen_coord, gen_coord_test
from Routines import convolving_white_noise
import matplotlib.cm as cm


'''Part 0: specifying the ￿￿￿￿Lotka Volterra system'''

'Part 0a: Specifying the parameters of the flow'

dim =2 #State space dimension

#parameters of Lorentz system
a = 1
b =1
c =1
d = 1
#only a/d affects the nature of the solution up to rescaling of space and time


'Part 0a: Specifying the numpy form of the flow'

def flow(x):
    y=np.empty(x.shape)
    y[0,:]= a*x[0,:]-b*x[0,:]*x[1,:] #rate of change of prey
    y[1, :] = c*x[0,:]*x[1,:]-d*x[1,:] #rate of change of predators
    return y

'Part 0a: Specifying the sympy form of the flow'

#symbolic time variable
t = sp.Symbol("t")

#symbolic space variable
x= sp.Matrix([sp.Function("x0")(t),
              sp.Function("x1")(t)])

#Flow of Lotka Volterra system
F=sp.Matrix([a*x[0]-b*x[0]*x[1],
             c*x[0]*x[1]-d*x[1]])


'Part 0b: Specifying the kernel of the noise process'

h = sp.symbols('h')
beta=1

K = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h) # NORMALISED Gaussian kernel given as a symbolic expression


'''Part 1: Specifying parameters of integration'''

N = 128 #number of sample paths

#initial condition
x0 = 1.5*np.ones([dim, N]) #initial condition representing the proportion of prey and predators


timesteps=10**4
Time = np.linspace(0,16,timesteps) #time grid over which we integrate


'Part 1a: getting the serial derivatives of the noise process'

order = 10

at=0

epsilon = 0.1 # scaling of noise
tilde_w0= epsilon*sampling_generalised_noise.sample_gen_noise_nd(K=K,wrt=h,at=at,order= order,N=N,dim=dim) #Generalised fluctuations at time zero



'''Part 2: Generalised coordinates integration'''

'Part 2a: Getting serial derivatives of solution for each sample at time <at> '

tilde_x0=gen_coord.sol_gen_coords(F,x,t,x0,tilde_w0)
tilde_x0_test=gen_coord_test.sol_gen_coords(F,x,t,x0,tilde_w0)

'Part 2b: Generating sample paths of solution'


xt= np.empty([dim, Time.size, N])

for n in range(N):
    xt[:,:, n] = Taylor_series.taylor_eval_nd(derivs=tilde_x0[:,:,n] ,at=0,Time=Time)


'Part 2c: Getting serial derivatives of solution for each sample at time <at> using linearised method'

tilde_x0_lin = lin_gen_coord.sol_lin_gen_coord(F,x,t,x0,tilde_w0)

'Part 2d: Generating sample paths of solution'


xt_lin= np.empty([dim, Time.size, N])

for n in range(N):
    xt_lin[:,:, n] = Taylor_series.taylor_eval_nd(derivs=tilde_x0_lin[:,:,n] ,at=0,Time=Time)





'''Part 3: Euler integration where white noise is convolved by a Gaussian kernel'''

'''Part 3a: Sampling white noise is convolved by a Gaussian kernel'''

wt_conv = epsilon*convolving_white_noise.white_noise_conv_Gaussian_kernel_nd(dim,Time, N,beta)

'''Part 3b: Euler integration with the above noise samples'''

xt_Euler_conv= Euler.Euler_integration(x0=x0,f=flow, wt=wt_conv,Time=Time)


'''Part 4: Path of least action'''


'Part 4a: getting the path of least action, ie path in the absence of noise'


xt_least= Euler.Euler_integration(x0=x0,f=flow, wt=np.zeros([dim,Time.size,N]),Time=Time)


'''Part 6: Plots'''

plot_timesteps=min(2*10**3,Time.size)
plot_indices=range(min(plot_timesteps,timesteps))
lw=0.5
alpha=0.3
c_prey= cm.get_cmap('Blues')
c_predator= cm.get_cmap('Reds')
c_2D = cm.get_cmap('cool')

xmax_2D= xt_least[0,plot_indices,:].max()
xmin_2D = xt_least[0,plot_indices,:].min()
deltax=xmax_2D-xmin_2D
xlim_2D=[xmin_2D-5*deltax,xmax_2D+5*deltax]
xlim_2D=[-1,2]
ymax_2D= xt_least[1,plot_indices,:].max()
ymin_2D = xt_least[1,plot_indices,:].min()
deltay=ymax_2D-ymin_2D
ylim_2D=[ymin_2D-5*deltay,ymax_2D+5*deltay]
ylim_2D=[0,2.5]


'Part 6a: 2D plot of gen coord Zig-zag method sample paths'

#n=1 #trajectory to plot

plt.figure(1)
print('Plot 1')
plt.clf()
plt.suptitle(f'Lotka Volterra (2D view)', fontsize=16)
plt.title(f'Zig-zag method, order={tilde_x0.shape[1]}', fontsize=14)
for n in range(N):
    #plt.plot(xt[0, plot_indices, n], xt[1, plot_indices, n], linewidth=lw, alpha=alpha)
    colourline.plot_cmap(xt[0, plot_indices, n], xt[1, plot_indices, n], lw=lw,alpha=alpha,cmap=c_2D,crev=True)
plt.plot(xt_least[0,plot_indices,0], xt_least[1,plot_indices,0], color='gray',linestyle=':')
plt.xlim(xlim_2D)
plt.ylim(ylim_2D)
#plt.ylim(top=12,bottom=-5)
plt.xlabel(r'prey')
plt.ylabel(r'predator')
plt.savefig("Lotka-Volterra_2D_zigzag.png", dpi=100)


'Part 6b: 2D plot of linearised gen coord Zig-zag method sample paths'

#n=1 #trajectory to plot

plt.figure(2)
print('Plot 2')
plt.clf()
plt.suptitle(f'Lotka Volterra (2D view)', fontsize=16)
plt.title(f'Linearised zig-zag method, order={tilde_x0.shape[1]}', fontsize=14)
for n in range(N):
    #plt.plot(xt[0, plot_indices, n], xt[1, plot_indices, n], linewidth=lw, alpha=alpha)
    colourline.plot_cmap(xt_lin[0, plot_indices, n], xt_lin[1, plot_indices, n], lw=lw,alpha=alpha,cmap=c_2D,crev=True)
plt.plot(xt_least[0,plot_indices,0], xt_least[1,plot_indices,0], color='gray',linestyle=':')
plt.xlim(xlim_2D)
plt.ylim(ylim_2D)
#plt.ylim(top=12,bottom=-5)
plt.xlabel(r'prey')
plt.ylabel(r'predator')
plt.savefig("Lotka-Volterra_2D_linzigzag.png", dpi=100)

'Part 6c: 2D plot of classical Euler-conv method'

plt.figure(3)
print('Plot 3')
plt.clf()
for n in range(N):  # Iterating over samples of white noise
    #plt.plot(xt_Euler_gen[0,plot_indices, n], xt_Euler_gen[1,plot_indices, n], linewidth=lw,alpha=alpha)
    colourline.plot_cmap(xt_Euler_conv[0, plot_indices, n], xt_Euler_conv[1, plot_indices, n], lw=lw,alpha=alpha,cmap=c_2D,crev=True)
plt.plot(xt_least[0,plot_indices,0], xt_least[1,plot_indices,0], color='gray',linestyle=':')
plt.suptitle('Lotka Volterra (2D view)', fontsize=16)
plt.title(f'Classical Euler-Conv method', fontsize=14)
plt.xlim(xlim_2D)
plt.ylim(ylim_2D)
# plt.ylim(top=12,bottom=-5)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)
plt.xlabel(r'prey')
plt.ylabel(r'predator')
plt.savefig("Lotka-Volterra_2D_Eulerconv.png", dpi=100)



'Part 6d: 1D plot of gen coord Zig-zag method sample paths'
ylim_1D=(-0,2.5)
#n=1 #trajectory to plot

plt.figure(4)
print('Plot 4')
plt.clf()
plt.suptitle(f'Lotka Volterra', fontsize=16)
plt.title(f'Zig-zag method, order={tilde_x0.shape[1]}', fontsize=14)
for n in range(N):
    #plt.plot(xt[0, plot_indices, n], xt[1, plot_indices, n], linewidth=lw, alpha=alpha)
    colourline.plot_cmap(Time[plot_indices], xt[0, plot_indices, n], lw=lw,alpha=alpha,cmap=c_prey,crev=True)
    colourline.plot_cmap(Time[plot_indices], xt[1, plot_indices, n], lw=lw, alpha=alpha,cmap=c_predator,crev=True)
plt.plot(Time[:], xt_least[0,:,0], color='blue',linestyle=':')
plt.plot(Time[:], xt_least[1,:,0], color='red',linestyle=':')
plt.ylim(ylim_1D)
plt.xscale('log')
plt.xlabel(r'Time')
plt.savefig("Lotka-Volterra_1D_zigzag.png", dpi=100)


'Part 6d: 1D plot of gen coord Zig-zag method sample paths'

plt.figure(5)
print('Plot 5')
plt.clf()
plt.suptitle(f'Lotka Volterra', fontsize=16)
plt.title(f'Linearised zig-zag method, order={tilde_x0.shape[1]}', fontsize=14)
for n in range(N):
    #plt.plot(xt[0, plot_indices, n], xt[1, plot_indices, n], linewidth=lw, alpha=alpha)
    colourline.plot_cmap(Time[plot_indices], xt_lin[0, plot_indices, n], lw=lw,alpha=alpha,cmap=c_prey,crev=True)
    colourline.plot_cmap(Time[plot_indices], xt_lin[1, plot_indices, n], lw=lw, alpha=alpha,cmap=c_predator,crev=True)
plt.plot(Time[:], xt_least[0,:,0], color='blue',linestyle=':')
plt.plot(Time[:], xt_least[1,:,0], color='red',linestyle=':')
plt.ylim(ylim_1D)
plt.xscale('log')
plt.xlabel(r'Time')
plt.savefig("Lotka-Volterra_1D_linzigzag.png", dpi=100)




'Part 6e: 1D plot of classical Euler-conv method'

#n=1 #trajectory to plot

plt.figure(6)
print('Plot 6')
plt.clf()
plt.suptitle(f'Lotka Volterra', fontsize=16)
plt.title(f'Classical Euler-Conv method', fontsize=14)
for n in range(N):
    #plt.plot(xt[0, plot_indices, n], xt[1, plot_indices, n], linewidth=lw, alpha=alpha)
    colourline.plot_cmap(Time[plot_indices], xt_Euler_conv[0, plot_indices, n], lw=lw,alpha=alpha,cmap=c_prey,crev=True)
    colourline.plot_cmap(Time[plot_indices], xt_Euler_conv[1, plot_indices, n], lw=lw, alpha=alpha,cmap=c_predator,crev=True)
plt.plot(Time[:], xt_least[0,:,0], color='blue',linestyle=':')
plt.plot(Time[:], xt_least[1,:,0], color='red',linestyle=':')
#plt.xlim(right=12,left=-5)
plt.ylim(ylim_1D)
plt.xscale('log')
plt.xlabel(r'Time')
plt.savefig("Lotka-Volterra_1D_Eulerconv.png", dpi=100)





'Part 6e: 2D plot of white noise sample paths convolved with a Gaussian kernel'

plt.figure(7)
print('Plot 7')
plt.clf()
for n in range(N):  # Iterating over samples of white noise
    #plt.plot(wt[0,plot_indices, n], wt[1,plot_indices, n], linewidth=lw,alpha=0.5)
    colourline.plot_cool(wt_conv[0, plot_indices, n].reshape(plot_timesteps), wt_conv[1, plot_indices, n].reshape(plot_timesteps), lw=lw,alpha=0.5)
plt.suptitle('2D Coloured noise', fontsize=16)
plt.title(f'Standard convolution method', fontsize=14)
#plt.xlim(right=2.5,left=-2.5)
#plt.ylim(top=2.5,bottom=-2.5)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)


'Part 6f: 1D plot of convolved white noise generalised coordinate sample paths'

plt.figure(8)
print('Plot 8')
plt.clf()
for n in range(N):
    #plt.plot(Time[plot_indices], wt[1, plot_indices, n] / wt[0, plot_indices, n], linewidth=lw, alpha=1,color='b')
    for d in range(dim):  # Iterating over samples of white noise
        plt.plot(Time[plot_indices], wt_conv[d,plot_indices, n], linewidth=lw,alpha=0.5)
plt.xlabel(r'Time $t$')
plt.ylabel(r'$w_t$')
plt.suptitle('Coloured noise', fontsize=16)#
plt.title(f'Standard convolution method', fontsize=14)
#plt.ylim(top=10,bottom=-10)
# plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)




'''tests'''
# print(np.sum(np.abs(tilde_x0_lin-tilde_x0)[:,3,:])) #should be equal up to order 2 included
res0=np.abs(tilde_x0_lin-tilde_x0)[:,0,:] #should be zeroes
res1=np.abs(tilde_x0_lin-tilde_x0)[:,1,:] #should be zeroes
res2=np.abs(tilde_x0_lin-tilde_x0)[:,2,:] #should be zeroes
res3=np.abs(tilde_x0_lin-tilde_x0)[:,3,:] #should be non zeroes
res4=np.abs(tilde_x0_lin-tilde_x0)[:,4,:] #should be non zeroes
print(np.sum(res0),np.sum(res1),np.sum(res2),np.sum(res3),np.sum(res4))

res= xt-xt_lin
print(np.sum(np.abs(res)))