import numpy as np
import sympy as sp
from Routines import Taylor_series
from Routines import sampling_generalised_noise
from Routines import colourline
import matplotlib.pyplot as plt
from integration import Euler
from integration import gen_coord
from scipy.linalg import expm




'''Part 0: specifying the ￿￿￿￿Lorenz system'''


'Part 0a: Specifying the parameters of the flow'

dim =3 #State space dimension

#parameters of Lorentz system
sigma = 10
rho=28
beta=8/3

'Part 0a: Specifying the numpy form of the flow'
def flow(x):
    y=np.empty(x.shape)
    y[0,:]= sigma*(x[1,:]-x[0,:])
    y[1, :] = x[0,:]*(rho-x[2,:])-x[1,:]
    y[2, :] = x[0,:]*x[1,:]-beta*x[2,:]
    return y

#print(flow(np.array([1,2,3]).reshape([3,1])))

'Part 0a: Specifying the sympy form of the flow'

#symbolic time variable
t = sp.Symbol("t")

#symbolic space variable
x= sp.Matrix([sp.Function("x0")(t),
              sp.Function("x1")(t),
              sp.Function("x2")(t)])

#Flow of Lorentz system
F=sp.Matrix([sigma*(x[1]-x[0]),
             x[0] * (rho-x[2]) - x[1],
             x[0]* x[1] - beta * x[2]])



'Part 0b: Specifying the kernel of the noise process'

h = sp.symbols('h')
beta=1

K = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h) # NORMALISED Gaussian kernel given as a symbolic expression


'''Part 1: Specifying parameters of integration'''

N = 512 #number of sample paths


#initial condition
x0_mean =np.array([0,0,27.5])
#x0 = np.random.multivariate_normal(mean=x0_mean, cov=10*np.eye(dim), size=N).T #initial condition
x0_scale=np.array([15,15,12.5])
x0= np.diag(x0_scale)@np.random.uniform(low=-1.0, high=1.0, size=(dim,N)) + np.tile(x0_mean,N).reshape([N,dim]).T


#time
timesteps=10**3

Time = np.linspace(0,10,timesteps) #time grid over which we integrate


'Part 1a: getting the serial derivatives of the noise process'

order = 10

at=0

epsilon=1 #scaling of noise
tilde_w0= epsilon*sampling_generalised_noise.sample_gen_noise_nd(K=K,wrt=h,at=at,order= order,N=N,dim=dim) #Generalised fluctuations at time zero


'''Part 2: Generalised coordinates integration'''

'Part 2a: Getting serial derivatives of solution for each sample at time at '

tilde_x0=gen_coord.sol_gen_coords(F,x,t,x0,tilde_w0)


'Part 2b: Generating sample paths of solution'


xt= np.empty([dim, Time.size, N])

for n in range(N):
    xt[:,:, n] = Taylor_series.taylor_eval_nd(derivs=tilde_x0[:,:,n] ,at=0,Time=Time)


'''Part 4: Path of least action'''


'Part 4a: getting the path of least action, ie path in the absence of noise'


xt_least= Euler.Euler_integration(x0=x0,f=flow, wt=np.zeros([dim,Time.size,N]),Time=Time)



'''Part 5: Plots'''

'Plot parameters'

N_plot= min(N, 10**3) #number of samples to plot
timesteps_plot=min(timesteps, 10**1)
cmap = plt.cm.get_cmap('binary')
#cmap = plt.cm.get_cmap('plasma')
lw=0.5
alpha=0.5

'Part 5a: 3D paths of least action'

fig = plt.figure(1)
print('====Plot least action paths 3D space===')
plt.clf()
ax = plt.axes(projection='3d')
for n in range(N_plot):
    if n % (10 ** 2) == 0:
        print('n', n, '/', N_plot)
    for t in range(1, timesteps_plot):
        if t %(10**2) ==0:
            print('t',t)
        ax.plot3D(xt_least[0, t-1:t+1, n], xt_least[1, t-1:t+1, n], xt_least[2, t-1:t+1, n], c=cmap(1-t/timesteps_plot), lw = lw,alpha=alpha)
ax.axes.set_xlabel('x')
ax.axes.set_ylabel('y')
ax.axes.set_zlabel('z')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
plt.suptitle('Paths of least action')
plt.title(f'3D')
#ax.legend() #loc="upper right"


'Part 5b: 2D paths of least action (projections)'

plt.figure(2)
plt.clf()
print('====Plot least action paths xy plane===')
for n in range(N_plot):
    if n % (10 ** 2) == 0:
        print('n', n, '/', N_plot)
    colourline.plot_cmap(xt_least[0, :timesteps_plot, n], xt_least[1, :timesteps_plot, n], cmap=cmap, lw=lw,alpha=alpha)
plt.suptitle('Paths of least action ', fontsize=16)
plt.title(f'x-y plane', fontsize=14)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')



plt.figure(3)
plt.clf()
print('====Plot least action paths xz plane===')
for n in range(N_plot):
    if n % (10 ** 2) == 0:
        print('n', n, '/', N_plot)
    colourline.plot_cmap(xt_least[0, :timesteps_plot, n], xt_least[2, :timesteps_plot, n], cmap=cmap, lw=lw,alpha=alpha)
plt.suptitle('Paths of least action', fontsize=16)
plt.title(f'x-z plane', fontsize=14)
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')



plt.figure(4)
plt.clf()
print('====Plot least action paths yz plane===')
for n in range(N_plot):
    if n % (10 ** 2) == 0:
        print('n', n, '/', N_plot)
    colourline.plot_cmap(xt_least[1, :timesteps_plot, n], xt_least[2, :timesteps_plot, n], cmap=cmap, lw=lw,alpha=alpha)
plt.suptitle('Paths of least action', fontsize=16)
plt.title(f'y-z plane', fontsize=14)
plt.xlabel(r'$y$')
plt.ylabel(r'$z$')



'Part 5c: 3D generalised coordinate paths'


fig = plt.figure(5)
print('====Plot generalised coordinate paths 3D space===')
plt.clf()
ax = plt.axes(projection='3d')
for n in range(N_plot):
    if n % (10 ** 2) == 0:
        print('n', n, '/', N_plot)
    for t in range(1, timesteps_plot):
        if t %(10**2) ==0:
            print('t',t)
        ax.plot3D(xt[0, t-1:t+1, n], xt[1, t-1:t+1, n], xt[2, t-1:t+1, n], c=cmap(1-t/timesteps_plot), lw = lw,alpha=alpha)
ax.axes.set_xlim3d(xlim)
ax.axes.set_ylim3d(ylim)
ax.axes.set_zlim3d(zlim)
ax.axes.set_xlabel('x')
ax.axes.set_ylabel('y')
ax.axes.set_zlabel('z')
plt.suptitle('Generalised coordinate sample paths')
plt.title(f'3D')


'Part 5d: 3D generalised coordinate paths'

plt.figure(6)
plt.clf()
print('====Plot generalised coordinate paths xy plane===')
for n in range(N_plot):
    if n % (10 ** 2) == 0:
        print('n', n, '/', N_plot)
    colourline.plot_cmap(xt[0, :timesteps_plot, n], xt[1, :timesteps_plot, n], cmap=cmap, lw=lw,alpha=alpha)
plt.suptitle('Generalised coordinate sample paths', fontsize=16)
plt.title(f'x-y plane', fontsize=14)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.xlim(xlim)
plt.ylim(ylim)

plt.figure(7)
plt.clf()
print('====Plot generalised coordinate paths xz plane===')
for n in range(N_plot):
    if n % (10 ** 2) == 0:
        print('n', n, '/', N_plot)
    colourline.plot_cmap(xt[0, :timesteps_plot, n], xt[2, :timesteps_plot, n], cmap=cmap, lw=lw,alpha=alpha)
plt.suptitle('Generalised coordinate sample paths', fontsize=16)
plt.title(f'x-z plane', fontsize=14)
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.xlim(xlim)
plt.ylim(zlim)


plt.figure(8)
plt.clf()
print('====Plot generalised coordinate paths yz plane===')
for n in range(N_plot):
    if n % (10 ** 2) == 0:
        print('n', n, '/', N_plot)
    colourline.plot_cmap(xt[1, :timesteps_plot, n], xt[2, :timesteps_plot, n], cmap=cmap, lw=lw,alpha=alpha)
plt.suptitle('Generalised coordinate sample paths', fontsize=16)
plt.title(f'y-z plane', fontsize=14)
plt.xlabel(r'$y$')
plt.ylabel(r'$z$')
plt.xlim(ylim)
plt.ylim(zlim)