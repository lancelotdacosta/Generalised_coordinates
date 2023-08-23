import numpy as np
import sympy as sp
from Routines import Taylor_series
from Routines import sampling_generalised_noise
from Routines import colourline
import matplotlib.pyplot as plt
from integration import Euler, gen_coord
from scipy.linalg import expm




'''Part 0: specifying the ￿￿￿￿Lorenz system'''


'Part 0a: Specifying the flow and its parameters'

dim =3 #State space dimension


s = 10

r=28

b=8/3


def flow(x):
    y=np.empty(x.shape)
    y[0,:]= s*(x[1,:]-x[0,:])
    y[1, :] = x[0,:]*(r-x[2,:])-x[1,:]
    y[2, :] = x[0,:]*x[1,:]-b*x[2,:]
    return y

print(flow(np.array([1,2,3]).reshape([3,1])))



'Part 0b: Specifying the kernel of the noise process'

h = sp.symbols('h')
beta=1

K = sp.sqrt(beta / (2 * sp.pi)) * sp.exp(-beta / 2 * h * h) # NORMALISED Gaussian kernel given as a symbolic expression


'''Part 1: Specifying parameters of integration'''

N = 2 #number of sample paths

x0 = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=N).T #initial condition


timesteps=10**4

Time = np.linspace(0,100,timesteps) #time grid over which we integrate


'Part 1a: getting the serial derivatives of the noise process'

order = 10

at=0

tilde_w0= sampling_generalised_noise.sample_gen_noise_nd(K=K,wrt=h,at=at,order= order,N=N,dim=dim) #Generalised fluctuations at time zero


'''Part 2: Generalised coordinates integration'''
#Input: flow function

'Part 2a: Getting serial derivatives of solution for each sample at time at '

tilde_x0= np.empty([dim, order+1, N])
tilde_x0[:,0,:]=x0

tilde_x0=gen_coord

for i in range(order):
    for n in range(N):
        tilde_x0[:,i+1,n]=


'Part 2b: Generating sample paths of solution'


xt= np.empty([dim, Time.size, N])

for n in range(N):
    xt[:,:, n] = Taylor_series.taylor_eval_nd(derivs=tilde_x0[:,:,n] ,at=0,Time=Time)







'''Part 4: Path of least action'''


'Part 4a: getting the path of least action, ie path in the absence of noise'


xt_least= Euler.Euler_integration(x0=x0,f=flow, wt=np.zeros([dim,Time.size,N]),Time=Time)


'''Part 5: Plots'''


'Part 5a: 3D paths of least action'
N_plot= min(N, 2) #number of samples to plot

fig = plt.figure(1)
plt.clf()
ax = plt.axes(projection='3d')
cmap = plt.cm.get_cmap('cool')
#cmap_rev = plt.cm.get_cmap('plasma')
for n in range(N_plot):
    print('n', n, '/', N_plot)
    for t in range(1, timesteps):
        if t %(10**3) ==0:
            print('t',t)
        ax.plot3D(xt_least[0, t-1:t+1, n], xt_least[1, t-1:t+1, n], xt_least[2, t-1:t+1, n], c=cmap(1-t/timesteps), lw = 0.3)
        #ax.plot3D(x[0, t-1:t+1, 1], x[1, t-1:t+1, 1], x[2, t-1:t+1, 1], c=cmap_rev(1 - t / T), lw=0.3)
#ax.scatter(x0[0], x0[1], x0[2], marker='o', c='red', s =20)
#ax.text(x0[0], x0[1], x0[2],s='$\mathbf{x_0}$')
plt.title('Paths of least action')
plt.title(f'3D')
#ax.legend() #loc="upper right"


'Part 5b: 2D paths of least action (projections)'
lw=0.5
alpha=0.5

plt.figure(2)
plt.clf()
print('====Plot least action paths xy plane===')
for n in range(N_plot):
    print('n', n,'/',N_plot)
    colourline.plot_cool(xt_least[0, :, n], xt_least[1, :, n], lw=lw,alpha=alpha)
plt.suptitle('Paths of least action ', fontsize=16)
plt.title(f'x-y plane', fontsize=14)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
#plt.xlim(right=12,left=-5)
#plt.ylim(top=12,bottom=-5)

plt.figure(3)
plt.clf()
print('====Plot least action paths xz plane===')
for n in range(N_plot):
    print('n', n, '/', N_plot)
    colourline.plot_cool(xt_least[0, :, n], xt_least[2, :, n], lw=lw,alpha=alpha)
plt.suptitle('Paths of least action', fontsize=16)
plt.title(f'x-z plane', fontsize=14)
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
#plt.xlim(right=12,left=-5)
#plt.ylim(top=12,bottom=-5)


plt.figure(4)
plt.clf()
print('====Plot least action paths yz plane===')
for n in range(N_plot):
    print('n', n, '/', N_plot)
    colourline.plot_cool(xt_least[1, :, n], xt_least[2, :, n], lw=lw,alpha=alpha)
plt.suptitle('Paths of least action', fontsize=16)
plt.title(f'y-z plane', fontsize=14)
plt.xlabel(r'$y$')
plt.ylabel(r'$z$')
#plt.xlim(right=12,left=-5)
#plt.ylim(top=12,bottom=-5)
