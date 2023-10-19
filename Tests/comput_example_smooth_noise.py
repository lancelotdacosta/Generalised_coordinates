'''in the script we test whether the square rational kernel fits the conditions so that the associated sample paths are smooth'''
#wolfram prompt solved this: -0.01*x**0.01*log(x)**(-0.01 - 1)/x - x**0.01*(-0.01- 1)*log(x)**(-0.01 - 1)/(x*log(x))
#

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# define symbolic variable
h = sp.symbols('h')
e = sp.symbols('e')
d= sp.symbols('d')

# F= -sp.exp(e*sp.log(h))/sp.exp((1+d)*sp.log(sp.log(h)))
F= h**e/(-sp.log(h))**(1+d)

N=2

derivs = [F]

for n in range(N):
    deriv= sp.diff(derivs[n], h)
    derivs.append(deriv)
    print(derivs[n+1])

derivs_sub=[None] * (N+1)

for n in range(N+1):
    derivs_sub[n]=derivs[n].subs(d,10**-2)
    derivs_sub[n]=derivs_sub[n].subs(e,10**-2)


F_prime= derivs_sub[1]

time=np.arange(10**-7,0,-10**-10)

F_prime_eval=np.zeros(time.size)
F_eval=np.zeros(time.size)

def F_prime_np(h):
    e=0.1
    d= 1
    return e*h**e*(-np.log(h))**(-d - 1)/h + h**e*(-np.log(h))**(-d - 1)*(-d - 1)/(h*np.log(h))
            # -e*np.exp(e*np.log(h))*np.exp(-(d + 1)*np.log(np.log(h)))/h - (-d - 1)*np.exp(e*np.log(h))*np.exp(-(d + 1)*np.log(np.log(h)))/(h*np.log(h)))

def F_np(h):
    e=0.01
    d= 0.01
    return h**e/(-np.log(h))**(1+d)

for t in range(time.size):
    # print(F_prime.subs(h,time[t]))
    # F_prime_eval[t]=F_prime.subs(h,time[t])#.astype(np.float64)
    F_prime_eval[t]=F_prime_np(time[t])#.astype(np.float64)
    F_eval[t] = F_np(time[t])
    print(F_prime_eval[t])


plt.figure(1)
plt.clf()
plt.plot(time, F_eval)
plt.plot(time, F_prime_eval)
plt.xlabel(r'Lag $h$')
plt.yscale('log')


