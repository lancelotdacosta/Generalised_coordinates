'''the goal of this routine is to compute the spectrum of the matrix D, for any finite dimension'''
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp



d=10

D= np.zeros([d,d])

for n in range(d-1):
    D[n,n+1]=1


spec= np.linalg.eig(D)
eigvals = np.linalg.eigvals(D)
eigvec = spec[1]

cond=np.linalg.cond(D)

'''finding eigenvalues values numerically by finding roots of the characteristic polynomial'''

lambdas=np.arange(-0.1,0.1,0.0001)

def char_pol(D,l):
    d= D.shape[0]
    I=np.eye(d)
    return np.linalg.det(D-l*I)

char_pol_eval= np.zeros(lambdas.shape)

for i in range(lambdas.size):
    char_pol_eval[i]=char_pol(D,lambdas[i])

plt.figure(1)
plt.clf()
plt.plot(lambdas, char_pol_eval)
#plt.yscale('log')

'''Computing the characteristic polynomial Symbolically'''


D_sym =sp.Matrix(D)
I_sym = sp.Matrix(np.eye(d))
l = sp.Symbol("l")
char_pol_sym= sp.det(D_sym-l*I_sym)



