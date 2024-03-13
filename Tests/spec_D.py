'''the goal of this routine is to compute the spectrum of the matrix D, for any finite dimension'''
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy



n=3 #orders of motion +1

D= np.zeros([n,n])

for i in range(n-1):
    D[i,i+1]=1

d=2

D=np.kron(D,np.eye(d))


spec= np.linalg.eig(D)
eigvals = np.linalg.eigvals(D)
eigvec = spec[1]

# print eigenvalues
print('eigvals',eigvals)
# print corresponding eigenvectors
for i in range(n*d):
    print(np.round(eigvec[:,i],2))

cond=np.linalg.cond(D)
# print conditioning number
print(cond)

'''finding eigenvalues values numerically by finding roots of the characteristic polynomial'''

lambdas=np.arange(-0.1,0.1,0.0001)

def char_pol(D,l):
    n= D.shape[0]
    I=np.eye(n)
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
I_sym = sp.Matrix(np.eye(n))
l = sp.Symbol("l")
char_pol_sym= sp.det(D_sym-l*I_sym)



'======='
D2=np.kron(D,np.eye(2))
E=scipy.linalg.expm(0.5*D2)
G=scipy.linalg.expm(0.5*D)
np.kron(G,np.eye(2))-E