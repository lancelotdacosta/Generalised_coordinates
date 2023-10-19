# import library
import sympy as sp
import numpy as np

# define symbolic variable
h = sp.symbols('h')

# define symbolic equation for kernel
K = 1/(1+h**2)

N=4

derivs = [K]

for n in range(N):
    deriv= sp.diff(derivs[n], (h,2))
    derivs.append(deriv)
    print(derivs[n])
