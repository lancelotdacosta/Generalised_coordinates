# import library
import sympy as sp
import numpy as np

# define symbolic variable
h = sp.symbols('h')
beta = 1 #sp.symbols('b')

# define symbolic equation for kernel
K = sp.sqrt(sp.pi / (2 * beta)) * sp.exp(-beta / 2 * h * h)

# find first order derivative
first_order_derivative = K.diff(h)


'''Script for finding the serial derivatives from order zero to N of K evaluated at the point 0'''


def formal_serial_derivatives(F, wrt, order):
    form_derivs = []
    for n in range(order+1):
        form_derivs.append(F)
        F = F.diff(wrt)
    return form_derivs

def serial_derivatives(F,wrt,at,order):
    form_derivs=formal_serial_derivatives(F, wrt, order)
    num_derivs = np.zeros(order + 1)
    for n in range(order+1):
        num_derivs[n] = form_derivs[n].subs(wrt, at)
    return num_derivs

F=K
wrt=h
at = 0
order=20

form_derivs = formal_serial_derivatives(K,h,order)
num_derivs = serial_derivatives(K,h,at,order)

print(num_derivs)
