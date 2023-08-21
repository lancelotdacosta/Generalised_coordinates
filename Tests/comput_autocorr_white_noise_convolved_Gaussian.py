'''in this file we check one of the steps in the computation of the autocorrelation function of white noise convolved with a Gaussian kernel done in the paper'''

import sympy as sp


beta = sp.symbols('beta')
t = sp.symbols('t')
tau = sp.symbols('tau')
s = sp.symbols('s')

exp=-beta*(t-tau)**2-beta*(s-tau)**2

exp2=- beta /2*(t-s)**2-2* beta*(tau-(s+t)/2)**2

print(sp.simplify(exp -exp2))