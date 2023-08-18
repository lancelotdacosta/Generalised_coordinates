'INCONCLUSIVE script'
'''In this script we test and implement the autodiff features of pytorch
Goal: learn how to automatically differentiate any function any number of times'''

''''''

import torch


'This code works for multivarite functions'
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2  #function from R^4 to R^2
#Q = 3*a**3   #function from R^2 to R^2

Q.backward(gradient=torch.tensor([1., 1.]))

#Q.grad_fn()

a.grad #gradient of Q wrt a evaluated at a
b.grad #gradient of Q wrt b evaluated at b

print(9*a**2 == a.grad)
print(-2*b == b.grad)


#torch.autograd.grad(Q￿)

'This code works for univariate functions'

x = torch.tensor(3.0, requires_grad = True)
y = 3 * x ** 2
print("Result of the equation is: ", y)
y.backward()
print("Derivative of y at x = 3 is: ", x.grad)


'Code for higher order derivatives'
from torch.autograd import grad

def nth_derivative(f, wrt, n):

    for i in range(n):

        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()

    return grads

x = torch.tensor([2., 3.], requires_grad=True)
loss = (x ** 2).sum()

print(nth_derivative(f=loss, wrt=x, n=2))


'Attempting to compute the derivative of the autocovariance of the process'
h = torch.tensor(0.0, requires_grad = True)
beta = torch.tensor(1, requires_grad = False)
# return np.sqrt(np.pi/(2*beta))*k(h, beta/2) #numpy version
K=torch.sqrt(torch.pi / (2 * beta)) * torch.exp(-beta / 2 * h * h)  # torch version
K

K.backward()

print("Derivative of K at h = 0 is: ", h.grad)

