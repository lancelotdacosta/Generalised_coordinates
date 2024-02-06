import sympy as sp
import numpy as np


order = 3
dim =3 #State space dimension
N = 1 #number of sample paths

np.random.seed(1)
#x0 = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=N).T #initial condition
x0 = np.array([1,2,3]).reshape([dim,N])
#tilde_w0= np.random.multivariate_normal(mean=np.zeros(dim*order), cov=np.eye(dim*order), size=N).reshape([dim, order, N])
tilde_w0 = np.zeros([dim,order,N])


'''Function
Input: x0,  tilde w0, order'''

sigma = 10

rho=28

beta=8/3

t = sp.Symbol("t")


#x0 = sp.Function("x0")(t)
#x1 = sp.Function("x1")(t)
#x2 = sp.Function("x2")(t)

#x= sp.Matrix([x0,x1,x2])
#print(x[0])

x= sp.Matrix([sp.Function("x0")(t),
              sp.Function("x1")(t),
              sp.Function("x2")(t)])


F0= sigma*(x[1]-x[0])
F1 = x[0] * (rho-x[2]) - x[1]
F2 = x[0]* x[1] - beta * x[2]

F=sp.Matrix([F0,F1,F2])

'Numpy form of flow'

def flow(x):
    y=np.empty(x.shape)
    y[0,:]= sigma*(x[1,:]-x[0,:])
    y[1, :] = x[0,:]*(rho-x[2,:])-x[1,:]
    y[2, :] = x[0,:]*x[1,:]-beta*x[2,:]
    return y

print('flow(x0)=',flow(x0))

'Test'
#dF=sp.diff(F, t)
#print(dF)
#dF.subs(sp.Derivative(x[0], t),2) #needs to be assigned!
#print(dF)
#dF_subs=dF.subs(x0,2)
#y = sp.Symbol("y")

#dF_subbed=dF.subs(sp.Derivative(x0, t),2)
#print(dF_subbed)

'Test 2'

# G= x[0]**2
# print(G)
# dG=sp.diff(G, t)
# print(dG)
# dG=dG.subs(sp.Derivative(x[0], t),2)
# print(dG)
# dG=dG.subs(x[0],3)
# print(dG)



'Simple method'

dxdtn= sp.zeros(dim,order) #serial time derivatives of x for each order and dimension (starting with order 0)

for d in range(dim):
    for o in range(order):
        dxdtn[d,o]=sp.diff(x[d], (t,o)) #o-th time derivative of the d-th component of the function x(t)


tilde_x0= np.empty([dim, order+1, N]) #initialise generalised coordinates at zero of solution to SDE
tilde_x0[:, 0, :]=x0 #say that order zero is the initial condition


#for n in range(N):
n=0

#dFdtn_eval=dFdtn #for this sample time derivatives that will subsequently be evaluated

dFdtn = sp.zeros(dim,order) #serial time derivatives of F for each order and dimension (starting with order 0)--sample dependent
for d in range(dim):
    for s in range(order):
        dFdtn[d,s]=sp.diff(F[d], (t,s)) #o-th time derivative of the d-th component of the function F(x(t))

for o in range(1,order +1):
    #evaluate (o-1)-th time derivative of F at the time derivatives of x
    #for d in range(dim):
    #o=1
    print('o',o)
    for r in reversed(range(o)): #starts by o-1 and goes backward to 0
        print('---r',r)
        for d in range(dim):
            print('------d', d)
            dFdtn[:,o-1] = dFdtn[:,o-1].subs(dxdtn[d,r],tilde_x0[d,r,n])

    #evaluate o-th time derivative of x
    tilde_x0[:,o,n]= np.array(dFdtn[:,o-1]).astype(np.float64).reshape([dim]) #dFdtn[d,o-1] @ tilde_x0[:,:o,n]

'''Check of method'''
#the following should be equal if all the noise increments were zero
print(tilde_x0[:,0,n], x0)
print(tilde_x0[:,1,n], flow(x0))

second_order= np.array([sigma*(tilde_x0[1,1,n]-tilde_x0[0,1,n]),
                tilde_x0[0,1,n]*(rho-tilde_x0[2,0,n])-tilde_x0[1,1,n]-tilde_x0[0,0,n]*tilde_x0[2,1,n],
                tilde_x0[0,1,n]*tilde_x0[1,0,n]+tilde_x0[0,0,n]*tilde_x0[1,1,n]-beta*tilde_x0[2,1,n]])
print(tilde_x0[:,2,n], second_order)

third_order= np.array([sigma*(tilde_x0[1,2,n]-tilde_x0[0,2,n]),
                tilde_x0[0,2,n]*(rho-tilde_x0[2,0,n]) -tilde_x0[1,2,n]-tilde_x0[0,0,n]*tilde_x0[2,2,n]-2*tilde_x0[0,1,n]*tilde_x0[2,1,n],
                tilde_x0[0,2,n]*tilde_x0[1,0,n]+tilde_x0[0,0,n]*tilde_x0[1,2,n]+2*tilde_x0[0,1,n]*tilde_x0[1,1,n]-beta*tilde_x0[2,2,n]])
print(tilde_x0[:,3,n], third_order)






'Try again'

# tilde_x0= np.empty([dim, order+1, N])
# tilde_x0[:, 0, :]=x0
#
# dF=F #serial derivatives of F
# for n in range(N):
#     for o in range(order):
#         print('outer loop', 'n=', n, 'o=',o)
#         'start of inner function'
#
#         for d in range(dim):
#             for r in reversed(range(o+1)):
#                 dx = x[d]
#                 print('--- inner loop', 'd=', d, 'r=', r, dx)
#                 # print('---', 'd', d, 'o', o, dx)
#                 for i in range(r):
#                     dx = sp.diff(dx, t)  # serial derivatives of x wrt t
#                 print('--------- inner loop', 'd=', d, 'r=', r, dx)
#                 dF = dF.subs(dx, tilde_x0[d, r, n])
#         print(dF)
#         step= np.array(dF).astype(np.float64).reshape([dim])  # this should be a matrix of real numbers
#
#         'resume outer function'
#         tilde_x0[:,o+1,n]= step + tilde_w0[:,o,n]
#         dF=sp.diff(dF, t) #serial derivatives of F
#
# print('tilde_x0',tilde_x0.reshape([dim,order+1]))








'Back to normal'
#input: sympy F, sympy x(t), sympy t, numpy x0, numpy tilde_w0
# def eval_derivs(dF,x,t,tilde_x0):
#     [dim,order]=tilde_x0.shape
#     for d in range(dim):
#         for o in reversed(range(order)):
#             dx = x[d]
#             print('--- inner loop A', 'd', d, 'o', o, dx)
#             #print('---', 'd', d, 'o', o, dx)
#             for i in range(o):
#                 dx = sp.diff(dx, t)  # serial derivatives of x wrt t
#             print('--- inner loop B','d',d,'o',o,dx)
#             dF=dF.subs(dx,tilde_x0[d,o])
#     print(dF)
#     return np.array(dF).astype(np.float64).reshape([dim]) #this should be a matrix of real numbers
#
# def sol_gen_coords(F,x,t,x0,tilde_w0):
#     tilde_x0= np.empty([dim, order+1, N])
#     tilde_x0[:, 0, :]=x0
#
#     dF=F #serial derivatives of F
#     for n in range(N):
#         for o in range(order):
#             print('outer loop', 'n', n, 'o',o)
#             step= eval_derivs(dF,x,t,tilde_x0[:,:o+1,n])
#             tilde_x0[:,o+1,n]= step + tilde_w0[:,o,n]
#             dF=sp.diff(dF, t) #serial derivatives of F
#     return tilde_x0

'''Test of method'''

#tilde_x0=sol_gen_coords(F,x,t,x0,tilde_w0)




