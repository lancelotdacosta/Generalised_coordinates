'''in this script, we compare two ways of generating the solutions of an OU process (ie linear SDE)
driven by white noise convolved with a Gaussian kernel (ie coloured noise)
For such a given two-dimensional process, we compare the sample paths of:
1) The Euler integration method, where the sample paths of the noise process will have been generated using the method of generalised coordinates
2) The full generalised coordinates method used to integrate the stochastic differential equation
'''

import numpy as np

'''Part 0: preparing the OU process and the serial derivatives at zero of the noise'''

'Part 0a: preparing the drift matrix of the OU process'''

dim =2 #State space dimension

alpha= 1 #Scaling of solenoidal flow

Q = alpha * np.array([0, 1, -1, 0]).reshape([dim, dim]) #solenoidal flow matrix

B= np.eye(dim)+Q #negative drift matrix of OU process for a steady state distribution N(0,I)



'''Part 0b: Preparing samples of the serial derivatives of the noise'''

