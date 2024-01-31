'Here we have routines concerning the use of generative models'

import numpy as np
import sympy as sp
from Routines import Taylor_series,sampling_generalised_noise,colourline,gen_coords, convolving_white_noise



'specification of generative model in terms of energies'


def prior_energy(F, x, t, order_x, Kw, hw, epsilon_w=1, lin=True, trun=True):
    # The Lagrangian equals the prior energy up to a constant
    prior_en = gen_coords.Lagrangian(F, x, t, Kw, hw, order_x, lin, trun) / epsilon_w ** 2
    # The latter will only have terms that include derivatives up to order order_x

    return prior_en


def likelihood_energy(g, y, t, order_y,Kz,hz, epsilon_z=1, lin=True):

    dim_y=y.shape[0]

    # Symbolic vector of generalized coordinates for the data
    geny = gen_coords.serial_derivs_xt(y, t, order_y + 1)  # shape [dim_y,order+1]

    # Symbolic observation function in generalized coordinates with the same number of orders as the data
    if lin:
        geng = gen_coords.generalised_flow_lin(g, y,t, order_y + 1)  # shape [dim,order+1]
    else:
        geng = gen_coords.generalised_flow(g, t, order_y + 1)  # shape [dim,order+1]

    # Generalized covariance of the observation noise
    gen_cov_z = epsilon_z ** 2 * sampling_generalised_noise.gen_covariance_nd(K=Kz, wrt=hz, at=0, order=order_y + 1,
                                                                              dim=dim_y)
    # Generalized precision of the observation noise
    gen_pres_z = np.linalg.inv(gen_cov_z)

    # Now we have to put it all together in the Quadratic form
    vec = geny - geng
    rvec = gen_coords.flatten_gen_coord(vec)

    likelihood_en = rvec.T @ gen_pres_z @ rvec / 2  # shape = 1
    # The latter will only have terms that include derivatives up to order order_y

    return likelihood_en


def genmod_energy(F, x, t, order_x,Kw, hw, g, y, order_y,Kz,hz,  epsilon_w=1, epsilon_z=1, lin=True, trun=True):
    # This has to be lesser or equal than order_x
    if order_y > order_x:
        raise TypeError('Incompatible orders of motion')

    prior_en = prior_energy(F, x, t, order_x, Kw, hw,epsilon_w, lin, trun)

    likelihood_en = likelihood_energy(g, y, t, order_y, Kz,hz, epsilon_z, lin)

    generative_energy = prior_en + likelihood_en

    return generative_energy


'Free energy under the Laplace approximation for generative model'

# genmod_energy(F, x, t, order_x,Kw, hw, g, y, order_y,Kz,hz,  epsilon_w=1, epsilon_z=1, lin=True, trun=True)