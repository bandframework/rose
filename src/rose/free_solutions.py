import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.interpolate import interp1d
from scipy.misc import derivative

def F(rho, ell):
    return rho*spherical_jn(ell, rho)


def G(rho, ell):
    return -rho*spherical_yn(ell, rho)


def H_plus(rho, ell):
    return G(rho, ell) + 1j*F(rho, ell)


def H_minus(rho, ell):
    return G(rho, ell) - 1j*F(rho, ell)


def H_plus_prime(rho, ell, dx=1e-6):
    return derivative(lambda z: H_plus(z, ell), rho, dx=dx)


def H_minus_prime(rho, ell, dx=1e-6):
    return derivative(lambda z: H_minus(z, ell), rho, dx=dx)


# def phase_shift(u, up, ell, x0):
#     rl = 1/x0 * (u/up)
#     return np.log(
#         (H_minus(x0, ell) - x0*rl*H_minus_prime(x0, ell)) / 
#         (H_plus(x0, ell) - x0*rl*H_plus_prime(x0, ell))
#     ) / 2j


def phase_shift(u, s, ell, x0, dx=1e-6):
    u_func = interp1d(s, u, kind='cubic')
    rl = 1/x0 * (u_func(x0)/derivative(u_func, x0, dx=dx))
    return np.log(
        (H_minus(x0, ell) - x0*rl*H_minus_prime(x0, ell)) / 
        (H_plus(x0, ell) - x0*rl*H_plus_prime(x0, ell))
    ) / 2j