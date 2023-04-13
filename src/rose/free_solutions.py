'''
Useful functions related to the solution to the free, radial Schrödinger equation.
'''

import numpy as np
from scipy.special import spherical_jn, spherical_yn
from mpmath import coulombf, coulombg
from scipy.interpolate import interp1d
from scipy.misc import derivative

def F(rho, ell, eta):
    '''
    Bessel function of the first kind.
    '''
    # return rho*spherical_jn(ell, rho)
    return np.complex128(coulombf(ell, eta, rho))


def G(rho, ell, eta):
    '''
    Bessel function of the second kind.
    '''
    # return -rho*spherical_yn(ell, rho)
    return np.complex128(coulombg(ell, eta, rho))


def H_plus(rho, ell, eta):
    '''
    Hankel function of the first kind.
    '''
    return G(rho, ell, eta) + 1j*F(rho, ell, eta)


def H_minus(rho, ell, eta):
    '''
    Hankel function of the second kind.
    '''
    return G(rho, ell, eta) - 1j*F(rho, ell, eta)


def H_plus_prime(rho, ell, eta, dx=1e-6):
    '''
    Derivative of the Hankel function (first kind) with respect to rho.
    '''
    return derivative(lambda z: H_plus(z, ell, eta), rho, dx=dx)


def H_minus_prime(rho, ell, eta, dx=1e-6):
    '''
    Derivative of the Hankel function (second kind) with respect to rho.
    '''
    return derivative(lambda z: H_minus(z, ell, eta), rho, dx=dx)


def phi_free(rho, ell, eta):
    '''
    Solution to the "free" (V = 0) radial Schrödinger equation.
    '''
    return -0.5j * (H_plus(rho, ell, eta) - H_minus(rho, ell, eta))


# def phase_shift(u, up, ell, x0):
#     rl = 1/x0 * (u/up)
#     return np.log(
#         (H_minus(x0, ell) - x0*rl*H_minus_prime(x0, ell)) / 
#         (H_plus(x0, ell) - x0*rl*H_plus_prime(x0, ell))
#     ) / 2j


def phase_shift(phi, phi_prime, ell, eta, x0):
    rl = 1/x0 * (phi/phi_prime)
    return np.log(
        (H_minus(x0, ell, eta) - x0*rl*H_minus_prime(x0, ell, eta)) / 
        (H_plus(x0, ell, eta) - x0*rl*H_plus_prime(x0, ell, eta))
    ) / 2j


def phase_shift_interp(u, s, ell, eta, x0, dx=1e-6):
    '''
    Given the solution, u, on the s grid, return the phase shift (with respect to the free solution).
    '''
    u_func = interp1d(s, u, kind='cubic')
    rl = 1/x0 * (u_func(x0)/derivative(u_func, x0, dx=dx))
    return np.log(
        (H_minus(x0, ell, eta) - x0*rl*H_minus_prime(x0, ell, eta)) / 
        (H_plus(x0, ell, eta) - x0*rl*H_plus_prime(x0, ell, eta))
    ) / 2j