"""
Useful functions related to the solution to the free, radial Schrödinger equation.
"""

import numpy as np
from mpmath import coulombf, coulombg
from scipy.interpolate import UnivariateSpline


def F(rho, ell, eta):
    """
    Bessel function of the first kind.
    """
    return np.complex128(coulombf(ell, eta, rho))


def G(rho, ell, eta):
    """
    Bessel function of the second kind.
    """
    return np.complex128(coulombg(ell, eta, rho))


def H_plus(rho, ell, eta):
    """
    Hankel function of the first kind.
    """
    return G(rho, ell, eta) + 1j * F(rho, ell, eta)


def H_minus(rho, ell, eta):
    """
    Hankel function of the second kind.
    """
    return G(rho, ell, eta) - 1j * F(rho, ell, eta)


def coulomb_func_deriv(func, s, l, eta):
    """
    Derivative of Coulomb functions F, G, and Coulomb Hankel functions H+ and H-
    """
    # recurrance relations from https://dlmf.nist.gov/33.4
    # dlmf Eq. 33.4.4
    R = np.sqrt(1 + eta**2 / (l + 1) ** 2)
    S = (l + 1) / s + eta / (l + 1)
    Xl = func(s, l, eta)
    Xlp = func(s, l + 1, eta)
    return S * Xl - R * Xlp


def H_plus_prime(s, l, eta):
    """
    Derivative of the Hankel function (first kind) with respect to s
    """
    return coulomb_func_deriv(H_plus, s, l, eta)


def H_minus_prime(s, l, eta, dx=1e-6):
    """
    Derivative of the Hankel function (second kind) with respect to s.
    """
    return coulomb_func_deriv(H_minus, s, l, eta)


def phi_free(rho, ell, eta):
    """
    Solution to the "free" (V = 0) radial Schrödinger equation.
    """
    return -0.5j * (H_plus(rho, ell, eta) - H_minus(rho, ell, eta))


def phase_shift(phi, phi_prime, ell, eta, x0):
    rl = 1 / x0 * (phi / phi_prime)
    return (
        np.log(
            (H_minus(x0, ell, eta) - x0 * rl * H_minus_prime(x0, ell, eta))
            / (H_plus(x0, ell, eta) - x0 * rl * H_plus_prime(x0, ell, eta))
        )
        / 2j
    )


def phase_shift_interp(u, s, ell, eta, x0, dx=1e-6):
    """
    Given the solution, u, on the s grid, return the phase shift (with respect to the free solution).
    """
    spl = UnivariateSpline(s, u, k=3)
    rl = 1 / x0 * (spl(x0) / spl.derivative()(x0))
    return (
        np.log(
            (H_minus(x0, ell, eta) - x0 * rl * H_minus_prime(x0, ell, eta))
            / (H_plus(x0, ell, eta) - x0 * rl * H_plus_prime(x0, ell, eta))
        )
        / 2j
    )
