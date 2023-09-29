'''
Useful utility functions that I don't want to clutter up other modules with.
'''
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.sparse import diags, lil_matrix
from scipy.misc import derivative
from scipy.special import eval_legendre

from .constants import MASS_N, MASS_P, HBARC, AMU


class Projectile(Enum):
    neutron = 0
    proton = 1


def finite_difference_first_derivative(
    s_mesh: npt.ArrayLike,
    sparse: bool = False
):
    '''
    Computes a finite difference matrix that (when applied) represent the first
    derivative.
    '''
    dx = s_mesh[1] - s_mesh[0]
    assert np.all(np.abs(s_mesh[1:] - s_mesh[:-1] - dx) < 1e-14), '''
Spacing must be consistent throughout the entire mesh.
    '''
    n = s_mesh.size
    coefficients = np.array([1, -8, 8, -1]) / (12*dx)
    if sparse:
        D1 = lil_matrix(diags(coefficients, [-2, -1, 1, 2], shape=(n, n)))
    else:
        D1 = lil_matrix(diags(coefficients, [-2, -1, 1, 2], shape=(n, n))).toarray()

    # Use O(dx^2) forward difference approximations for the first 2 rows.
    D1[0, :5] = np.array([-3, 4, -1, 0, 0]) / (2*dx)
    D1[1, :5] = np.array([0, -3, 4, -1, 0]) / (2*dx)

    # Use O(dx^2) backward difference approximations for the last 2 rows.
    D1[-2, -5:] = np.array([0, 1, -4, 3, 0]) / (2*dx)
    D1[-1, -5:] = np.array([0, 0, 1, -4, 3]) / (2*dx)

    return D1



def finite_difference_second_derivative(
    s_mesh: npt.ArrayLike,
    sparse: bool = False
):
    '''
    Computes a finite difference matrix that represents the second derivative
    (w.r.t. s or rho) operator in coordinate space.
    '''
    dx = s_mesh[1] - s_mesh[0]
    assert np.all(np.abs(s_mesh[1:] - s_mesh[:-1] - dx) < 1e-14), '''
Spacing must be consistent throughout the entire mesh.
    '''
    n = s_mesh.size
    coefficients = np.array([-1, 16, -30, 16, -1]) / (12*dx**2)
    if sparse:
        D2 = lil_matrix(diags(coefficients, [-2, -1, 0, 1, 2], shape=(n, n)))
    else:
        D2 = diags(coefficients, [-2, -1, 0, 1, 2], shape=(n, n)).toarray()

    # Use O(dx^2) forward difference approximation for the first 2 rows.
    D2[0, :5] = np.array([2, -5, 4, -1, 0]) / dx**2
    D2[1, :5] = np.array([0, 2, -5, 4, -1]) / dx**2

    # Use O(dx^2) backward difference approximation for the last 2 rows.
    D2[-2, -5:] = np.array([-1, 4, -5, 2, 0]) / dx**2 
    D2[-1, -5:] = np.array([0, -1, 4, -5, 2]) / dx**2 

    return D2


def regular_inverse_r(r, r_c):
    if isinstance(r, float):
        return 1/(2*r_c) * (3 - (r/r_c)**2) if r < r_c else 1/r
    else:
        ii = np.where(r <= r_c)[0]
        jj = np.where(r > r_c)[0]
        return np.hstack([1/(2*r_c) * (3 - (r[ii]/r_c)**2), 1/r[jj]])


def regular_inverse_s(s, s_c):
    if isinstance(s, float) or isinstance(s, int):
        return 1/(2*s_c) * (3 - (s/s_c)**2) if s < s_c else 1/s
    else:
        ii = np.where(s <= s_c)[0]
        jj = np.where(s > s_c)[0]
        return np.hstack([1/(2*s_c) * (3 - (s[ii]/s_c)**2), 1/s[jj]])


def eval_assoc_legendre(n, x):
        if n == 0:
            return np.zeros(x.size)
        else:
            return -(1-x**2)**(1/2) * derivative(lambda z: eval_legendre(n, z), x, dx=1e-9)


def nucleon_nucleus_kinematics(A: int, Z: int,  energy_lab: float, p : Projectile):
    """
    calculates the reduced mass, and the COM frame kinetic energy
    and wavenumber for a nucleon scattering on a target nuclide (A,Z)
    Parameters:
        A : mass number of target
        Z : proton number of target
        energy_lab: bombarding energy in the lab frame [MeV]
        p : projectile type
    Returns:
        mu (float) : reduced mass in MeV/c^2
        energy_com (float) : center-of-mass frame energy in MeV
        k (float) : center-of-mass frame wavenumber in fm^-1

    """
    N = A - Z

    #TODO use a table rather than semi-empirical mass formula
    delta = 0
    if N%2 == 0 and Z%2 == 0:
        delta = 12.0 / np.sqrt(A)
    elif N%2 != 0 and Z%2 != 0:
        delta = - 12.0 / np.sqrt(A)

    Eb = (
        15.8 * A
      - 18.3 * A**(2/3)
      - 0.714 * Z*(Z-1)/(A**(1/3))
      - 23.2 * (N-Z)**2/A
      + delta
    )

    target_mass = Z * MASS_P + N * MASS_N - Eb # MeV/c^2

    if p == Projectile.neutron:
        mu = target_mass * MASS_N / (target_mass + MASS_N)
        energy_com = target_mass / (MASS_N + target_mass) * energy_lab
    elif p == Projectile.neutron:
        mu = target_mass * MASS_P / (target_mass + MASS_P)
        energy_com = target_mass / (MASS_P + target_mass) * energy_lab

    k = np.sqrt(2 * mu * energy_com) / HBARC

    return mu, energy_com, k
