"""
Useful utility functions that I don't want to clutter up other modules with.
"""

from enum import Enum
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from numba import njit

from scipy.sparse import diags, lil_matrix
from scipy.misc import derivative
from scipy.special import eval_legendre
from scipy.stats import qmc

from .constants import MASS_N, MASS_P, HBARC, ALPHA

MAX_ARG = np.log(1 / 1e-16)

# AME mass table DB initialized at import
__AME_DB__ = None
__AME_PATH__ = (
    Path(__file__).parent.resolve() / Path("../data/mass_1.mas20.txt")
).resolve()


class Projectile(Enum):
    neutron = 0
    proton = 1


@dataclass
class NucleonNucleusXS:
    r"""
    Holds differential cross section, analyzing power, total cross section and reaction cross secton,
    all at a given energy
    """

    dsdo: np.array
    Ay: np.array
    t: float
    rxn: float


@njit
def xs_calc_neutral(
    k: float,
    angles: np.array,
    Splus: np.array,
    Sminus: np.array,
    P_l_theta: np.array,
    P_1_l_theta: np.array,
):
    xst = 0.0
    xsrxn = 0.0
    a = np.zeros_like(angles, dtype=np.complex128)
    b = np.zeros_like(angles, dtype=np.complex128)

    for l in range(Splus.shape[0]):
        # scattering amplitudes
        a += (
            (2 * l + 1 - (l + 1) * Splus[l] - l * Sminus[l])
            * P_l_theta[l, :]
            / (2j * k)
        )
        b += (Sminus[l] - Splus[l]) * P_1_l_theta[l, :] / (2j * k)

        # cross sections
        xsrxn += (l + 1) * (1 - np.absolute(Splus[l])**2) + l * (
            1 - np.absolute(Sminus[l])**2
        )
        xst += (l + 1) * (1 - np.real(Splus[l])) + l * (1 - np.real(Sminus[l]))

    dsdo = np.real(a * np.conj(a) + b * np.conj(b)) * 10
    Ay = 2 * np.real( a * np.conj(b)) * 10 / dsdo
    xst *= 10 * 2 * np.pi / k**2
    xsrxn *= 10 * np.pi / k**2

    return dsdo, Ay, xst, xsrxn


@njit
def xs_calc_coulomb(
    k: float,
    angles: np.array,
    Splus: np.array,
    Sminus: np.array,
    P_l_theta: np.array,
    P_1_l_theta: np.array,
    f_c: np.array,
    sigma_l: np.array,
    rutherford: np.array,
):
    a = np.zeros_like(angles, dtype=np.complex128) + f_c
    b = np.zeros_like(angles, dtype=np.complex128)
    xsrxn = 0.0

    for l in range(Splus.shape[0]):
        # scattering amplitudes
        a += 1j * (
             (2*l + 1 - (l + 1) * Splus[l] - l * Sminus[l])
             * P_l_theta[l, :]
             * np.exp(2j * sigma_l[l])
             / (2 * k)
         )
        b += 1j * (
             (Sminus[l] - Splus[l])
             * P_1_l_theta[l, :]
             * np.exp(2j * sigma_l[l])
             / (2 * k)
         )

        xsrxn += (l + 1) * (1 - np.absolute(Splus[l])**2) + l * (
            1 - np.absolute(Sminus[l])**2
        )

    dsdo = np.real(a * np.conj(a) + b * np.conj(b)) * 10
    Ay = np.real(a * np.conj(b) + b * np.conj(a)) * 10 / dsdo
    xsrxn *= 10 * np.pi / k**2

    dsdo = dsdo / rutherford

    return dsdo, Ay, None, xsrxn


def max_vol(basis, indxGuess):
    r"""basis looks like a long matrix, the columns are the "pillars" V_i(x):
    [   V_1(x)
        V_2(x)
        .
        .
        .
    ]
    indxGuess is a first guess of where we should "measure", or ask the questions

    """
    nbases = basis.shape[1]
    interpBasis = np.copy(basis)

    for ij in range(len(indxGuess)):
        interpBasis[[ij, indxGuess[ij]], :] = interpBasis[[indxGuess[ij], ij], :]
    indexing = np.array(range(len(interpBasis)))

    for ij in range(len(indxGuess)):
        indexing[[ij, indxGuess[ij]]] = indexing[[indxGuess[ij], ij]]

    for iIn in range(1, 100):
        B = np.dot(interpBasis, np.linalg.inv(interpBasis[:nbases]))
        b = np.max(B)
        if b > 1:
            p1, p2 = np.where(B == b)[0][0], np.where(B == b)[1][0]
            interpBasis[[p1, p2], :] = interpBasis[[p2, p1], :]
            indexing[[p1, p2]] = indexing[[p2, p1]]
        else:
            break
        # this thing returns the indexes of where we should measure
    return np.sort(indexing[:nbases])


def latin_hypercube_sample(n_sample: int, bounds: np.array, seed=None):
    r"""
    Generates N Latin hypercube samples in the k-Dimensional box
    defined by bounds, with the first column being lower and 2nd being upper
    bounds, and the column lengths determine the dimension k. If, for any dimension,
    the upper and lower bounds are equal, keeps them frozen.
    """
    # Generate training points using the user-provided bounds,
    # first sanitizing bounds to freeze parameters that are equal
    mask = bounds[:, 0] == bounds[:, 1]
    frozen_params = bounds[mask][:, 0]
    n_unfrozen = bounds[np.logical_not(mask)][:, 0].size

    # bounds for unfrozen params only
    bounds_unfrozen = bounds[np.logical_not(mask)]

    # set up training array (just copy lower bounds for now, we will keep
    # only the frozen parameter values)
    train = np.tile(bounds[:, 0], (n_sample, 1))

    # perform latin hypercube sampling for only the un-frozen params
    sampler = qmc.LatinHypercube(d=n_unfrozen, seed=seed)
    samples = sampler.random(n_sample)
    samples = qmc.scale(samples, bounds_unfrozen[:, 0], bounds_unfrozen[:, 1])

    # fil un frozen indices of training away with samples
    train[:, np.logical_not(mask)] = samples

    return train


def finite_difference_first_derivative(s_mesh: npt.ArrayLike, sparse: bool = False):
    """
    Computes a finite difference matrix that (when applied) represent the first
    derivative.
    """
    dx = s_mesh[1] - s_mesh[0]
    assert np.all(
        np.abs(s_mesh[1:] - s_mesh[:-1] - dx) < 1e-14
    ), """
Spacing must be consistent throughout the entire mesh.
    """
    n = s_mesh.size
    coefficients = np.array([1, -8, 8, -1]) / (12 * dx)
    if sparse:
        D1 = lil_matrix(diags(coefficients, [-2, -1, 1, 2], shape=(n, n)))
    else:
        D1 = lil_matrix(diags(coefficients, [-2, -1, 1, 2], shape=(n, n))).toarray()

    # Use O(dx^2) forward difference approximations for the first 2 rows.
    D1[0, :5] = np.array([-3, 4, -1, 0, 0]) / (2 * dx)
    D1[1, :5] = np.array([0, -3, 4, -1, 0]) / (2 * dx)

    # Use O(dx^2) backward difference approximations for the last 2 rows.
    D1[-2, -5:] = np.array([0, 1, -4, 3, 0]) / (2 * dx)
    D1[-1, -5:] = np.array([0, 0, 1, -4, 3]) / (2 * dx)

    return D1


def finite_difference_second_derivative(s_mesh: npt.ArrayLike, sparse: bool = False):
    """
    Computes a finite difference matrix that represents the second derivative
    (w.r.t. s or rho) operator in coordinate space.
    """
    dx = s_mesh[1] - s_mesh[0]
    assert np.all(
        np.abs(s_mesh[1:] - s_mesh[:-1] - dx) < 1e-14
    ), """
Spacing must be consistent throughout the entire mesh.
    """
    n = s_mesh.size
    coefficients = np.array([-1, 16, -30, 16, -1]) / (12 * dx**2)
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


@njit
def regular_inverse_r(r, r_c):
    if isinstance(r, float):
        return 1 / (2 * r_c) * (3 - (r / r_c) ** 2) if r < r_c else 1 / r
    else:
        ii = np.where(r <= r_c)[0]
        jj = np.where(r > r_c)[0]
        return np.hstack((1 / (2 * r_c) * (3 - (r[ii] / r_c) ** 2), 1 / r[jj]))


@njit
def regular_inverse_s(s, s_c):
    if isinstance(s, float) or isinstance(s, int):
        return 1 / (2 * s_c) * (3 - (s / s_c) ** 2) if s < s_c else 1 / s
    else:
        ii = np.where(s <= s_c)[0]
        jj = np.where(s > s_c)[0]
        within_cutoff = np.zeros(ii.shape)
        if ii.size > 0:
            within_cutoff = 1.0 / (2.0 * s_c) * (3.0 - (s[ii] / s_c) ** 2)

        return np.hstack((within_cutoff, 1.0 / s[jj]))


@njit
def Gamow_factor(l, eta):
    r"""This returns the... Gamow factor.
    See [Wikipedia](https://en.wikipedia.org/wiki/Gamow_factor).

    Parameters:
        l (int): angular momentum
        eta (float): Sommerfeld parameter (see
            [Wikipedia](https://en.wikipedia.org/wiki/Sommerfeld_parameter))

    Returns:
        C_l (float): Gamow factor

    """
    if eta == 0.0:
        if l == 0:
            return 1
        else:
            return 1 / (2 * l + 1) * Gamow_factor(l - 1, 0)
    elif l == 0:
        return np.sqrt(2 * np.pi * eta / (np.exp(2 * np.pi * eta) - 1))
    else:
        return np.sqrt(l**2 + eta**2) / (l * (2 * l + 1)) * Gamow_factor(l - 1, eta)


def eval_assoc_legendre(n, x):
    if n == 0:
        return np.zeros(x.size)
    else:
        return -((1 - x**2) ** (1 / 2)) * derivative(
            lambda z: eval_legendre(n, z), x, dx=1e-9
        )


def init_AME_db():
    r"""
    Should be called once during import to load the AME mass table into memory
    """
    global __AME_PATH__
    global __AME_DB__
    if __AME_PATH__ is None:
        __AME_PATH__ = (
            Path(__file__).parent.resolve() / Path("../../data/mass_1.mas20.txt")
        ).resolve()
        assert __AME_PATH__.is_file()
    if __AME_DB__ is None:
        __AME_DB__ = pd.read_csv(__AME_PATH__, sep="\s+")


def get_AME_binding_energy(A, Z):
    r"""Calculates binding in MeV/c^2 given mass number, A, proton number, Z, by AME2020 lookup"""
    # look up nuclide in AME2020 table
    global __AME_DB__
    df = __AME_DB__
    mask = (df["A"] == A) & (df["Z"] == Z)
    if mask.any():
        # use AME if data exists
        # format is Eb/A [keV/nucleon]
        return float(df[mask]["BINDING_ENERGY/A"].iloc[0]) * A / 1e3
    return None


@njit
def semiempirical_binding_energy(A, Z):
    r"""Calculates binding in MeV/c^2 given mass number, A, proton number, Z, by semi-empriical mass fomrula"""
    N = A - Z
    delta = 0
    if N % 2 == 0 and Z % 2 == 0:
        delta = 12.0 / np.sqrt(A)
    elif N % 2 != 0 and Z % 2 != 0:
        delta = -12.0 / np.sqrt(A)

    Eb = (
        15.8 * A
        - 18.3 * A ** (2 / 3)
        - 0.714 * Z * (Z - 1) / (A ** (1 / 3))
        - 23.2 * (N - Z) ** 2 / A
        + delta
    )
    return Eb


def get_binding_energy(A, Z):
    r"""Calculates binding in MeV/c^2 given mass number, A, proton number, Z, by AME2020 lookup if possible,
    or semi-empriical mass fomrula if not
    """
    Eb = get_AME_binding_energy(A, Z)
    if Eb is None:
        Eb = semiempirical_binding_energy(A, Z)
    return Eb


@njit
def mass(A, Z, Eb):
    r"""Calculates rest mass in MeV/c^2 given mass number, A, proton number, Z, and binding energy in MeV/c^2"""
    N = A - Z
    return Z * MASS_P + N * MASS_N - Eb


def kinematics(
    target: tuple,
    projectile: tuple,
    E_lab: float = None,
    E_com: float = None,
    binding_model: Callable[[int, int], float] = get_binding_energy,
):
    r"""Calculates the reduced mass, COM frame kinetic energy and wavenumber for a projectile (A,Z)
    scattering on a target nuclide (A,Z), with binding energies from binding_model, which defaults
    to lookup in AME2020 mass table. Uses relatavistic approximation of Ingemarsson, 1974:
    https://doi.org/10.1088/0031-8949/9/3/004
    Parameters:
        t : target (A,Z)
        p : projectile (A,Z)
        E_lab: bombarding energy in the lab frame [MeV]. Either E_lab or E_com must be provided, not both.
        E_com: bombarding energy in the com frame [MeV]. Either E_lab or E_com must be provided, not both.
        binding_model : optional callable taking in (A,Z) and returning binding energy in [MeV/c^2],
                        defaults to lookup in AME2020, and semi-empirical mass formula if not available
                        there
    Returns:
        mu (float) : reduced mass in MeV/c^2
        E_com (float) : center-of-mass frame energy in MeV
        k (float) : center-of-mass frame wavenumber in fm^-1
    """
    Eb_target = binding_model(*target)
    Eb_projectile = binding_model(*projectile)
    m_t = mass(*target, Eb_target)
    m_p = mass(*projectile, Eb_projectile)

    if E_lab is None:
        return_Elab = True
        assert E_com is not None
        E_com = np.fabs(E_com)
        E_lab = (m_t + m_p) / m_t * E_com
    else:
        return_Elab = False
        assert E_com is None
        E_lab = np.fabs(E_lab)
        E_com = m_t / (m_t + m_p) * E_lab

    Ep = E_com + m_p

    # relativisitic correction from A. Ingemarsson 1974, Eqs. 17 & 20
    k = (
        m_t
        * np.sqrt(E_lab * (E_lab + 2 * m_p))
        / np.sqrt((m_t + m_p) ** 2 + 2 * m_t * E_lab)
        / HBARC
    )
    mu = k**2 * Ep / (Ep**2 - m_p * m_p) * HBARC**2
    k_C = ALPHA * projectile[1] * target[1] * mu / HBARC
    eta = k_C / k

    if return_Elab:
        return mu, E_lab, k, eta
    else:
        return mu, E_com, k, eta


@njit
def woods_saxon(r, R, a):
    """Woods-Saxon potential"""
    return 1 / (1 + np.exp((r - R) / a))


@njit
def woods_saxon_safe(r, R, a):
    """Woods-Saxon potential

    * avoids `exp` overflows

    """
    x = (r - R) / a
    if isinstance(x, float):
        return 1 / (1 + np.exp(x)) if x < MAX_ARG else 0
    else:
        ii = np.where(x <= MAX_ARG)[0]
        jj = np.where(x > MAX_ARG)[0]
        return np.hstack((1 / (1 + np.exp(x[ii])), np.zeros(jj.size)))


@njit
def woods_saxon_prime(r, R, a):
    """derivative of the Woods-Saxon potential w.r.t. $r$"""
    return -1 / a * np.exp((r - R) / a) / (1 + np.exp((r - R) / a)) ** 2


@njit
def woods_saxon_prime_safe(r, R, a):
    """derivative of the Woods-Saxon potential w.r.t. $r$

    * avoids `exp` overflows

    """
    x = (r - R) / a
    if isinstance(x, float):
        return -1 / a * np.exp(x) / (1 + np.exp(x)) ** 2 if x < MAX_ARG else 0
    else:
        ii = np.where(x <= MAX_ARG)[0]
        jj = np.where(x > MAX_ARG)[0]
        return np.hstack(
            (-1 / a * np.exp(x[ii]) / (1 + np.exp(x[ii])) ** 2, np.zeros(jj.size))
        )


@njit
def thomas_safe(r, R, a):
    """1/r * derivative of the Woods-Saxon potential w.r.t. $r$

    * avoids `exp` overflows, while correctly handeling 1/r term

    """
    x = (r - R) / a
    y = 1.0 / r
    if isinstance(x, float):
        return y * -1 / a * np.exp(x) / (1 + np.exp(x)) ** 2 if x < MAX_ARG else 0
    else:
        ii = np.where(x <= MAX_ARG)[0]
        jj = np.where(x > MAX_ARG)[0]
        return np.hstack(
            (
                y[ii] * -1 / a * np.exp(x[ii]) / (1 + np.exp(x[ii])) ** 2,
                np.zeros(jj.size),
            )
        )


@njit
def coulomb_charged_sphere(
    r: np.double,
    R_C: np.double,
    ZZ: np.double,
):
    r"""Returns the coulomb potential

    Parameters:
        r (double) : scaled radial coordinate s = k * r
        alpha (ndarray)
        ZZ (double),
        R_C (double)
        v_r (callable)
        v_so (callable)
        l_dot_s (int)
    """
    fine_structure = ALPHA * HBARC
    return ZZ * fine_structure * regular_inverse_r(r, R_C)


@njit
def potential(
    r: np.double,
    alpha: np.array,
    v_r,
    v_so,
    l_dot_s: np.int32,
):
    r"""Returns the local radial potential

    Parameters:
        r (double) : scaled radial coordinate s = k * r
        alpha (ndarray)
        v_r (callable)
        v_so (callable)
        l_dot_s (int)
    """
    return v_r(r, alpha) + v_so(r, alpha, l_dot_s)


@njit
def potential_plus_coulomb(
    r: np.double,
    alpha: np.array,
    ZZ: np.double,
    R_C: np.double,
    v_r,
    v_so,
    l_dot_s: np.int32,
):
    r"""Returns the local radial potential + coulomb

    Parameters:
        r (double) : scaled radial coordinate s = k * r
        alpha (ndarray)
        ZZ (double),
        R_C (double)
        v_r (callable)
        v_so (callable)
        l_dot_s (int)
    """
    return potential(r, alpha, v_r, v_so, l_dot_s) + coulomb_charged_sphere(r, R_C, ZZ)


@njit
def potential_plus_coulomb_scaled(
    s: np.double,
    alpha: np.array,
    k: np.double,
    S_C: np.double,
    E: np.double,
    eta: np.double,
    l: np.int32,
    v_r,
    v_so,
    l_dot_s: np.int32,
):
    r"""Returns the  scaled, reduced, radial local potential

    Parameters:
        s (double) : scaled radial coordinate s = k * r
        alpha (ndarray)
        k (double)
        S_C (double)
        E (double)
        eta (double)
        l (int)
        v_r (callable)
        v_so (callable)
        l_dot_s (int)
    """
    return (
        v_r(s / k, alpha) + v_so(s / k, alpha, l_dot_s)
    ) / E + 2 * eta * regular_inverse_s(s, S_C)


@njit
def g_coeff(
    s: np.double,
    alpha: np.array,
    k: np.double,
    S_C: np.double,
    E: np.double,
    eta: np.double,
    l: np.int32,
    v_r,
    v_so,
    l_dot_s: np.int32,
):
    r"""Returns the coefficient g(s )for the parameteric differential equation y'' + g(s, args)  y = 0
        for the case where y is the wavefunction solution of the scaled, reduced, radial Schrödinger
        equation in the continuum with local, complex potential depending on parameters alpha

    Returns:
        value of g(x) = [V(x;...)/E + l(l+1)/x**2 + 2 eta/x -1] , given other args to g

    Parameters:
        s (double) : scaled radial coordinate s = k * r
        alpha (ndarray)
        k (double)
        S_C (double)
        E (double)
        eta (double)
        l (int)
        v_r (callable)
        v_so (callable)
        l_dot_s (int)
    """

    return -1 * (
        potential_plus_coulomb_scaled(s, alpha, k, S_C, E, eta, l, v_r, v_so, l_dot_s)
        + l * (l + 1) / s**2
        - 1.0
    )


@njit
def numerov_kernel(
    g,
    g_args: tuple,
    domain: tuple,
    dx: np.double,
    initial_conditions: tuple,
):
    r"""Solves the parametric differential equation y'' + g(x; g_args)  y = 0 for y via the
        Numerov method, for complex function y over real domain x

    Returns:
        y (ndarray) : values of y evaluated at the points x_grid

    Parameters:
        g : callable for g(x), returns a complex value given real argument
        g_args : any other arguments for `g`, passed as g(x, *g_args)
        domain (tuple) : the bounds of the x domain
        dx (float) : the step size to use in the x domain
        initial_conditions (list) : the value of y and y' at the minimum of x_grid
    """
    # convenient factor
    f = dx * dx / 12.0

    # intial conditions
    x0, xf = domain
    xnm = x0
    (y0, y0_prime) = (initial_conditions[0], initial_conditions[1])

    # number of steps
    N = int(np.ceil((xf - x0) / dx))

    # use Taylor expansion for y1
    y0_dbl_prime = -g(x0, *g_args) * y0
    y1 = y0 + y0_prime * dx + y0_dbl_prime * dx**2 / 2

    # initialize range walker
    y = np.empty(N, dtype=np.cdouble)
    y[0] = y0
    y[1] = y1

    ynm = y0
    yn = y1

    gnm = g(xnm, *g_args)
    gn = g(xnm + dx, *g_args)

    def forward_stepy(n, yn, ynp):
        y[n] = ynp
        return yn, ynp

    for n in range(2, N):
        # determine next y
        gnp = g(xnm + dx + dx, *g_args)
        ynp = (2 * yn * (1.0 - 5.0 * f * gn) - ynm * (1.0 + f * gnm)) / (1.0 + f * gnp)

        # forward step
        ynm, yn = forward_stepy(n, yn, ynp)
        xnm += dx

        gnm = gn
        gn = gnp

    return y


@njit
def numerov_kernel_meshless(
    g,
    g_args: tuple,
    domain: tuple,
    dx: np.double,
    initial_conditions: tuple,
    output_size: int = 8,
):
    r"""Solves the parametric differential equation y'' + g(x; g_args)  y = 0 for y via the
        Numerov method, for complex function y over real domain x

    Returns:
        x (ndarray): values of x up to x_f + dx
        y (ndarray): y evaluated at those x values

    Parameters:
        g : callable for g(x), returns a complex value given real argument
        g_args : any other arguments for `g`, passed as g(x, *g_args)
        domain (tuple) : [x0, xf] the bounds of the x domain
        dx (float) : the step size to use in the x domain
        initial_conditions (list) : the value of y and y' at the minimum of x_grid
        output_size (int) : how many values of x,y to return. For example; output_size = 2
            corresponds to returning x = [x_f, x_f + dx], and y evaluated at those x values. 3
            would be x = [x_f - dx, x_f, x_f + dx], and so on. Must be positive integer (not 0).
    """
    # convenient factor
    f = dx * dx / 12.0

    # intial conditions
    x0, xf = domain
    xnm = x0
    (y0, y0_prime) = (initial_conditions[0], initial_conditions[1])

    # use Taylor expansion for y1
    y0_dbl_prime = -g(x0, *g_args) * y0
    y1 = y0 + y0_prime * dx + y0_dbl_prime * dx**2 / 2

    # set up y array
    y = np.empty(output_size, dtype=np.cdouble)
    y[0] = y0
    y[1] = y1
    ynm = y[0]
    yn = y[1]
    idx = 2

    gnm = g(xnm, *g_args)
    gn = g(xnm + dx, *g_args)

    N = int(np.ceil((xf + dx - x0) / dx))
    x = np.linspace(xf - (output_size - 2) * dx, xf + dx, output_size)

    for n in range(2, N):
        # determine next y
        gnp = g(xnm + 2 * dx, *g_args)

        # index into y array
        y[idx] = (2 * yn * (1.0 - 5.0 * f * gn) - ynm * (1.0 + f * gnm)) / (
            1.0 + f * gnp
        )

        # forward step
        ynm = yn
        yn = y[idx]
        xnm += dx

        gnm = gn
        gn = gnp

        idx += 1
        idx = idx % 8  # if we reach 8, go back to 0

    # cyclic indexing means we need to split and merge back to proper order
    y = np.hstack((y[idx:], y[:idx]))

    return x, y
