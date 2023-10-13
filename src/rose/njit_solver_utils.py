r"""
Utilities for just-in-time (JIT) compilation of solvers of the radial Schrödinger equation
"""
from collections.abc import Callable

import numpy as np
from numba import njit

from .interaction import Interaction


def run_solver(interaction: Interaction, alpha: np.array, solver, solver_args: tuple):
    r"""
    Runs an arbitrary solver, passing in NJIT-compatible versions of the second derivative operator
    in the reduced radial Schrödinger equation:

        $u'' = (\tilde{U}(s, \alpha) + l(l+1) f(s) + 2 eta / s + \tilde{U}_{so}(s, \alpha) - 1.0)u$

    Returns:
        output of `solver`

    Parameters:
        interaction (Interaction) : potential determining $\tilde{U}$ and $\tilde{U}_{so}$, as
            well as scattering kinematics
        alpha (ndarray) : potential parameters
        solver (Callable) : solver for reduced radial Schrödinger equation, takes in `*solver_args`,
            followed by a Callable for the second derivative operator
        solver_args (tuple) : all arguments for `solver`, packed into a tuple for safe-keeping

    """
    if interaction.include_spin_orbit:
        utilde = tilde_so_NJIT(
            interaction.v_r,
            interaction.spin_orbit_term.v_so,
            interaction.spin_orbit_term.l_dot_s,
            interaction.momentum(alpha),
            alpha,
            interaction.E(alpha),
        )
    else:
        utilde = tilde_NJIT(
            interaction.v_r,
            interaction.momentum(alpha),
            alpha,
            interaction.E(alpha),
        )

    return solver(
        *solver_args,
        radial_se_deriv2_NJIT(
            interaction.eta(alpha),
            l,
            alpha,
            S_C,
            utilde,
            factor=-1,
        ),
    )


@njit
def tilde_NJIT(
    v_r: Callable[[float, np.array], float],
    k: np.double,
    alpha: np.array,
    energy: np.double,
):
    r"""
    A just-in-time (JIT) compatible function for the scaled radial potential

    Returns:
        v (Callable): an NJIT-compilable function for the scaled radial potential as a function of s

    Parameters:
        v_r (Callable) : takes in r [fm] and alpha and returns the radial potential in MeV. Must be
            decorated with @njit.
        k (float) : wavenumber [fm^-1]
        alpha (ndarray) : parameter vector, 2nd arg passed into v_r (and spin_orbit)
        energy (float) : in [MeV]
        v_so (Callable) : same as v_r but for spin orbit term. Must be decorated with @njit
    """

    return (v_r(s / k, alpha)) / energy


@njit
def tilde_so_NJIT(
    v_r: Callable[[float, np.array], float],
    v_so: Callable[[float, np.array], float],
    l_dot_s: float,
    k: np.double,
    alpha: np.array,
    energy: np.double,
):
    r"""
    A just-in-time (JIT) compatible function for the scaled radial potential with spin-orbit

    Returns:
        v (Callable): an NJIT-compilable function for the scaled radial potential as a function of s

    Parameters:
        v_r (Callable) : takes in r [fm] and alpha and returns the radial potential in MeV. Must be
            decorated with @njit.
        v_so (Callable) : same as v_r but for spin orbit term. Must be decorated with @njit
        l_dot_s (int) : expectation value of the projection of the projectile spin onto the COM-frame
            orbital angular momentum operator
        k (float) : wavenumber [fm^-1]
        alpha (ndarray) : parameter vector, 2nd arg passed into v_r (and spin_orbit)
        energy (float) : in [MeV]
    """

    return (v_r(s / k, alpha) + v_so(s / k, alpha, l_dot_s)) / energy


@njit
def radial_se_deriv2_NJIT(
    eta: np.double,
    l: int,
    alpha: np.array,
    S_C: np.double,
    utilde: Callable[[float, np.array], float],
    factor: np.double = 1.0,
):
    r"""
    Produces a just-in-time compilable function of s evaluating the coefficient of y in RHS of the
    radial reduced Schroedinger equation as below:

        $y'' = (\tilde{U}(s, \alpha) + l(l+1) f(s) + 2 eta / s + \tilde{U}_{so}(s, \alpha) - 1.0) y$,

        where $f(s)$ is the form of the Coulomb term (a function of only `S_C`).

    Returns:
        (Callable) : RHS of the scaled Schrodinger eqn, as a function of s, where the LHS is
        the second derivative operator. The value produced at a given s, multiplied by the value
        of the radial wavefunction evaluated at the same value of ss, gives the second derivative
        of the radial wavefunction at s

    Parameters:
        eta (float) : the Sommmerfield parameter
        utilde (callable[s,alpha]->V/E) : the scaled radial potential including spin-orbit coupling
            if applicable, must be decorated with @njit
        alpha (ndarray): parameter vector
        s (float): values of dimensionless radial coordinate $s=kr$
        l (int): angular momentum
        S_C (float) : Coulomb cutoff (charge radius)
        factor (float) : optional scaling factor

    """

    return factor * (
        utilde(s) + 2 * eta * regular_inverse_s(s, S_C) + l * (l + 1) / s**2 - 1.0
    )


@njit
def numerov_kernel(
    x0: np.double,
    dx: np.double,
    N: np.int,
    initial_conditions: tuple,
    g,
):
    r"""Solves the the equation y'' + g(x)  y = 0 via the Numerov method,
    for complex functions over real domain

    Returns:
    value of y evaluated at the points x_grid

    Parameters:
        x_grid : the grid of points on which to run the solver and evaluate the solution.
                 Must be evenly spaced and monotonically increasing.
        initial_conditions : the value of y and y' at the minimum of x_grid
        g : callable for g(x)
    """

    # convenient factor
    f = dx * dx / 12.0

    # intialize domain walker
    xnm = x0

    # intial conditions
    ynm = initial_conditions[0]
    yn = ynm + initial_conditions[1] * dx

    # initialize range walker
    y = np.empty(N, dtype=np.cdouble)
    y[0] = ynm
    y[1] = yn

    def forward_stepy(n, ynm, yn, ynp):
        y[n] = ynp
        return yn, ynp

    for n in range(2, y.shape[0]):
        # determine next y
        gnm = g(xnm)
        gn = g(xnm + dx)
        gnp = g(xnm + dx + dx)
        ynp = (2 * yn * (1.0 - 5.0 * f * gn) - ynm * (1.0 + f * gnm)) / (1.0 + f * gnp)

        # forward step
        ynm, yn = forward_stepy(n, ynm, yn, ynp)
        xnm += dx

    return y


@njit
def numerov_kernel_meshless(
    x0: np.double,
    dx: np.double,
    N: np.int,
    s_0: np.double,
    initial_conditions: tuple,
    g,
):
    r"""Solves the the equation y'' + g(x)  y = 0 via the Numerov method,
    for complex functions over real domain

    Returns:
        x (ndarray): values of x [s_0 - dx, s_0, s_0 + dx]
        y (ndarray): y evaluated at those x values

    Parameters:
        x_grid : the grid of points on which to run the solver and evaluate the solution.
                 Must be evenly spaced and monotonically increasing.
        initial_conditions : the value of y and y' at the minimum of x_grid
        g : callable for g(x)
    """

    # convenient factor
    f = dx * dx / 12.0

    # intial conditions
    xnm = x0
    xn = x0 + dx
    xnp = x0 + dx + dx
    ynm = initial_conditions[0]
    yn = ynm + initial_conditions[1] * dx

    for n in range(2, N + 1):
        # determine next y
        gnm = g(xnm)
        gn = g(xnm + dx)
        gnp = g(xnm + dx + dx)
        ynp = (2 * yn * (1.0 - 5.0 * f * gn) - ynm * (1.0 + f * gnm)) / (1.0 + f * gnp)

        if s_0 >= xnm and s_0 < xnp:
            break

        # forward step
        ynm = yn
        yn = ynp

        xnm += dx
        xn += dx
        xnp += dx

    return np.array([xnm, xn, xnp]), np.array([ynm, yn, ynp])
