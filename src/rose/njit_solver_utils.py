r"""
Utilities for just-in-time (JIT) compilation of solvers of the radial Schrödinger equation
"""
from collections.abc import Callable

import numpy as np
from numba import njit

from .interaction import Interaction
from .utility import regular_inverse_s


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
        value of g(x), given other args to g

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
        (v_r(s / k, alpha) + v_so(s / k, alpha, l_dot_s)) / E
        + 2 * eta * regular_inverse_s(s, S_C)
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
