r"""`NumerovSolver` is another high-fidelity (HF) Schrödinger-equation solver for
local, complex, single-channel interactions, which uses numba.njit for JIT compilation
"""

from collections.abc import Callable

import numpy as np
from scipy.interpolate import splrep, splev

from .utility import (
    regular_inverse_s,
    numerov_kernel_meshless,
    numerov_kernel,
    g_coeff,
)
from .interaction import Interaction
from .schroedinger import SchroedingerEquation


class NumerovSolver(SchroedingerEquation):
    """
    Solver for the single-channel, reduced, radial Schrödinger equation with a local, complex
    potential, using the Numerov method: https://en.wikipedia.org/wiki/Numerov%27s_method
    """

    def __init__(
        self,
        interaction: Interaction,
        domain: tuple,
        dx: np.double,
    ):
        r"""

        Parameters:
            interaction (Interaction): See [Interaction documentation](interaction.md). To use
                `NumerovSolver`, the radial interaction, including the spin-orbit term, must be
                decorated with @njit
            domain (tuple) : the upper and lower bounds of the problem domain $s$
            dx (double) : the step size with which to integrate over the domain. The numerov method
                is O(dx**4) in error for a single step

        Returns:
            solver (NumerovSolver): instance of `NumerovSolver`

        """
        self.domain = domain
        self.dx = dx
        self.s_mesh = np.arange(self.domain[0], self.domain[1], self.dx)
        self.mesh_size = self.s_mesh.shape[0]

        super().__init__(interaction, None)

    def clone_for_new_interaction(self, interaction: Interaction):
        return NumerovSolver(interaction, self.domain, self.dx)

    def phi(
        self,
        alpha: np.array,
        s_mesh: np.array = None,
        l: int = None,
        rho_0=None,
        phi_threshold=SchroedingerEquation.PHI_THRESHOLD,
    ):
        r"""Computes the reduced, radial wave function $\phi$ (or $u$) on `s_mesh` using the
        Numerov method.

        Parameters:
            alpha (ndarray): parameter vector
            s_mesh (ndarray): values of $s$ at which $\phi$ is evaluated; uses self.mesh if None.
                If s_mesh is supplied, simply calculates on self.s_mesh and interpolates solution
                onto s_mesh. Must be contained in the domain.
            l (int): angular momentum
            rho_0 (float): starting point for the solver
            phi_threshold (float): minimum $\phi$ value; The wave function is
                considered zero below this value.

        Returns:
            phi (ndarray): reduced, radial wave function

        """
        if l is None:
            l = self.interaction.ell

        if s_mesh is not None:
            assert s_mesh[0] >= self.domain[0]
            assert s_mesh[1] <= self.domain[1]

        # determine initial conditions
        rho_0, initial_conditions = self.initial_conditions(
            alpha, phi_threshold, l, rho_0
        )

        args = self.interaction.bundle_gcoeff_args(alpha)
        y = numerov_kernel(
            g_coeff,
            args,
            self.domain,
            self.dx,
            initial_conditions,
        )

        if s_mesh is None:
            return y
        else:
            mask = np.where(self.s_mesh < rho_0)[0]
            y[mask] = 0
            return np.interp(s_mesh, self.s_mesh, y)

    def rmatrix(
        self,
        alpha: np.array,
        s_0: float,
        l: int = None,
        domain=None,
        phi_threshold=SchroedingerEquation.PHI_THRESHOLD,
    ):
        r"""Calculates the $\ell$-th partial wave R-matrix element at the specified energy,
            using the Numerov method for integrating the Radial SE

        Parameters:
            alpha (ndarray): parameter vector
            l (int): angular momentu the scaled radial potentialm
            rho_0 (float): initial $\rho$ (or $s$) value; starting point for the
                solver
            domain : unused
            phi_threshold (float): minimum $\phi$ value; The wave function is
                considered zero below this value.

        Returns:
            rl (float)  : r-matrix element, or logarithmic derivative of wavefunction at the channel
                radius; s_0
        """
        if l is None:
            l = self.interaction.ell

        assert s_0 >= self.domain[0] and s_0 < self.domain[1]

        # determine initial conditions
        rho_0, initial_conditions = self.initial_conditions(alpha, phi_threshold, l)

        args = self.interaction.bundle_gcoeff_args(alpha)
        x, y = numerov_kernel_meshless(
            g_coeff,
            args,
            (self.domain[0], s_0),
            self.dx,
            initial_conditions,
        )

        spl_real = splrep(x, y.real)
        spl_imag = splrep(x, y.imag)
        u = splev(s_0, spl_real) + 1j * splev(s_0, spl_imag)
        uprime = splev(s_0, spl_real, der=1) + 1j * splev(s_0, spl_imag, der=1)
        rl = 1.0 / s_0 * (u / uprime)
        return rl
