r"""`NumerovSolver` is another high-fidelity (HF) Schrödinger-equation solver for
local, complex, single-channel interactions, which uses numba.njit for JIT compilation
"""

from collections.abc import Callable

import numpy as np
from scipy.interpolate import splrep, splev

from .njit_solver_utils import (
    numerov_kernel_meshless,
    numerov_kernel,
    g_coeff,
    bundle_gcoeff_args,
)
from .utility import regular_inverse_s
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
        return NumerovSolver(interaction, self.mesh_size, self.domain)

    def phi(
        self,
        alpha: np.array,
        s_mesh: np.array = None,
        l: int = 0,
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
        assert s_mesh[0] >= self.domain[0]
        assert s_mesh[1] <= self.domain[1]

        # determine initial conditions
        rho_0, initial_conditions = self.initial_conditions(
            alpha, phi_threshold, l, rho_0
        )

        y = numerov_kernel(
            g_coeff,
            bundle_gcoeff_args(self.interaction, alpha),
            self.domain,
            self.dx,
            initial_conditions,
        )

        mask = np.where(self.s_mesh < rho_0)[0]
        y[mask] = 0

        if s_mesh is None:
            return y
        else:
            return np.interp(s_mesh, self.s_mesh, y)

    def rmatrix(
        self,
        alpha: np.array,
        l: int,
        s_0: float,
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
        assert s_0 >= self.domain[0] and s_0 < self.domain[1]

        # determine initial conditions
        rho_0, initial_conditions = self.initial_conditions(alpha, phi_threshold, l)

        x, y = numerov_kernel_meshless(
            g_coeff,
            bundle_gcoeff_args(self.interaction, alpha),
            (self.domain[0], s_0),
            self.dx,
            initial_conditions,
        )

        spl = splrep(x, y)
        u = splev(s_0, spl)
        uprime = splev(s_0, spl, der=1)
        rl = 1.0 / s_0 * (u / uprime)
        return rl
