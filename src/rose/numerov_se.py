r"""`NumerovSolver` is another high-fidelity (HF) Schrödinger-equation solver for
local, complex, single-channel interactions, which uses numba.njit for JIT compilation
"""

from collections.abc import Callable

from numba import njit
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.misc import derivative

from .njit_solver_utils import run_solver, numerov_kernel, numerov_kernel_meshless
from .utility import regular_inverse_s
from .interaction import Interaction
from .schroedinger import SchroedingerEquation


class NumerovSolver(SchroedingerEquation):
    """
    Solver for the single-channel, reduced, radial Schrödinger equation using the Numerov method:
    https://en.wikipedia.org/wiki/Numerov%27s_method

    Wraps the numerov kernels in .njit_solver_utils
    """

    def __init__(
        self,
        interaction: Interaction,
        mesh_size: int,
        domain: tuple,
    ):
        r"""Solves the Shrödinger equation for local, complex potentials.

        Parameters:
            interaction (Interaction): See [Interaction documentation](interaction.md).
            mesh_size (int) : the number of grid points in the radial mesh to use
                for the Numerov solve
            domain (tuple) : the upper and lower bounds of the problem domain $s$

        Returns:
            solver (NumerovSolver): instance of `NumerovSolver`

        """
        self.domain = domain
        self.mesh_size = int(mesh_size)
        self.s_mesh = np.linspace(self.domain[0], self.domain[1], self.mesh_size)

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
        S_C = self.interaction.momentum(alpha) * self.interaction.coulomb_cutoff(alpha)

        solver_args = (
            self.s_mesh[0],
            self.s_mesh[1] - self.s_mesh[0],
            self.mesh_size,
            initial_conditions,
        )
        y = run_solver(self.interaction, alpha, solver_args, numerov_kernel)

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
        # rho_0, initial_conditions = self.initial_conditions(alpha, phi_threshold, l, self.domain[0])
        S_C = self.interaction.momentum(alpha) * self.interaction.coulomb_cutoff(alpha)

        solver_args = (
            self.s_mesh[0],
            self.s_mesh[1] - self.s_mesh[0],
            self.mesh_size,
            s_0,
            initial_conditions,
        )
        x, y = run_solver(self.interaction, alpha, solver_args, numerov_kernel_meshless)

        u = interp1d(x, y, bounds_error=True)
        rl = 1.0 / s_0 * (u(s_0) / derivative(u, s_0, 1.0e-6))
        return rl
