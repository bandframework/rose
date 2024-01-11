r"""`SchroedingerEquation` is a high-fidelity (HF), Schrödinger-equation solver for
local, complex interactions.

By default, `rose` will provide HF solution using `scipy.integrate.solve_ivp`.
For details about providing your own solutions, see [Basis
documentation](basis.md).

"""
import numpy as np
from scipy.integrate import solve_ivp

from .njit_solver_utils import g_coeff
from .interaction import Interaction
from .free_solutions import H_minus, H_plus, H_minus_prime, H_plus_prime
from .utility import regular_inverse_s, Gamow_factor


class SchroedingerEquation:
    """
    Solver for the single-channel, reduced, radial Schrödinger equation using scipy.integrate
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

    All other solvers for the single-channel, reduced, radial Schrödinger equation inherit from
    this class
    """

    # Default values for solving the SE.
    DEFAULT_R_MIN = 1e-2  # fm
    DEFAULT_R_MAX = 30.0  # fm
    DEFAULT_S_MIN = 1e-2  # dimensionless
    DEFAULT_S_MAX = 8 * np.pi  # dimensionless
    DEFAULT_S_EPSILON = 1.0e-4
    DEFAULT_R_0 = 20.0  # fm
    DEFAULT_NUM_PTS = 2000
    MAX_STEPS = 20000
    PHI_THRESHOLD = 1e-10

    @classmethod
    def make_base_solver(
        cls,
        rk_tols=[1e-9, 1e-9],
        s_0=None,
        domain=None,
    ):
        return SchroedingerEquation(None, rk_tols, s_0, domain)

    def __init__(
        self,
        interaction: Interaction,
        rk_tols: list = [1e-9, 1e-9],
        s_0=None,
        domain=None,
    ):
        r"""Solves the Shrödinger equation for local, complex potentials.

        Parameters:
            interaction (Interaction): See [Interaction documentation](interaction.md).
            rk_tols (list): 2-element list of numbers specifying tolerances for the
                Runge-Kutta solver: the relative tolerance and the  absolute tolerance
            s_0 (float) :
            domain (ndarray) :

        Returns:
            solver (SchroedingerEquation): instance of `SchroedingerEquation`

        """
        if s_0 is None:
            s_0 = self.DEFAULT_S_MAX - self.DEFAULT_S_EPSILON

        if domain is None:
            domain = np.array([self.DEFAULT_S_MIN, s_0 + self.DEFAULT_S_EPSILON])

        assert domain[1] > s_0

        self.s_0 = s_0
        self.domain = domain
        self.interaction = interaction
        self.rk_tols = rk_tols

        if self.interaction is not None:
            if self.interaction.k_c == 0:
                self.eta = 0
            else:
                # There is Coulomb, but we haven't (yet) worked out how to emulate
                # across energies, so we can precompute H+ and H- stuff.
                self.eta = self.interaction.eta(None)

            self.Hm = H_minus(self.s_0, self.interaction.ell, self.eta)
            self.Hp = H_plus(self.s_0, self.interaction.ell, self.eta)
            self.Hmp = H_minus_prime(self.s_0, self.interaction.ell, self.eta)
            self.Hpp = H_plus_prime(self.s_0, self.interaction.ell, self.eta)

            self.domain[0], self.init_cond = self.initial_conditions(
                self.eta,
                self.PHI_THRESHOLD,
                self.interaction.ell,
                self.domain[0],
            )

            assert self.domain[0] < self.s_0

    def clone_for_new_interaction(self, interaction: Interaction):
        return SchroedingerEquation(
            interaction, self.rk_tols, s_0=self.s_0, domain=self.domain.copy()
        )

    def initial_conditions(self, eta: float, phi_threshold: float, l: int, rho_0=None):
        r"""
        Returns:
            initial_conditions (tuple) : initial conditions [phi, phi'] at rho_0

        Parameters:
            eta (float): sommerfield param
            phi_threshold (float): minimum $\phi$ value; The wave function is
                considered zero below this value.
            rho_0 (float): starting point for the solver
        """

        C_l = Gamow_factor(l, eta)
        min_rho_0 = (phi_threshold / C_l) ** (1 / (l + 1))

        if min_rho_0 > rho_0:
            rho_0 = min_rho_0

        phi_0 = C_l * rho_0 ** (l + 1)
        phi_prime_0 = C_l * (l + 1) * rho_0**l

        if self.interaction.is_complex:
            initial_conditions = np.array([phi_0 * (1 + 0j), phi_prime_0 * (1 + 0j)])
        else:
            initial_conditions = np.array([phi_0, phi_prime_0])

        return rho_0, initial_conditions

    def solve_se(
        self,
        alpha: np.array,
        **kwargs,
    ):
        r"""Solves the reduced, radial Schrödinger equation using the builtin in Runge-Kutta
            solver in scipy.integrate.solve_ivp

        Parameters:
            alpha (ndarray): parameter vector
            kwargs (dict) : passed to scipy.integrate.solve_ivp

        Returns:
            sol  (scipy.integrate.OdeSolution) : the radial wavefunction
            and its first derivative; see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolution.html#scipy.integrate.OdeSolution
        """

        args = self.interaction.bundle_gcoeff_args(alpha)
        sol = solve_ivp(
            lambda s, phi: np.array(
                [
                    phi[1],
                    -1 * g_coeff(s, *args) * phi[0],
                ]
            ),
            self.domain,
            self.init_cond,
            rtol=self.rk_tols[0],
            atol=self.rk_tols[1],
            dense_output=True,
            **kwargs,
        )

        return sol.sol

    def rmatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        r"""Calculates the $\ell$-th partial wave R-matrix element at the specified energy.
            using the Runge-Kutta method for integrating the Radial SE. kwargs are passed to
            solve_se.

        Parameters:
            alpha (ndarray): parameter vector
            kwargs (dict) : passed to scipy.integrate.solve_ivp

        Returns:
            rl (float)  : r-matrix element, or logarithmic derivative of wavefunction at the channel
                radius; s_0

        """
        solution = self.solve_se(alpha, **kwargs)
        u = solution(self.s_0)
        rl = 1 / self.s_0 * (u[0] / u[1])
        return rl

    def phi(
        self,
        alpha: np.array,
        s_mesh: np.array,
        **kwargs,
    ):
        r"""Computes the reduced, radial wave function $\phi$ (or $u$) on `s_mesh` using the
        Runge-Kutta method

        Parameters:
            alpha (ndarray): parameter vector
            s_mesh (ndarray): values of $s$ at which $\phi$ is calculated
            kwargs (dict) : passed to scipy.integrate.solve_ivp

        Returns:
            phi (ndarray): reduced, radial wave function

        """
        solution = np.zeros_like(s_mesh, dtype=np.complex128)
        mask = s_mesh > self.domain[0]
        phi = self.solve_se(alpha, **kwargs)
        solution[mask] = phi(s_mesh)[0][mask]
        return solution

    def smatrix(
        self,
        alpha: np.array,
        **kwargs,
    ):
        rl = self.rmatrix(alpha, **kwargs)

        return (self.Hm - self.s_0 * rl * self.Hmp) / (
            self.Hp - self.s_0 * rl * self.Hpp
        )

    def delta(
        self,
        alpha: np.array,
        **kwargs,
    ):
        r"""Calculates the $\ell$-th partial wave phase shift

        Parameters:
            alpha (ndarray): parameter vector

        Returns:
            delta (float): phase shift extracted from the reduced, radial
                wave function

        """
        sl = self.smatrix(alpha, **kwargs)

        return np.log(sl) / 2j
