r"""`SchroedingerEquation` is a high-fidelity (HF), Schrödinger-equation solver for
local, complex interactions.

By default, `rose` will provide HF solution using `scipy.integrate.solve_ivp`.
For details about providing your own solutions, see [Basis
documentation](basis.md).

"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.misc import derivative

from .interaction import Interaction
from .free_solutions import H_minus, H_plus, H_minus_prime, H_plus_prime
from .utility import regular_inverse_s


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
    DEFAULT_S_MAX = 10 * np.pi  # dimensionless
    DEFAULT_R_0 = 20.0  # fm
    DEFAULT_NUM_PTS = 2000
    MAX_STEPS = 20000
    PHI_THRESHOLD = 1e-10

    def __init__(
        self,
        interaction: Interaction,
        RK_tolerances: list = [1e-7, 1e-7],
    ):
        r"""Solves the Shrödinger equation for local, complex potentials.

        Parameters:
            interaction (Interaction): See [Interaction documentation](interaction.md).
            RK_tolerances (list): 2-element list of numbers specifying tolerances for the
                Runge-Kutta solver: the relative tolerance and the  absolute tolerance

        Returns:
            solver (SchroedingerEquation): instance of `SchroedingerEquation`

        """
        self.interaction = interaction
        self.rk_tols = RK_tolerances

    def clone_for_new_interaction(self, interaction: Interaction):
        return SchroedingerEquation(interaction, self.rk_tols)

    def initial_conditions(
        self, alpha: np.array, phi_threshold: float, l: int, rho_0=None
    ):
        r"""
        Returns:
            initial_conditions (tuple) : initial conditions [phi, phi'] at rho_0

        Parameters:
            alpha (ndarray): parameter vector
            phi_threshold (float): minimum $\phi$ value; The wave function is
                considered zero below this value.
            rho_0 (float): starting point for the solver
        """

        C_l = Gamow_factor(l, self.interaction.eta(alpha))

        if rho_0 is None:
            rho_0 = (phi_threshold / Gamow_factor(l, self.interaction.eta(alpha))) ** (
                1 / (l + 1)
            )

        phi_0 = C_l * rho_0 ** (l + 1)
        phi_prime_0 = C_l * (l + 1) * rho_0**l

        if self.interaction.is_complex:
            initial_conditions = np.array([phi_0 * (1 + 0j), phi_prime_0 * (1 + 0j)])
        else:
            initial_conditions = np.array([phi_0, phi_prime_0])

        return rho_0, initial_conditions

    def radial_se_deriv2(self, s, l, alpha, S_C):
        r"""Evaluates the coefficient of y in RHSof the radial reduced Schroedinger equation as below:
            $u'' = (\tilde{U}(s, \alpha) + l(l+1) f(s) + 2 eta / s + \tilde{U}_{so}(s, \alpha) - 1.0)u$

            where $f(s)$ is the form of the Coulomb term (a function of only `S_C`).

        Returns:
            (float) : RHS of the scaled SE, with the LHS being the second derivative operator.
            This value, multiplied by the value of the radial wavefunction, gives its second derivative

        Parameters:
            alpha (ndarray): parameter vector
            s (float): values of dimensionless radial coordinate $s=kr$
            l (int): angular momentum
            S_C (float) : Coulomb cutoff (charge radius)
        """
        return (
            self.interaction.tilde(s, alpha)
            + 2 * self.interaction.eta(alpha) * regular_inverse_s(s, S_C)
            + l * (l + 1) / s**2
            - 1.0
        )

    def solve_se(
        self,
        alpha: np.array,  # interaction parameters
        domain: np.array,  # s where phi(s) is calculated
        l: int = 0,  # angular momentum
        rho_0=None,  # initial rho value ("effective zero")
        phi_threshold=PHI_THRESHOLD,  # minimum phi value (zero below this value)
        **kwargs,
    ):
        r"""Solves the reduced, radial Schrödinger equation using the builtin in Runge-Kutta
            solver in scipy.integrate.solve_ivp

        Parameters:
            alpha (ndarray): parameter vector
            domain (ndarray): lower and upper bounds of the $s$ mesh.
            l (int): angular momentum
            rho_0 (float): initial $\rho$ (or $s$) value; starting point for the
                solver
            phi_threshold (float): minimum $\phi$ value; The wave function is
                considered zero below this value.

        Returns:
            sol  (scipy.integrate.OdeSolution) : the radial wavefunction
            and its first derivative; see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolution.html#scipy.integrate.OdeSolution
        """
        rho_0, initial_conditions = self.initial_conditions(
            alpha, phi_threshold, l, rho_0
        )
        S_C = self.interaction.momentum(alpha) * self.interaction.coulomb_cutoff(alpha)

        sol = solve_ivp(
            lambda s, phi: np.array(
                [
                    phi[1],
                    self.radial_se_deriv2(s, l, alpha, S_C) * phi[0],
                ]
            ),
            [rho_0, domain[1]],
            initial_conditions,
            rtol=self.rk_tols[0],
            atol=self.rk_tols[1],
            dense_output=True,
            **kwargs,
        )

        return sol.sol, rho_0

    def rmatrix(
        self,
        alpha: np.array,
        l: int,
        s_0: float,
        domain=[DEFAULT_S_MIN, DEFAULT_S_MAX],
        **kwargs,
    ):
        r"""Calculates the $\ell$-th partial wave R-matrix element at the specified energy.
            using the Runge-Kutta method for integrating the Radial SE. kwargs are passed to
            solve_se.

        Parameters:
            alpha (ndarray): parameter vector
            l (int): angular momentum
            s_0 (float): $s$ value where the phase shift is calculated (must be
                less than the second element in `domain`)
            domain (ndarray): lower and upper bounds of the $s$ mesh.

        Returns:
            rl (float)  : r-matrix element, or logarithmic derivative of wavefunction at the channel
                radius; s_0

        """
        # Should domain be [s_min, domain[1]]?
        solution, _ = self.solve_se(alpha, domain, l=l, **kwargs)
        u = solution(s_0)
        rl = 1 / s_0 * (u[0] / u[1])
        return rl

    def phi(
        self,
        alpha: np.array,  # interaction parameters
        s_mesh: np.array,  # s where phi(s) in calculated
        l: int,  # angular momentum
        rho_0: float = None,  # What do we call "zero"?
        phi_threshold: float = PHI_THRESHOLD,
        **kwargs,  # passed to solve_se
    ):
        r"""Computes the reduced, radial wave function $\phi$ (or $u$) on `s_mesh` using the
        Runge-Kutta method

        Parameters:
            alpha (ndarray): parameter vector
            s_mesh (ndarray): values of $s$ at which $\phi$ is calculated
            l (int): angular momentum
            rho_0 (float): starting point for the solver
            phi_threshold (float): minimum $\phi$ value; The wave function is
                considered zero below this value.

        Returns:
            phi (ndarray): reduced, radial wave function

        """
        solution, rho_0 = self.solve_se(
            alpha,
            [rho_0, s_mesh[-1]],
            l,
            rho_0=rho_0,
            phi_threshold=phi_threshold,
            **kwargs,
        )

        mask = np.where(s_mesh < rho_0)
        y = solution(s_mesh)[0]
        y[mask] = 0
        return y

    def delta(
        self,
        alpha: np.array,  # interaction parameters
        l: int,  # angular momentum
        s_0: float,  # phaseshift is extracted at phi(s_0)
        domain=[DEFAULT_S_MIN, DEFAULT_S_MAX],
        **kwargs,  # passed to solver
    ):
        r"""Calculates the $\ell$-th partial wave phase shift

        Parameters:
            alpha (ndarray): parameter vector
            domain (ndarray): lower and upper bounds of the $s$ mesh.
            l (int): angular momentum
            s_0 (float): $s$ value where the phase shift is calculated (must be
                less than the second element in `domain`)

        Returns:
            delta (float): phase shift extracted from the reduced, radial
                wave function

        """
        if domain[1] <= s_0:
            domain = s_0 + 0.1

        rl = self.rmatrix(alpha, l, s_0, domain=domain, **kwargs)

        return (
            np.log(
                (
                    H_minus(s_0, l, self.interaction.eta(alpha))
                    - s_0 * rl * H_minus_prime(s_0, l, self.interaction.eta(alpha))
                )
                / (
                    H_plus(s_0, l, self.interaction.eta(alpha))
                    - s_0 * rl * H_plus_prime(s_0, l, self.interaction.eta(alpha))
                )
            )
            / 2j
        )


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
