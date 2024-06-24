"""Wraps the user-defined interaction into a class that stores several relevant
parameters of the problem.
"""

from typing import Callable
import numpy as np
from numba import njit

from .constants import HBARC, ALPHA
from .spin_orbit import SpinOrbitTerm


class Interaction:
    """Defines a local, (possibly) complex, affine, fixed-energy interaction."""

    def __init__(
        self,
        ell: int = 0,
        spin_orbit_term: SpinOrbitTerm = None,
        coordinate_space_potential: Callable[[float, np.array], float] = None,
        n_theta: int = None,
        mu: float = None,
        energy: float = None,
        k: float = None,
        Z_1: int = 0,
        Z_2: int = 0,
        R_C: float = 0.0,
        is_complex: bool = False,
    ):
        r"""Creates a local, (possibly) complex, affine, fixed-energy interaction.

        Parameters:
            coordinate_space_potential (Callable[[float,ndarray],float]): V(r, theta)
            n_theta (int): number of parameters
            mu (float): reduced mass
            energy (float): center-of-mass, scattering energy
            ell (int): angular momentum
            Z_1 (int): charge of particle 1
            Z_2 (int): charge of particle 2
            R_C (float): Coulomb "cutoff" radius
            is_complex (bool): Is the interaction complex?
            spin_orbit_term (SpinOrbitTerm): See [Spin-Orbit section](#spin-orbit).

        Returns:
            instance (Interaction): instance of `Interaction`

        Attributes:
            v_r (Callable[[float,ndarray],float]): coordinate-space potential; $V(r, \alpha)$
            n_theta (int): number of parameters
            mu (float): reduced mass
            ell (int): angular momentum
            k_c (float): Coulomb momentum; $k\eta$
            is_complex (bool): Is this a complex potential?
            spin_orbit_term (SpinOrbitTerm): See [Spin-Orbit section](#spin-orbit)

        """
        assert coordinate_space_potential is not None
        assert n_theta > 0

        assert ell >= 0
        self.Z_1 = Z_1
        self.Z_2 = Z_2
        self.v_r = coordinate_space_potential
        self.n_theta = n_theta
        self.ell = ell
        self.R_C = R_C
        self.is_complex = is_complex
        self.spin_orbit_term = spin_orbit_term

        if spin_orbit_term is None:
            self.include_spin_orbit = False
        else:
            self.include_spin_orbit = True

        self.k = k
        self.mu = mu
        self.energy = energy
        self.sommerfeld = 0.0

        if mu is not None:
            self.k_c = ALPHA * Z_1 * Z_2 * self.mu / HBARC
        else:
            self.k_c = 0  # TODO mu/energy emulation does not support Coulomb
            assert self.Z_1 * self.Z_1 == 0

        if energy is not None:
            # If the energy is specified (not None as it is when subclass
            # EnergizedInteraction instantiates), set up associated attributes.
            if mu is not None and k is None:
                self.k = np.sqrt(2 * self.mu * self.energy) / HBARC
            self.sommerfeld = self.k_c / self.k

    def tilde(self, s: float, alpha: np.array):
        r"""Scaled potential, $\tilde{U}(s, \alpha, E)$.

        * Does not include the Coulomb term.
        * $E = E_{c.m.}$; `[E] = MeV = [v_r]`

        Parameters:
            s (float): mesh point; $s = pr/\hbar$
            alpha (ndarray): the varied parameters

        Returns:
            u_tilde (float | complex): value of scaled interaction

        """
        vr = self.v_r(s / self.k, alpha) + self.spin_orbit_term.spin_orbit_potential(
            s / self.k, alpha
        )
        return 1.0 / self.energy * vr

    def basis_functions(self, rho_mesh: np.array):
        r"""In general, we approximate the potential as

        $\hat{U} = \sum_{j} \beta_j(\alpha) u_j$

        For affine interactions (like those defined in this class) the basis
        functions (or "pillars), $u_j$, are just the "naked" parts of the
        potential. As seen below, it is assumed that the $\beta_j(\alpha)$
        coefficients are just the affine parameters, $\alpha$, themselves.

        Parameters:
            rho_mesh (ndarray): discrete $\rho$ values at which the potential is
                going to be evaluated

        Returns:
            value (ndarray): values of the scaled potential at provided $\rho$ points
        """
        return np.array([self.tilde(rho_mesh, row) for row in np.eye(self.n_theta)]).T

    def coefficients(self, alpha: np.array):  # interaction parameters
        r"""As noted in the `basis_functions` documentation, the coefficients
        for affine interactions are simply the parameter values. The inverse of
        the momentum is also returned to support energy emulation.

        Parameters:
            alpha (ndarray): parameter point

        Returns:
            result (tuple): inverse momentum and coefficients

        """
        return 1 / self.k, alpha

    def eta(self, alpha: np.array):
        r"""Sommerfeld parameter. Implemented as a function to support energy
        emulation (where the energy could be a part of the parameter vector,
        `alpha`).

        Parameters:
            alpha (ndarray): parameter vector

        Returns:
            eta (float): Sommerfeld parameter
        """
        return self.sommerfeld

    def E(self, alpha: np.array):
        r"""Energy. Implemented as a function to support energy
        emulation (where the energy could be a part of the parameter vector,
        `alpha`).

        Parameters:
            alpha (ndarray): parameter vector

        Returns:
            Energy (float): in [MeV]
        """
        return self.energy

    def momentum(self, alpha: np.array):
        r"""Momentum. Implemented as a function to support energy emulation
        (where the energy/momentum could be a part of the parameter vector,
        `alpha`).

        Parameters:
            alpha (ndarray): parameter vector

        Returns:
            k (float): center-of-mass, scattering momentum
        """
        return self.k

    def coulomb_cutoff(self, alpha: np.array):
        r"""Coulomb cutoff. Implemented as a function to support energy emulation
        (where the energy/momentum could be a part of the parameter vector,
        `alpha`).

        Parameters:
            alpha (ndarray): parameter vector

        Returns:
            R_C (float): Coulomb cutoff
        """
        return self.R_C

    def bundle_gcoeff_args(self, alpha: np.array):
        r"""Bundles parameters for the Schrödinger equation

        Returns:
            args (tuple) : all the arguments to g_coeff except for $s$

        Parameters:
            alpha (ndarray) : the parameters for the interaction
        """
        k = self.momentum(alpha)
        S_C = self.coulomb_cutoff(alpha) * k
        E = self.E(alpha)
        eta = self.eta(alpha)
        l = self.ell
        v_r = self.v_r
        if self.include_spin_orbit:
            l_dot_s = self.spin_orbit_term.l_dot_s
            v_so = self.spin_orbit_term.v_so
        else:
            l_dot_s = 0
            v_so = None

        return (alpha, k, S_C, E, eta, l, v_r, v_so, l_dot_s)


class InteractionSpace:
    def __init__(
        self,
        l_max: int = 15,
        interaction_type=Interaction,
        **kwargs,
    ):
        r"""Generates a list of $\ell$-specific interactions.

        Parameters:
            l_max (int): maximum angular momentum
            interaction_type (Type): type of `Interaction` to construct
            kwargs (dict): arguments to constructor of `interaction_type`

        Returns:
            instance (InteractionSpace): instance of InteractionSpace

        Attributes:
            interaction (list): list of `Interaction`s
            l_max (int): partial wave cutoff
            type (Type): interaction type
        """
        self.l_max = l_max
        self.type = interaction_type
        self.interactions = []

        if "spin_orbit_term" not in kwargs:
            for l in range(self.l_max + 1):
                self.interactions.append([self.type(ell=l, **kwargs)])
        else:
            spin_orbit_potential = kwargs["spin_orbit_term"]
            kwargs.pop("spin_orbit_term")

            for l in range(self.l_max + 1):
                self.interactions.append(
                    [
                        self.type(
                            ell=l,
                            spin_orbit_term=SpinOrbitTerm(spin_orbit_potential, lds),
                            **kwargs,
                        )
                        for lds in couplings(l)
                    ]
                )


def couplings(l):
    r"""For a spin-1/2 nucleon scattering off a spin-0 nucleus, there are
    maximally 2 different total angular momentum couplings: l+1/2 and l-1/2.

    Parameters:
        l (int): angular momentum

    Returns:
        couplings (list): epectation value of l dot s
    """
    js = [l + 1.0 / 2] if l == 0 else [l + 1.0 / 2, l - 1.0 / 2]
    return [(j * (j + 1) - l * (l + 1) - 0.5 * (0.5 + 1)) for j in js]
