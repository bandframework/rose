"""
Defines a class for "affinizing" Interactions using the Empirical Interpolation
Method (EIM).
"""

from typing import Callable
import numpy as np
from scipy.stats import qmc

from .interaction import couplings
from .interaction_eim import InteractionEIM, InteractionEIMSpace
from .constants import HBARC, DEFAULT_RHO_MESH
from .spin_orbit import SpinOrbitTerm, null


class EnergizedInteractionEIM(InteractionEIM):
    r"""
    Extension of InteractionEIM that supports energy, mu and k as parameters. Expected format
    for alpha is [energy, mu, k, *rest_of_params]
    """

    def __init__(
        self,
        **kwargs,
    ):
        r"""
        Parameters:
            kwargs (dict): arguments to InteractionEIM. Note; the `energy` argument should not
            be given, it will be ignored. Energy is the first element of alpha. If `mu is None`,
            the reduced mass will expected to be the second element of alpha.

        Attributes:
            singular_values (ndarray): `S` in `U, S, Vt = numpy.linalg.svd(...)`
            snapshots (ndarray): pillars, columns of `U`
            match_indices (ndarray): indices of points in $\rho$ mesh that are
                matched to the true potential
            match_points (ndarray): points in $\rho$ mesh that are matched to
                the true potential
            Ainv (ndarray): inverse of A matrix (Ax = b)
        """
        super().__init__(**kwargs)

    def tilde(self, s: float, alpha: np.array):
        r"""Computes the energy-scaled interaction.

        Parameters:
            s (float): mesh point
            alpha (ndarray): interaction parameters

        Returns:
            u_tilde (float | complex): energy-scaled interaction

        """
        energy = self.E(alpha)
        k = self.momentum(alpha)
        alpha_truncated = alpha[3:]
        vr = self.v_r(
            s / k, alpha_truncated
        ) + self.spin_orbit_term.spin_orbit_potential(s / k, alpha_truncated)
        return 1.0 / energy * vr

    def coefficients(self, alpha: np.array):  # interaction parameters
        r"""Computes the EIM expansion coefficients.

        Parameters:
            alpha (ndarray): interaction parameters

        Returns:
            coefficients (ndarray): EIM expansion coefficients

        """
        k = self.momentum(alpha)
        u_true = self.tilde(self.r_i, alpha)
        return 1 / k, self.Ainv @ u_true

    def eta(self, alpha: np.array):
        r"""Returns the Sommerfeld parameter.

        Parameters:
            alpha (ndarray): interaction parameters

        Returns:
            eta (float): Sommerfeld parameter

        """
        return self.k_c / self.momentum(alpha)

    def tilde_emu(self, s: float, alpha: np.array):
        r"""Emulated interaction = $\hat{U}(s, \alpha, E)$

        Parameters:
            alpha (ndarray): interaction parameters

        Returns:
            u_hat (ndarray): emulated interaction

        """
        _, x = self.coefficients(alpha)
        emu = np.sum(x * self.snapshots, axis=1)
        return emu

    def basis_functions(self, s_mesh: np.array):
        r"""$u_j$ in $\tilde{U} \approx \hat{U} \equiv \sum_j \beta_j(\alpha) u_j$

        Parameters:
            s_mesh (ndarray): $s$ mesh points

        Returns:
            u_j (ndarray): "pillars" (MxN matrix; M = number of mesh points; N = number of pillars)

        """
        return np.copy(self.snapshots)

    def momentum(self, alpha: np.array):
        r"""Center-of-mass, scattering momentum. Implemented as a function to support energy
        emulation (where k could be a part of the parameter vector, `alpha`).

        Parameters:
            alpha (ndarray): interaction parameters

        Returns:
            k (float): momentum
        """
        return alpha[2]

    def E(self, alpha: np.array):
        r"""Energy. Implemented as a function to support energy emulation (where the energy
        could be a part of the parameter vector, `alpha`).

        Parameters:
            alpha (ndarray): parameter vector

        Returns:
            Energy (float): in [MeV]
        """
        return alpha[0]

    def reduced_mass(self, alpha: np.array):
        r"""Mu. Implemented as a function to support energy emulation (where mu could be a
        part of the parameter vector, `alpha`).

        Parameters:
            alpha (ndarray): parameter vector

        Returns:
            Mu (float): in [MeV/c^2]
        """
        return alpha[1]

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
            v_so = null

        # remove the energy term for alpha, so we return just the parameters that plug into v_r
        return (alpha[3:], k, S_C, E, eta, l, v_r, v_so, l_dot_s)


class EnergizedInteractionEIMSpace(InteractionEIMSpace):
    def __init__(
        self,
        l_max: int = 15,
        interaction_type=EnergizedInteractionEIM,
        **kwargs,
    ):
        r"""Generates a list of $\ell$-specific interactions.

        Parameters:
            interaction_args (list): positional arguments for constructor of `interaction_type`
            interaction_kwargs (dict): arguments to constructor of `interaction_type`
            l_max (int): maximum angular momentum
            interaction_type (Type): type of `Interaction` to construct

        Returns:
            instance (EnergizedInteractionEIMSpaceInteractionSpace):
                instance of EnergizedInteractionEIMSpace

        Attributes:
            interaction (list): list of `Interaction`s
            l_max (int): partial wave cutoff
            type (Type): interaction type
        """
        super().__init__(
            interaction_type=interaction_type,
            l_max=l_max,
            **kwargs,
        )
