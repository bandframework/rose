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
    Extension of InteractionEIM that supports energy and, optionally, mu as parameters. Expected format
    for alpha depends on optional parameters passed in during initialization.
    """

    def __init__(
        self,
        coordinate_space_potential: Callable[[float, np.array], float],  # V(r, theta)
        n_theta: int,  # How many parameters does the interaction have?
        mu: float,  # reduced mass (MeV)
        ell: int,
        training_info: np.array,
        Z_1: int = 0,  # atomic number of particle 1
        Z_2: int = 0,  # atomic number of particle 2
        R_C: float = 0.0,  # Coulomb "cutoff"
        is_complex: bool = False,
        spin_orbit_term: SpinOrbitTerm = None,
        n_basis: int = None,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None,
        method="collocation",
    ):
        r"""
        Parameters:
            coordinate_space_potential (Callable[[float,ndarray],float]): V(r,
                theta) where theta are the interaction parameters
            n_theta (int): number of interaction parameters
            mu (float): reduced mass (MeV); By default, energy (in Mev) is expected to be in position 0
                of the param array; alpha. If a value of mu i
                n MeV is provided, then the rest of alpha (positons 1:), are expected to the parameters
                passed to `coordinate_space_potential`; otherwise, is the value of mu passed
                is None, mu in MeV is expected to be in position 1 of alpha, and alpha[2:] will be
                passed to `coordinate_space_potential`.
            ell (int): angular momentum
            training_info (ndarray): Either (1) parameters bounds or (2)
                explicit training points

                If (1):
                    This is a 2-column matrix. The first column are the lower
                    bounds. The second are the upper bounds. Each row maps to a
                    single parameter.

                If (2):
                    This is an MxN matrix. N is the number of parameters. M is
                    the number of samples.
            Z_1 (int): charge of particle 1
            Z_2 (int): charge of particle 2
            R_C (float): Coulomb "cutoff" radius
            is_complex (bool): Is the interaction complex (e.g. optical
                potentials)?
            spin_orbit_term (SpinOrbitTerm): spin-orbit part of the interaction
            n_basis (int): number of basis states, or "pillars" in $\hat{U}$ approximation
            explicit_training (bool): Is training_info (1) or (2)? (1) is
                default
            n_train (int): How many snapshots to generate? Ignored if
                explicit_training is True.
            rho_mesh (ndarray): coordinate-space points at which the interaction
                is generated (used for training)
            match_points (ndarray): $\rho$ points where agreement with the true
                potential is enforced
            method (str) : 'collocation' or 'least-squares'. If 'collocation', match_points must be the
                same length as n_basis; otherwise match_points can be any size.

        Attributes:
            singular_values (ndarray): `S` in `U, S, Vt = numpy.linalg.svd(...)`
            snapshots (ndarray): pillars, columns of `U`
            match_indices (ndarray): indices of points in $\rho$ mesh that are
                matched to the true potential
            match_points (ndarray): points in $\rho$ mesh that are matched to
                the true potential
            r_i (ndarray): copy of `match_points` (???)
            Ainv (ndarray): inverse of A matrix (Ax = b)
        """
        self.param_mask = np.ones((n_theta), dtype=bool)
        if mu is None:
            self.param_mask[:2] = False
        else:
            self.param_mask[:1] = False
            self.reduced_mass = lambda alpha: self.mu

        super().__init__(
            coordinate_space_potential,
            n_theta,
            mu,
            None,
            ell,
            training_info,
            Z_1=Z_1,
            Z_2=Z_2,
            R_C=R_C,
            is_complex=is_complex,
            spin_orbit_term=spin_orbit_term,
            n_basis=n_basis,
            explicit_training=explicit_training,
            n_train=n_train,
            rho_mesh=rho_mesh,
            match_points=match_points,
            method="collocation",
        )

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
        alpha_truncated = alpha[self.param_mask]
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
        r"""Center-of-mass, scattering momentum

        Parameters:
            alpha (ndarray): interaction parameters

        Returns:
            k (float): momentum
        """
        return np.sqrt(2 * self.reduced_mass(alpha) * self.E(alpha)) / HBARC

    def E(self, alpha: np.array):
        r"""Energy. Implemented as a function to support energy
        emulation (where the energy could be a part of the parameter vector,
        `alpha`).

        Parameters:
            alpha (ndarray): parameter vector

        Returns:
            Energy (float): in [MeV]
        """
        return alpha[0]

    def reduced_mass(self, alpha: np.array):
        r"""Mu. Implemented as a function to support energy
        emulation (where the mu could be a part of the parameter vector,
        `alpha`).

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
        return (alpha[self.param_mask], k, S_C, E, eta, l, v_r, v_so, l_dot_s)


class EnergizedInteractionEIMSpace(InteractionEIMSpace):
    def __init__(
        self,
        coordinate_space_potential: Callable[[float, np.array], float],  # V(r, theta)
        n_theta: int,  # How many parameters does the interaction have?
        mu: float,  # reduced mass (MeV)
        training_info: np.array,
        l_max: int = 20,
        Z_1: int = 0,  # atomic number of particle 1
        Z_2: int = 0,  # atomic number of particle 2
        R_C: float = 0.0,  # Coulomb "cutoff"
        is_complex: bool = False,
        spin_orbit_potential: Callable[
            [float, np.array, float], float
        ] = None,  # V_{SO}(r, theta, l•s)
        n_basis: int = None,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None,
        method="collocation",
    ):
        r"""Generates a list of $\ell$-specific, energy-emulated, EIMed interactions.

        Parameters:
            coordinate_space_potential (Callable[[float,ndarray],float]): V(r, theta)
            n_theta (int): number of parameters
            mu (float): reduced mass
            training_info (ndarray): See `InteractionEIM` documentation.
            l_max (int): maximum angular momentum
            Z_1 (int): charge of particle 1
            Z_2 (int): charge of particle 2
            R_C (float): Coulomb "cutoff" radius
            is_complex (bool): Is the interaction complex?
            spin_orbit_potential (Callable[[float, np.array, float], float]):
                used to create a `SpinOrbitTerm`
            n_basis (int): number of pillars --- basis states in $\hat{U}$ expansion
            explicit_training (bool): See `InteractionEIM` documentation.
            n_train (int): number of training samples
            rho_mesh (ndarray): discrete $\rho$ points
            match_points (ndarray): $\rho$ points where agreement with the true
                potential is enforced

        Returns:
            instance (InteractionEIMSpace): instance of InteractionEIMSpace

        Attributes:
            interaction (list): list of `InteractionEIM`s
        """
        self.l_max = l_max
        self.interactions = []
        if spin_orbit_potential is None:
            for l in range(l_max + 1):
                self.interactions.append(
                    [
                        EnergizedInteractionEIM(
                            coordinate_space_potential,
                            n_theta,
                            mu,
                            l,
                            training_info,
                            Z_1=Z_1,
                            Z_2=Z_2,
                            R_C=R_C,
                            is_complex=is_complex,
                            n_basis=n_basis,
                            explicit_training=explicit_training,
                            n_train=n_train,
                            rho_mesh=rho_mesh,
                            match_points=match_points,
                            method=method,
                        )
                    ]
                )
        else:
            for l in range(l_max + 1):
                self.interactions.append(
                    [
                        EnergizedInteractionEIM(
                            coordinate_space_potential,
                            n_theta,
                            mu,
                            l,
                            training_info,
                            Z_1=Z_1,
                            Z_2=Z_2,
                            R_C=R_C,
                            is_complex=is_complex,
                            spin_orbit_term=SpinOrbitTerm(spin_orbit_potential, lds),
                            n_basis=n_basis,
                            explicit_training=explicit_training,
                            n_train=n_train,
                            rho_mesh=rho_mesh,
                            match_points=match_points,
                            method=method,
                        )
                        for lds in couplings(l)
                    ]
                )
