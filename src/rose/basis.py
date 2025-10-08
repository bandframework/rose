import pickle
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from mpmath import coulombf

from .constants import HBARC
from .schroedinger import SchroedingerEquation
from .energized_interaction_eim import EnergizedInteractionEIM


class Basis:
    """Base class / template"""

    @classmethod
    def load(cls, filename):
        r"""Loads a previously saved Basis."""
        with open(filename, "rb") as f:
            basis = pickle.load(f)
        return basis

    def __init__(
        self,
        solver: SchroedingerEquation,
        theta_train: np.array,
        rho_mesh: np.array,
        n_basis: int,
    ):
        r"""Builds a reduced basis.

        Parameters:
            solver (SchroedingerEquation): high-fidelity solver
            theta_train (ndarray): training space
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            l (int): orbital angular momentum

        Attributes:
            solver (SchroedingerEquation): high-fidelity solver
            theta_train (ndarray): training space
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            l (int): orbital angular momentum

        """
        self.solver = solver
        self.l = solver.interaction.ell
        self.theta_train = theta_train
        self.rho_mesh = rho_mesh
        self.n_basis = n_basis

    def phi_hat(self, coefficients):
        r"""Emulated wave function.

        Every basis should know how to reconstruct hat{phi} from a set of
        coefficients. However, this is going to be different for each basis, so
        we will leave it up to the subclasses to implement this.

        Parameters:
            coefficients (ndarray): expansion coefficients

        Returns:
            phi_hat (ndarray): approximate wave function

        """
        raise NotImplementedError

    def phi_exact(self, theta: np.array):
        r"""Exact wave function.

        Parameters:
            theta (ndarray): parameters
            l (int) : partial wave

        Returns:
            phi (ndarray): wave function

        """
        return self.solver.phi(theta, self.rho_mesh)

    def save(self, filename):
        """Saves a basis to file.

        Parameters:
            filename (string): name of file

        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)


class RelativeBasis(Basis):
    def __init__(
        self,
        solver: SchroedingerEquation,
        theta_train: np.array,
        rho_mesh: np.array,
        n_basis: int,
        expl_var_ratio_cutoff: float = None,
        phi_0_energy: float = None,
        use_svd: bool = True,
        center: bool = None,
        scale: bool = None,
    ):
        r"""Builds a "relative" reduced basis. This is the default choice.

        $$
        \phi_{\rm HF} \approx \hat{\phi} = \phi_0 + \sum_i c_i \tilde{\phi}_i~.
        $$

        Parameters:
            solver (SchroedingerEquation): high-fidelity solver
            theta_train (ndarray): training space
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            l (int): orbital angular momentum
            use_svd (bool): Use principal components for $\tilde{\phi}$?
            phi_0_energy (float): energy at which $\phi_0$ is calculated

        Attributes:
            solver (SchroedingerEquation): high-fidelity solver
            theta_train (ndarray): training space
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            l (int): orbital angular momentum
            phi_0 (ndarray): free solution (no interaction)
            pillars (ndarray): $\tilde{\phi}_i$
            singular_values (ndarray): singular values from SVD
            vectors (ndarray): copy of `pillars`

        """

        super().__init__(
            solver,
            theta_train,
            rho_mesh,
            n_basis,
        )

        if phi_0_energy is not None:
            k = np.sqrt(2 * self.solver.interaction.mu * phi_0_energy / HBARC**2)
            eta = self.solver.interaction.k_c / k
        else:
            if isinstance(self.solver.interaction, EnergizedInteractionEIM):
                # k_mean = np.sqrt(2*self.solver.interaction.mu*np.mean(theta_train[:, 0])/HBARC**2)
                # eta = self.solver.interaction.k_c / k_mean
                # Does not support Coulomb (yet).
                eta = 0.0
            else:
                # If the phi_0 energy is not specified, we're only going to work
                # with non-Coulombic systems (for now).
                eta = 0.0

        # Returns Bessel functions when eta = 0.
        self.phi_0 = np.array(
            [coulombf(self.l, eta, rho) for rho in self.rho_mesh], dtype=np.complex128
        )
        self.solutions = np.array([self.phi_exact(theta) for theta in theta_train]).T

        self.pillars, self.singular_values, self.phi_0 = pre_process_solutions(
            self.solutions, self.phi_0, self.rho_mesh, center, scale, use_svd
        )

        # keeping at min n_basis PC's, find cutoff
        if expl_var_ratio_cutoff is not None:
            expl_var = self.singular_values**2 / np.sum(self.singular_values**2)
            n_basis_svs = np.sum(expl_var > expl_var_ratio_cutoff)
            self.n_basis = max(n_basis_svs, self.n_basis)
        else:
            self.n_basis = n_basis

        self.vectors = self.pillars[:, : self.n_basis].copy()

    def phi_hat(self, coefficients):
        r"""Emulated wave function.

        Parameters:
            coefficients (ndarray): expansion coefficients

        Returns:
            phi_hat (ndarray): approximate wave function

        """
        return self.phi_0 + np.sum(coefficients * self.vectors, axis=1)

    def project(self, x):
        r"""
        Return projection of x onto vectors
        """
        x -= self.phi_0
        x /= np.trapz(np.absolute(x), self.rho_mesh)
        return [
            np.trapz(self.vectors[:, i].conj() * x, self.rho_mesh)
            for i in range(self.n_basis)
        ]

    def percent_explained_variance(self):
        r"""
        Returns:
            (float) : percent of variance explained in the training set by the first n_basis principal
            components
        """
        if self.singular_values is None:
            return 100
        else:
            return np.array(
                [
                    100
                    * np.sum(self.singular_values[:i] ** 2)
                    / np.sum(self.singular_values**2)
                ]
                for i in range(self.nbasis)
            )


class CustomBasis(Basis):
    def __init__(
        self,
        solutions: np.array,  # HF solutions, columns
        phi_0: np.array,  # "offset", generates inhomogeneous term
        rho_mesh: np.array,  # rho mesh; MUST BE EQUALLY SPACED POINTS!!!
        n_basis: int,
        expl_var_ratio_cutoff: float = None,
        solver: SchroedingerEquation = None,
        subtract_phi0=True,
        use_svd: bool = None,
        center: bool = None,
        scale: bool = None,
    ):
        r"""Builds a custom basis. Allows the user to supply their own.

        $$
        \phi_{\rm HF} \approx \hat{\phi} = \phi_0 + \sum_i c_i \tilde{\phi}_i~.
        $$

        Parameters:
            solutions (ndarray): HF solutions
            phi_0 (ndarray): free solution (no interaction)
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): min number of states in the expansion
            expl_var_ratio_cutoff (float) : the cutoff in sv**2/sum(sv**2), sv
                being the singular values, at which the number of kept bases is chosen
            use_svd (bool): Use principal components for $\tilde{\phi}$?

        Attributes:
            solver (SchroedingerEquation): not specified or assumed at construction
            theta_train (ndarray): not specified or assumed at construction
            rho_mesh (ndarray): discrete $s=kr$ mesh points
            n_basis (int): number of states in the expansion
            phi_0 (ndarray): free solution (no interaction)
            solutions (ndarray): HF solutions provided by the user
            pillars (ndarray): $\tilde{\phi}_i$
            singular_values (ndarray): singular values from SVD
            vectors (ndarray): copy of `pillars`
            phi_0_interp (interp1d): interpolating function for $\phi_0$
            vectors_interp (interp1d): interpolating functions for vectors (basis states)

        """

        super().__init__(solver, None, rho_mesh, n_basis)

        self.rho_mesh = rho_mesh
        self.n_basis = n_basis
        self.phi_0 = phi_0

        self.pillars, self.singular_values, self.phi_0 = pre_process_solutions(
            solutions, self.phi_0, self.rho_mesh, center, scale, use_svd, subtract_phi0
        )

        # keeping at min n_basis PC's, find cutoff
        if expl_var_ratio_cutoff is not None:
            expl_var = self.singular_values**2 / np.sum(self.singular_values**2)
            n_basis_svs = np.sum(expl_var > expl_var_ratio_cutoff)
            self.n_basis = max(n_basis_svs, self.n_basis)
        else:
            self.n_basis = n_basis

        self.vectors = self.pillars[:, : self.n_basis]

    def phi_hat(self, coefficients):
        r"""Emulated wave function.

        Parameters:
            coefficients (ndarray): expansion coefficients

        Returns:
            phi_hat (ndarray): approximate wave function

        """
        return self.phi_0 + np.sum(coefficients * self.vectors, axis=1)

    def project(self, x):
        r"""
        Return projection of x onto vectors
        """
        x -= self.phi_0
        x /= np.trapz(np.absolute(x), self.rho_mesh)
        return [
            np.trapz(self.vectors[:, i].conj() * x, self.rho_mesh)
            for i in range(self.n_basis)
        ]

    def percent_explained_variance(self):
        r"""
        Returns:
            (float) : percent of variance explained in the training set by the first n_basis principal
            components
        """
        if self.singular_values is None:
            return 100
        else:
            return np.array(
                [
                    100
                    * np.sum(self.singular_values[:i] ** 2)
                    / np.sum(self.singular_values**2)
                ]
                for i in range(self.nbasis)
            )


def pre_process_solutions(
    solutions,
    phi_0,
    rho_mesh,
    center=None,
    scale=None,
    svd=None,
    subtract_phi0=True,
):
    s = rho_mesh
    A = solutions

    if scale:
        phi_0 /= np.trapz(np.absolute(phi_0) ** 2, s)
        row_norms = np.array(
            [np.trapz(np.absolute(A[:, i]) ** 2, rho_mesh) for i in range(A.shape[1])]
        )
        A /= row_norms

    if center:
        mean = np.mean(A, axis=1)
        A -= mean[:, np.newaxis]
        phi_0 += mean

    if subtract_phi0:
        A -= phi_0[:, np.newaxis]

    if svd:
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        singular_values = S
        pillars = U
    else:
        singular_values = None
        pillars = A

    return pillars, singular_values, phi_0
