"""Interactions that leverage the Empirical Interpolation Method (EIM) to allow
the emulation of parameters in which the coordinate-space potential is not
affine.
"""

from typing import Callable
import numpy as np
from scipy.stats import qmc

from .interaction import Interaction, InteractionSpace, couplings
from .constants import HBARC, DEFAULT_RHO_MESH
from .spin_orbit import SpinOrbitTerm
from .utility import latin_hypercube_sample, max_vol


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class InteractionEIM(Interaction):
    def __init__(
        self,
        training_info: np.array = None,
        n_basis: int = None,
        expl_var_ratio_cutoff: float = None,
        explicit_training: bool = False,
        n_train: int = 1000,
        rho_mesh: np.array = DEFAULT_RHO_MESH,
        match_points: np.array = None,
        method="collocation",
        **kwargs,
    ):
        r"""
        Parameters:
            training_info (ndarray): Either (1) parameters bounds or (2)
                explicit training points

                If (1):
                    This is a 2-column matrix. The first column are the lower
                    bounds. The second are the upper bounds. Each row maps to a
                    single parameter.

                If (2):
                    This is an MxN matrix. N is the number of parameters. M is
                    the number of samples.
            n_basis (int): min number of states in the expansion
            expl_var_ratio_cutoff (float) : the cutoff in sv**2/sum(sv**2), sv
                being the singular values, at which the number of kept bases is chosen
            explicit_training (bool): Is training_info (1) or (2)? (1) is
                default
            n_train (int): How many snapshots to generate? Ignored if
                explicit_training is True.
            rho_mesh (ndarray): coordinate-space points at which the interaction
                is generated (used for training)
            match_points (ndarray): $\rho$ points where agreement with the true
                potential is enforced
            method (str) : 'collocation' or 'least-squares'. If 'collocation',
                match_points must be the same length as n_basis; otherwise match_points
                can be any size.
            kwargs (dict): kwargs to `Interaction.__init__`

        Attributes:
            s_mesh (ndarray): $s$ points
            singular_values (ndarray): `S` in `U, S, Vt = numpy.linalg.svd(...)`
            snapshots (ndarray): pillars, columns of `U`
            match_indices (ndarray): indices of points in $\rho$ mesh that are
                matched to the true potential
            match_points (ndarray): points in $\rho$ mesh that are matched to
                the true potential
            r_i (ndarray): copy of `match_points` (???)
            Ainv (ndarray): inverse of A matrix (Ax = b)
        """
        assert training_info is not None

        if n_basis is None:
            if "n_theta" in kwargs:
                n_basis = kwargs["n_theta"]
            else:
                n_basis = 8

        super().__init__(**kwargs)

        self.method = method
        self.n_train = n_train
        self.n_basis = n_basis
        self.training_info = training_info
        self.s_mesh = rho_mesh

        # Generate a basis used to approximate the potential.
        # Did the user specify the training points?
        if explicit_training:
            snapshots = np.array(
                [self.tilde(rho_mesh, theta) for theta in training_info]
            ).T
        else:
            train = latin_hypercube_sample(n_train, training_info)
            snapshots = np.array([self.tilde(rho_mesh, theta) for theta in train]).T

        U, S, _ = np.linalg.svd(snapshots, full_matrices=False)

        # keeping at min n_basis PC's, find cutoff
        # avoids singular matrix in MAXVOL when we find a region of param
        # space w/ very similar potentials
        self.singular_values = S
        if expl_var_ratio_cutoff is not None:
            expl_var = self.singular_values**2 / np.sum(self.singular_values**2)
            n_basis_svs = np.sum(expl_var > expl_var_ratio_cutoff)
            self.n_basis = max(n_basis_svs, self.n_basis)
        else:
            self.n_basis = n_basis

        self.snapshots = U[:, : self.n_basis]
        self.match_points = match_points

        if match_points is not None and method == "collocation":
            self.n_basis = match_points.size
            self.match_points = match_points
            self.match_indices = np.array(
                [np.argmin(np.abs(rho_mesh - ri)) for ri in self.match_points]
            )
            self.r_i = rho_mesh[self.match_indices]
            self.Ainv = np.linalg.inv(self.snapshots[self.match_indices])
        elif match_points is None and method == "collocation":
            # random r points between 0 and 2π fm
            i_max = self.snapshots.shape[0] // 4
            di = i_max // (self.n_basis - 1)
            i_init = np.arange(0, i_max + 1, di)
            self.match_indices = max_vol(self.snapshots, i_init)
            self.match_points = rho_mesh[self.match_indices]
            self.r_i = self.match_points
            self.Ainv = np.linalg.inv(self.snapshots[self.match_indices])
        elif method == "least-squares":
            if match_points is None:
                self.match_points = rho_mesh
            else:
                self.match_points = match_points
            self.match_indices = np.array(
                [find_nearest_idx(self.s_mesh, x) for x in self.match_points]
            )
            self.match_points = self.s_mesh[self.match_indices]
            self.r_i = self.match_points
            self.Ainv = np.linalg.pinv(self.snapshots[self.match_indices])
        else:
            raise ValueError(
                "argument 'method' should be one of `collocation` or `least-squares`"
            )

    def coefficients(self, alpha: np.array):
        r"""Computes the EIM expansion coefficients.

        Parameters:
            alpha (ndarray): interaction parameters

        Returns:
            coefficients (ndarray): EIM expansion coefficients

        """
        u_true = self.tilde(self.r_i, alpha)
        return 1 / self.k, self.Ainv @ u_true

    def tilde_emu(self, alpha: np.array):
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

    def percent_explained_variance(self, n=None):
        r"""
        Returns:
            (float) : percent of variance explained in the training set by the first n_basis principal
            components
        """
        if n is None:
            n = self.n_basis
        return (
            100
            * np.sum(self.singular_values[:n] ** 2)
            / np.sum(self.singular_values**2)
        )


class InteractionEIMSpace(InteractionSpace):
    def __init__(
        self,
        l_max: int = 15,
        interaction_type=InteractionEIM,
        **kwargs,
    ):
        r"""Generates a list of $\ell$-specific, EIMed interactions.

        Parameters:
            interaction_args (list): positional arguments for constructor of `interaction_type`
            interaction_kwargs (dict): arguments to constructor of `interaction_type`
            l_max (int): maximum angular momentum
            interaction_type (Type): type of `Interaction` to construct

        Returns:
            instance (InteractionEIMSpace): instance of InteractionEIMSpace

        Attributes:
            interaction (list): list of `Interaction`s
            l_max (int): partial wave cutoff
            type (Type): interaction type
        """
        super().__init__(interaction_type=interaction_type, l_max=l_max, **kwargs)

    def percent_explained_variance(self, n=None):
        return [
            [
                interaction.percent_explained_variance(n)
                for interaction in interaction_list
            ]
            for interaction_list in self.interactions
        ]
