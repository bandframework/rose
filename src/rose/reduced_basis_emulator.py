'''
Defines a ReducedBasisEmulator.
'''
import numpy as np
import numpy.typing as npt

from .interaction import Interaction
from .schroedinger import SchroedingerEquation, DEFAULT_NUM_PTS
from .basis import RelativeBasis
from .constants import DEFAULT_RHO_MESH
from .free_solutions import phase_shift
from .utility import finite_difference_first_derivative, finite_difference_second_derivative

# How many points should be ignored at the beginning
# and end of the vectors (due to finite-difference
# inaccuracies)?
ni = 2

class ReducedBasisEmulator:
    '''
    A ReducedBasisEmulator (RBE) uses the specified interaction and theta_train
    to generate solutions to the Schrödinger equation at a specific energy
    (energy) and partial wave (l).

    Using the Galerkin projection method, a linear combination of those
    solutions (or a PCA basis of them) is found at some arbitrary point in
    parameter space, theta.
    '''
    def __init__(self,
        interaction: Interaction, # desired local interaction
        theta_train: npt.ArrayLike, # training points in parameter space
        energy: float, # center-of-mass energy (MeV)
        l: int, # angular momentum
        n_basis: int = 4, # How many basis vectors?
        use_svd: bool = True, # Use principal components as basis vectors?
        s_mesh: npt.ArrayLike = DEFAULT_RHO_MESH, # s = rho = kr; solutions are phi(s)
        s_0: float = None, # phase shift is "extracted" at s_0
        **kwargs # passed to SchroedingerEquation.solve_se
    ):
        self.energy = energy
        self.l = l
        self.se = SchroedingerEquation(interaction)

        if s_mesh is None:
            self.s_mesh = np.linspace(1e-6, 8*np.pi, DEFAULT_NUM_PTS)
        else:
            self.s_mesh = np.copy(s_mesh)

        if s_0 is None:
            s_0 =  6*np.pi

        # Index of the point in the s mesh that is closest to s_0.
        self.i_0 = np.argmin(np.abs(self.s_mesh - s_0))
        # We want to choose a point at which the solution has already been
        # calculated so we can avoid interpolation.
        self.s_0 = self.s_mesh[self.i_0]

        self.basis = RelativeBasis(
            self.se.interaction,
            theta_train,
            self.s_mesh,
            n_basis,
            self.energy,
            self.l,
            use_svd
        )

        # \tilde{U}_{bare} takes advantage of the linear dependence of \tilde{U}
        # on the parameters. The first column is multiplied by args[0]. The
        # second by args[1]. And so on. The "total" potential is the sum across
        # columns.
        self.utilde_basis_functions = self.se.interaction.basis_functions(self.s_mesh)

        # Precompute what we can for < psi | F(hat{phi}) >.
        d2_operator = finite_difference_second_derivative(self.s_mesh)
        phi_basis = self.basis.vectors
        self.d2 = -d2_operator @ phi_basis
        self.A_1 = phi_basis[ni:-ni].T @ self.d2[ni:-ni]
        self.A_2 = np.array([
            phi_basis[ni:-ni].T @ (row[:, np.newaxis] * phi_basis[ni:-ni]) for row in self.utilde_basis_functions[ni:-ni, :].T
        ])
        self.A_3 = phi_basis[ni:-ni].T @ -phi_basis[ni:-ni]

        # Precompute what we can for the inhomogeneous term ( -< psi | F(phi_0) > ).
        d2_phi_0 = d2_operator @ self.basis.phi_0
        self.b_1 = phi_basis[ni:-ni].T @ d2_phi_0[ni:-ni]
        self.b_2 = np.array([
            phi_basis[ni:-ni].T @ (-row * self.basis.phi_0[ni:-ni]) for row in self.utilde_basis_functions[ni:-ni].T
        ])
        self.b_3 = phi_basis[ni:-ni].T @ self.basis.phi_0[ni:-ni]

        # Can when extract the phase shift faster?
        self.phi_components = np.hstack(( self.basis.phi_0[:, np.newaxis], self.basis.vectors ))
        d1_operator = finite_difference_first_derivative(self.s_mesh)
        self.phi_prime_components = d1_operator @ self.phi_components
    

    def coefficients(self,
        theta: npt.ArrayLike
    ):
        A_utilde = np.sum([
            xi * Ai for (xi, Ai) in zip(self.se.interaction.coefficients(theta), self.A_2)
        ], axis=0)
        A = self.A_1 + A_utilde + self.A_3 # I should go ahead and store A_1 + A_3

        b_utilde = np.sum([
            xi * Ai for (xi, Ai) in zip(theta, self.b_2)
        ], axis=0)
        b = self.b_1 + b_utilde + self.b_3 # I should store b_1 + b_3.

        return np.linalg.solve(A, b)


    def emulate_wave_function(self,
        theta: npt.ArrayLike
    ):
        x = self.coefficients(theta)
        return self.basis.phi_hat(x)
    

    def emulate_phase_shift(self,
        theta: npt.ArrayLike
    ):
        x = self.coefficients(theta)
        phi = np.sum(np.hstack((1, x)) * self.phi_components[self.i_0, :])
        phi_prime = np.sum(np.hstack((1, x)) * self.phi_prime_components[self.i_0, :])
        return phase_shift(np.real(phi), np.real(phi_prime), self.l, self.s_mesh[self.i_0])