'''
Defines a ReducedBasisEmulator.
'''
import pickle
import numpy as np
import numpy.typing as npt

from .interaction import Interaction
from .schroedinger import SchroedingerEquation
from .basis import RelativeBasis, CustomBasis, Basis
from .constants import HBARC, DEFAULT_RHO_MESH
from .free_solutions import phase_shift, H_minus, H_plus, H_minus_prime, H_plus_prime
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
    @classmethod
    def load(obj, filename):
        with open(filename, 'rb') as f:
            rbe = pickle.load(f)
        return rbe


    @classmethod
    def from_train(cls,
        interaction: Interaction,
        theta_train: np.array, # training points in parameter space
        ell: int, # angular momentum
        n_basis: int = 4, # How many basis vectors?
        use_svd: bool = True, # Use principal components as basis vectors?
        s_mesh: np.array = DEFAULT_RHO_MESH, # s = rho = kr; solutions are phi(s)
        s_0: float = 6*np.pi, # phase shift is "extracted" at s_0
        hf_tols: list = None, # 2 numbers: high-fidelity solver tolerances, relative and absolute
    ):
        basis = RelativeBasis(
            SchroedingerEquation(interaction, hifi_tolerances=hf_tols),
            theta_train,
            s_mesh,
            n_basis,
            ell,
            use_svd
        )
        return cls(interaction, basis, ell, s_0=s_0)


    def __init__(self,
        interaction: Interaction,
        basis: Basis,
        ell: int,
        s_0: float = 6*np.pi # phase shift is "extracted" at s_0
    ):
        self.interaction = interaction
        self.basis = basis
        self.l = ell
        self.se = self.basis.solver

        self.s_mesh = np.copy(basis.rho_mesh)

        # Index of the point in the s mesh that is closest to s_0.
        self.i_0 = np.argmin(np.abs(self.s_mesh - s_0))
        # We want to choose a point at which the solution has already been
        # calculated so we can avoid interpolation.
        self.s_0 = self.s_mesh[self.i_0]

        # \tilde{U}_{bare} takes advantage of the linear dependence of \tilde{U}
        # on the parameters. The first column is multiplied by args[0]. The
        # second by args[1]. And so on. The "total" potential is the sum across
        # columns.
        self.utilde_basis_functions = self.interaction.basis_functions(self.s_mesh)

        # Precompute what we can for < psi | F(hat{phi}) >.
        d2_operator = finite_difference_second_derivative(self.s_mesh)
        phi_basis = self.basis.vectors
        ang_mom = self.l*(self.l+1) / self.s_mesh**2
        coulomb = 2*self.interaction.eta / self.s_mesh

        self.d2 = -d2_operator @ phi_basis
        self.A_1 = phi_basis[ni:-ni].T @ self.d2[ni:-ni]
        self.A_2 = np.array([
            phi_basis[ni:-ni].T @ (row[:, np.newaxis] * phi_basis[ni:-ni]) for row in self.utilde_basis_functions[ni:-ni, :].T
        ])
        self.A_3 = np.einsum('ij,j,jk',
                             phi_basis[ni:-ni].T,
                             coulomb[ni:-ni] + ang_mom[ni:-ni] - 1,
                             phi_basis[ni:-ni])
        # self.A_3 = phi_basis[ni:-ni].T @ -phi_basis[ni:-ni]

        # Precompute what we can for the inhomogeneous term ( -< psi | F(phi_0) > ).
        d2_phi_0 = d2_operator @ self.basis.phi_0
        self.b_1 = phi_basis[ni:-ni].T @ d2_phi_0[ni:-ni]
        self.b_2 = np.array([
            phi_basis[ni:-ni].T @ (-row * self.basis.phi_0[ni:-ni]) for row in self.utilde_basis_functions[ni:-ni].T
        ])
        self.b_3 = phi_basis[ni:-ni].T @ ((1 - ang_mom[ni:-ni] - coulomb[ni:-ni]) * self.basis.phi_0[ni:-ni])

        # Can we extract the phase shift faster?
        self.phi_components = np.hstack(( self.basis.phi_0[:, np.newaxis], self.basis.vectors ))
        d1_operator = finite_difference_first_derivative(self.s_mesh)
        self.phi_prime_components = d1_operator @ self.phi_components
    

    def coefficients(self,
        theta: np.array
    ):
        beta = self.interaction.coefficients(theta)

        A_utilde = np.einsum('i,ijk', beta, self.A_2)
        A = self.A_1 + A_utilde + self.A_3

        b_utilde = beta @ self.b_2
        b = self.b_1 + b_utilde + self.b_3

        return np.linalg.solve(A, b)


    def emulate_wave_function(self,
        theta: np.array
    ):
        x = self.coefficients(theta)
        return self.basis.phi_hat(x)
    

    def emulate_phase_shift(self,
        theta: np.array
    ):
        x = self.coefficients(theta)
        phi = np.sum(np.hstack((1, x)) * self.phi_components[self.i_0, :])
        phi_prime = np.sum(np.hstack((1, x)) * self.phi_prime_components[self.i_0, :])
        return phase_shift(phi, phi_prime, self.l, self.s_mesh[self.i_0])
    
    
    def logarithmic_derivative(self,
        theta: np.array
    ):
        a = self.s_mesh[self.i_0]
        x = self.coefficients(theta)
        phi = np.sum(np.hstack((1, x)) * self.phi_components[self.i_0, :])
        phi_prime = np.sum(np.hstack((1, x)) * self.phi_prime_components[self.i_0, :])
        return 1/a * phi / phi_prime
    
    
    def S_matrix_element(self,
        theta: np.array
    ):
        a = self.s_mesh[self.i_0]
        Rl = self.logarithmic_derivative(theta)
        return (H_minus(a, self.l) - a*Rl*H_minus_prime(a, self.l)) / \
            (H_plus(a, self.l) - a*Rl*H_plus_prime(a, self.l))


    def exact_phase_shift(self, theta: np.array):
        return self.se.delta(self.basis.solver.interaction.energy,
            theta, self.s_mesh[[0, -1]], self.l, self.s_0)
    

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)